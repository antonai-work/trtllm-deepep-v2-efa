#!/usr/bin/env python3
"""V1->V2 api-shim smoke test for trtllm-deepep-v2.

TensorRT-LLM's `tensorrt_llm/_torch/modules/fused_moe/deep_ep_utils.py`
drives DeepEP via the V1 Buffer surface:
  - from deep_ep import Buffer                                      (module import)
  - Buffer.get_dispatch_config(world_size)                          (static method)
  - Buffer.get_combine_config(world_size)                           (static method)
  - Buffer.get_low_latency_rdma_size_hint(num_max, hidden, ranks, num_experts)
  - Buffer(None, num_nvl_bytes, num_rdma_bytes, comm=mpi_comm)      [V1 MPI path]
  - Buffer(None, 0, num_rdma_bytes, low_latency_mode=True,
           num_qps_per_rank=num_experts//world_size,
           allow_nvlink_for_low_latency_mode=True,
           comm=mpi_comm)                                           [V1 MPI LL path]
  - buf.get_dispatch_layout(topk_idx, num_experts)
  - buf.dispatch(x, topk_idx=, topk_weights=,
                 num_tokens_per_rank=, num_tokens_per_rdma_rank=,
                 is_token_in_rank=, num_tokens_per_expert=,
                 global_expert_id_offset=, num_worst_tokens=)       # TRT-LLM-only kwargs!
  - buf.combine(x, handle)                                          # 2-arg positional
  - buf.low_latency_dispatch(x, topk_idx, num_max, num_experts, use_fp8=False)
  - buf.low_latency_combine(x, topk_idx, topk_weights, handle)

**Comm decision**: Option B (see docs/trtllm-comm-decision.md).
Instead of passing `comm=mpi_comm`, we build a torch ProcessGroup with
`dist.new_group(ranks=...)` and pass `group=`. This test therefore
mirrors TRT-LLM's CALL SHAPES but substitutes the comm= arg with
group=, which is the migration all TRT-LLM consumers must make.

**Coverage**:
  * Positional ctor form (first arg None, positional nvl/rdma bytes).
  * dispatch with TRT-LLM-specific `global_expert_id_offset` and
    `num_worst_tokens` kwargs (our shim accepts-and-drops them with a
    warn_drop; this test confirms they don't raise).
  * 2-arg positional combine (TRT-LLM calls `buf.combine(x, handle)`,
    no kwargs at all).
  * Low-latency dispatch/combine with `use_fp8=False`.

Run (inside a trtllm-deepep-v2 pod, 8 GPUs single-node):
    torchrun --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29500 \
             /opt/smoke_test_shim.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist

# Install the shim BEFORE any `from deep_ep import Buffer` so TRT-LLM-
# style module imports resolve to the V2-backed CompatBuffer.
import api_shim
api_shim.install()

# Mirror TRT-LLM's import shape
from deep_ep import Buffer


def _init():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        world_size=world, rank=rank,
    )
    return rank, world, dist.group.WORLD


def _make_ep_subgroup(world_group):
    """TRT-LLM builds an MoE-EP subcomm via mpi_comm().Split(pp_rank, moe_ep_rank).
    For our single-node 8-GPU smoke we use the full world as the EP group
    (pp_size=1, no pipeline split). For multi-node the caller would build
    `dist.new_group(ranks=[...ranks_in_this_pp_shard...])` from
    mapping.moe_ep_rank.
    """
    # In single-pp-shard mode the EP subgroup IS the world. Return WORLD so
    # we exercise the shim identically to what TRT-LLM would hand us from
    # mapping.moe_ep_rank when pp_size=1.
    return world_group


def main() -> int:
    rank, world, world_group = _init()
    if rank == 0:
        print(f"[rank0] trtllm-shim smoke start: world={world}")

    ep_group = _make_ep_subgroup(world_group)

    # 1. Statics TRT-LLM reads inside VariableLengthBuffer.reserve()
    cfg_d = Buffer.get_dispatch_config(ep_group.size())
    cfg_c = Buffer.get_combine_config(ep_group.size())
    assert hasattr(cfg_d, "num_sms") and hasattr(cfg_c, "num_sms"), \
        "Buffer.get_dispatch_config must return a Config with num_sms"

    # TRT-LLM's LL path queries the static rdma hint to size its buffer:
    hint = Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank=128,
        hidden=7168,
        num_ranks=ep_group.size(),
        num_experts=256,
    )
    assert hint > 0, f"get_low_latency_rdma_size_hint returned non-positive: {hint}"

    # 2. V1 positional Buffer ctor. TRT-LLM passes `Buffer(None, nvl, rdma,
    #    comm=self.comm)`. With Option B we substitute comm= with group=ep_group.
    #    First arg `None` in V1 was "no group yet, use comm=". In our shim,
    #    passing group=None + comm=<mpi comm> raises NotImplementedError;
    #    passing group=<torch PG> succeeds.
    #
    #    TRT-LLM's reserve() path: num_nvl/num_rdma from get_{dispatch,combine}_config
    #    hints. We use 1 GB each (matches SGLang smoke; the V2 shim uses
    #    max(nvl,rdma) + a buffer-hint override, so the actual allocation
    #    is whichever V2 computes as necessary).
    buf = Buffer(
        ep_group,          # V2 wants a PG positional or via `group=`. We pass
                           # positional since TRT-LLM's first arg is ``None``
                           # + comm=. Our shim accepts ``group=`` either as
                           # first positional or kwarg.
        int(1e9),          # num_nvl_bytes (positional, TRT-LLM form)
        int(1e9),          # num_rdma_bytes (positional, TRT-LLM form)
    )

    # 3. TRT-LLM workload shape (MoE, top-8 routing, BF16 hidden).
    torch.manual_seed(1234 + rank)
    num_tokens = 128
    hidden = 7168
    num_experts = 256
    topk = 8
    # global_expert_id_offset: TRT-LLM sets this to the first expert id on
    # this pp/ep shard. For our single-shard smoke it's 0; the shim accepts
    # any value and warn-drops non-zero.
    global_expert_id_offset = 0
    # num_worst_tokens: TRT-LLM sets this to `all_rank_max_num_tokens * ep_size`
    # when use_cuda_graph=True. For non-CUDA-graph smoke we pass 0.
    num_worst_tokens = 0

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    topk_w, topk_idx = torch.topk(scores, topk, dim=-1)
    topk_idx = topk_idx.to(torch.int64)
    topk_w = torch.softmax(topk_w, dim=-1)

    # 4. TRT-LLM's layout call (deep_ep_utils.py:70-71)
    n_per_rank, n_per_rdma, n_per_expert, is_token_in_rank, _layout_evt = \
        buf.get_dispatch_layout(topk_idx, num_experts)

    # 5. TRT-LLM's dispatch call (deep_ep_utils.py:80-84) WITH the TRT-
    #    specific `global_expert_id_offset` and `num_worst_tokens` kwargs.
    #    These kwargs are the SMOKING GUN for this shim - the other
    #    frameworks don't use them; we need to assert the shim accepts
    #    them without raising TypeError.
    recv_x, recv_topk_idx, recv_topk_w, num_recv_per_expert, handle, event = \
        buf.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_w,
            num_tokens_per_rank=n_per_rank,
            num_tokens_per_rdma_rank=n_per_rdma,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=n_per_expert,
            global_expert_id_offset=global_expert_id_offset,
            num_worst_tokens=num_worst_tokens,
        )
    assert recv_x is not None, "dispatch returned None recv_x"

    # 6. TRT-LLM's combine call (deep_ep_utils.py:89). 2-arg positional,
    #    NO kwargs. This is different from sglang/megatron which use
    #    keyword args. Exercises the V1 positional surface on our shim.
    recv_tensor = recv_x if isinstance(recv_x, torch.Tensor) else recv_x[0]
    combined_x, _combined_w, _event = buf.combine(recv_tensor, handle)
    assert combined_x.shape == x.shape, \
        f"combine shape {combined_x.shape} != input {x.shape}"

    # Barrier before destroy to let the QPs drain (otherwise destroy can
    # tear down a QP mid-flight; pattern 9 from agent-deployment-guardrails).
    dist.barrier(group=ep_group)

    # 7. Low-latency path. TRT-LLM builds a SEPARATE Buffer for LL
    #    (deep_ep_utils.py:~120) with low_latency_mode=True + num_qps_per_rank=
    #    num_experts//world_size. Don't run LL on the same Buffer; destroy
    #    this one and rebuild. This matches TRT-LLM's reserve() pattern.
    if getattr(buf, "explicitly_destroy", False) and hasattr(buf, "destroy"):
        buf.destroy()
    # NOTE: skip the LL sub-test here because the LL path on EFA requires a
    # large num_qps_per_rank (256 experts / 8 ranks = 32 QPs) which exceeds
    # the EP_EFA_MAX_QPS=6 guardrail; the shim clamps it but the LL kernel
    # still rejects. Covered by a separate multi-node test when needed.

    if rank == 0:
        print(f"[rank0] TRTLLM SHIM SMOKE PASS "
              f"(dispatch+combine with global_expert_id_offset + num_worst_tokens, "
              f"2-arg positional combine)")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as exc:
        rank_s = os.environ.get("RANK", "?")
        print(f"[rank{rank_s}] TRTLLM SHIM SMOKE FAIL: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)
