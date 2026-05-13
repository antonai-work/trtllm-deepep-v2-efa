"""V1 `deep_ep.Buffer` compatibility facade over V2 `deep_ep.ElasticBuffer`.

Goal: any customer code that imports `deep_ep.Buffer` (V1 API) keeps
working unchanged on the V2 NCCL-Gin backend. The call sites surveyed:

    integrations docs/v1-buffer-call-sites.md

Covered frameworks:
    - vLLM 0.19.1 (HT + LL paths)
    - SGLang 0.5.6+
    - Megatron-LM main
    - TensorRT-LLM 0.20 (MPI comm path + TRT-specific extensions)
    - NeMo-RL (via Megatron shim)
    - LLM-D (orchestrates vLLM; shimming vLLM is enough)

Usage (add to framework entrypoint BEFORE any `from deep_ep import Buffer`):

    import api_shim
    api_shim.install()

Semantics:
    * V1 "normal" (high-throughput) path -> V2 ElasticBuffer.dispatch /
      ElasticBuffer.combine. Return-tuple arity preserved.
    * V1 low-latency path -> V2 ElasticBuffer.dispatch with
      allow_hybrid_mode=False. Packed-expert output shape reshaped to
      match V1 contract. This is the high-risk mapping; unit test via
      scripts/shared/test_shim_v1_surface.py.
    * Buffer stores the group/bytes at construction. ElasticBuffer is
      constructed LAZILY on first dispatch once we learn the shapes.
      V1 accepts ctor-time bytes; V2 accepts bytes OR (shape-based) MoE
      construction; we pass through bytes and get MoE shapes from the
      first dispatch call.

Unmapped surfaces (raise NotImplementedError with a migration hint):
    * TRT-LLM `low_latency_dispatch_fp4`, `low_latency_combine_low_precision`
    * V1 `use_nvfp4` + `x_global_scale` LL kwargs (V2's fp4 path differs)
    * V1 `comm=mpi4py.MPI.Comm` (V2 requires torch.distributed.ProcessGroup)
    * V1 `return_recv_hook=True` with actual deferred hook (V2 has
      async_with_compute_stream + event; the shim returns a no-op callable
      that is safe to call but does not defer anything)

See docs/V1-to-V2-API-migration.md for per-kwarg semantics and docs/
v1-buffer-call-sites.md for exact framework surface surveyed.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

import deep_ep
from deep_ep import ElasticBuffer  # V2 canonical entry point

# Re-exported V2 utility aliases so `from deep_ep.utils import EventHandle,
# EventOverlap` (Megatron) continues to work. deep_ep package exposes these
# under deep_ep.utils in V2; we re-point to V2's event classes.
try:
    from deep_ep.utils import EventHandle  # V2 has this in utils
    from deep_ep.utils import EventOverlap  # V2 has this in utils
except ImportError:
    # Fallback: minimal shims. Only Megatron constructs these directly.
    class EventHandle:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._event = torch.cuda.current_stream().record_event()

    class EventOverlap:  # type: ignore[no-redef]
        def __init__(self, event: Any = None, tensors_to_record: Any = None):
            self.event = event


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _warn_drop(flag: str, value: Any) -> None:
    """Warn-and-drop a V1 flag that has no V2 equivalent but is safe to
    silently ignore. Keeps customer code running."""
    if value not in (None, False, 0):
        warnings.warn(
            f"[api_shim] V1 Buffer kwarg '{flag}={value}' has no V2 equivalent "
            f"and is being silently ignored; see docs/V1-to-V2-API-migration.md",
            RuntimeWarning,
            stacklevel=3,
        )


def _to_bytes(num_nvl: int, num_rdma: int) -> Optional[int]:
    # V2 ElasticBuffer allocates ONE buffer covering both intranode + internode.
    # V1 Buffer allocated TWO (NVL + RDMA). LLM consensus (Gemini, Claude)
    # recommends max() not sum() because V2 uses it as a single pool both
    # directions can draw from — sum would over-allocate memory.
    total = max(num_nvl, num_rdma, 0)
    return total if total > 0 else None


# -----------------------------------------------------------------------------
# Module-level V1 symbols that frameworks import directly
# -----------------------------------------------------------------------------

class Config:
    """V1 `deep_ep.Config` shim. Frameworks (SGLang, vLLM HT) call this
    with 3 positional args `(num_sms, nvl_chunk_size, nvl_buffer_size)` or
    with the 5-arg form including rdma params. V2 moved `num_sms` / `num_qps`
    into per-call kwargs on `dispatch`/`combine`, so we capture num_sms and
    the rest is dropped as advisory."""

    def __init__(self,
                 num_sms: int = 0,
                 nvl_chunk_size: int = 0,
                 nvl_buffer_size: int = 0,
                 rdma_chunk_size: int = 0,
                 rdma_buffer_size: int = 0) -> None:
        self.num_sms = num_sms
        self.nvl_chunk_size = nvl_chunk_size
        self.nvl_buffer_size = nvl_buffer_size
        self.rdma_chunk_size = rdma_chunk_size
        self.rdma_buffer_size = rdma_buffer_size

    # V1 method SGLang calls on Config objects to size the NVL ring pool.
    # V2's ElasticBuffer owns the pool internally; this is advisory and
    # we return the buffer_size (or a sane default) so the V1 caller
    # can plumb it into the Buffer ctor. Either arg set works:
    #   get_nvl_buffer_size_hint(num_ranks, hidden)             SGLang 0.5.6
    #   get_nvl_buffer_size_hint(num_max_tokens, hidden, num_topk) SGLang main
    def get_nvl_buffer_size_hint(self, *args: Any, **kwargs: Any) -> int:
        if self.nvl_buffer_size > 0:
            return int(self.nvl_buffer_size)
        return int(512 * 1024 * 1024)  # 512 MB default matches V1

    def get_rdma_buffer_size_hint(self, *args: Any, **kwargs: Any) -> int:
        if self.rdma_buffer_size > 0:
            return int(self.rdma_buffer_size)
        return int(512 * 1024 * 1024)

    # V1 Config's hint methods for ring counts (SGLang uses these too).
    def get_nvl_buffer_count_hint(self, *args: Any, **kwargs: Any) -> int:
        return int(self.nvl_chunk_size) if self.nvl_chunk_size > 0 else 8

    def get_rdma_buffer_count_hint(self, *args: Any, **kwargs: Any) -> int:
        return int(self.rdma_chunk_size) if self.rdma_chunk_size > 0 else 256


# -----------------------------------------------------------------------------
# Main compatibility class
# -----------------------------------------------------------------------------

class CompatBuffer:
    """V1 `Buffer`-compatible facade over V2 `ElasticBuffer`.

    Construction is lazy: we record the byte budget at __init__ time, and
    create the underlying V2 ElasticBuffer on first dispatch call (at which
    point we have the tensor shapes needed). This keeps the V1 surface
    (constructor knows only the buffer size) while satisfying V2's
    shape-or-bytes constructor contract.
    """

    # V1 class-level attribute; some frameworks read it as a fallback for
    # num_sms before kernel launch.
    num_sms: int = 20

    def __init__(
        self,
        group: Optional[dist.ProcessGroup] = None,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        use_fabric: bool = False,
        explicitly_destroy: bool = False,
        enable_shrink: bool = False,
        comm: Any = None,
        **_unknown: Any,
    ) -> None:
        if _unknown:
            warnings.warn(
                f"[api_shim] Unknown V1 Buffer kwargs ignored: {list(_unknown)}",
                RuntimeWarning, stacklevel=2,
            )
        if group is None:
            if comm is not None:
                # TRT-LLM passes mpi4py.MPI.Comm. V2 requires torch.distributed
                # ProcessGroup. Fail early with a migration hint.
                raise NotImplementedError(
                    "[api_shim] CompatBuffer received mpi4py comm= instead of "
                    "torch.distributed.ProcessGroup. V2 ElasticBuffer requires "
                    "a torch PG. TRT-LLM consumers: construct a torch PG from "
                    "the MPI comm using torch.distributed.ProcessGroupMPI or "
                    "bootstrap via torch.distributed.init_process_group on the "
                    "MPI-assigned ranks before calling Buffer()."
                )
            raise ValueError(
                "[api_shim] Buffer requires a torch.distributed.ProcessGroup "
                "via `group=` (V1 also allowed `comm=` but V2 does not)."
            )

        # vLLM's DeepEPHTAll2AllManager passes `cpu_group` (a gloo PG)
        # into Buffer(group=...). DeepEP V2 requires a GPU NCCL group
        # (get_nccl_comm_handle extracts the NCCL communicator). Using
        # a gloo group leaves the Gin cross-node barrier unable to
        # deliver signals -> tag-6 Gin barrier timeout at devCommCreate.
        # Fix: if the supplied group is gloo, substitute the WORLD NCCL
        # group (which vLLM has already bootstrapped for DP handshake).
        try:
            if dist.is_initialized() and group is not None:
                backend = None
                try:
                    backend = dist.get_backend(group)
                except Exception:
                    backend = None
                if backend and backend.lower() in ("gloo", "mpi"):
                    nccl_group = dist.group.WORLD
                    if dist.get_backend(nccl_group).lower() == "nccl":
                        warnings.warn(
                            f"[api_shim] Got {backend!r} group for DeepEP "
                            f"Buffer; substituting NCCL WORLD group so "
                            f"cross-node Gin transport is available.",
                            RuntimeWarning, stacklevel=2,
                        )
                        group = nccl_group
        except Exception:
            pass

        self.group = group
        self.rank = group.rank()
        self.group_size = group.size()
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_qps_per_rank = num_qps_per_rank
        self.explicitly_destroy = explicitly_destroy

        _warn_drop("allow_nvlink_for_low_latency_mode",
                   not allow_nvlink_for_low_latency_mode)  # only warn if non-default False
        _warn_drop("allow_mnnvl", allow_mnnvl)
        _warn_drop("use_fabric", use_fabric)
        _warn_drop("enable_shrink", enable_shrink)

        # Eager V2 construction. V2 accepts num_bytes directly (bypasses
        # MoE-shape path), so we don't need to defer.
        # low_latency_mode=True -> force all-RDMA via allow_hybrid_mode=False
        # (pr521gin v2 recipe).
        total_bytes = _to_bytes(num_nvl_bytes, num_rdma_bytes)

        # V2 asserts buffer.hpp:899 that the allocated buffer fits at
        # least get_dispatch_buffer_size(...) for the runtime workload.
        # Framework callers supply bytes based on their own heuristics
        # (vLLM: VLLM_DEEPEP_BUFFER_SIZE_MB=64 default ~2-8 MB each rank),
        # which was tuned for V1 and can under-provision V2 (which needs
        # separate send/recv + hybrid scratch). Ask V2 for its recommended
        # size and pick the larger of user-supplied and hint.
        try:
            hint_bytes = ElasticBuffer.get_buffer_size_hint(
                group=group,
                num_max_tokens_per_rank=256,   # sane ceiling for most MoE
                hidden=7168,                   # conservative upper bound
                num_topk=8,
                use_fp8_dispatch=False,
                allow_hybrid_mode=not low_latency_mode,
                allow_multiple_reduction=True,
            )
            if total_bytes is None or hint_bytes > total_bytes:
                total_bytes = hint_bytes
        except Exception as _hint_err:
            # Best-effort. If the hint call fails we fall back to the
            # user-supplied bytes (may still succeed on small workloads).
            pass

        # EFA QP safety cap. V2's elastic.py:227 only clamps num_allocated_qps
        # to _EFA_MAX_QPS=2 when num_allocated_qps==0 (auto mode). Framework
        # callers (vLLM HT: num_sms//2=10, SGLang: 24, Megatron: 24) pass
        # explicit values that bypass that gate and over-subscribe
        # aws-ofi-nccl's 128-slot shared GIN ring, tripping CUDA 719 at
        # dispatch.hpp:183 / combine.hpp:95 (see docs/dispatch-hpp-183-
        # analysis/root-cause.md). Pass 0 so V2 auto-sizes AND hits its
        # own EFA cap branch.
        import os
        if os.environ.get("FI_PROVIDER", "") == "efa" or os.environ.get("DEEP_EP_BACKEND", "") == "nccl":
            _effective_num_qps = 0
        else:
            _effective_num_qps = num_qps_per_rank

        # Pre-construction barrier. V2's ElasticBuffer.__init__ calls
        # nccl.cu devCommCreate which runs an internal tag-6 Gin barrier
        # across all ranks. With vLLM DP=16 split 8/8 over two nodes,
        # each vLLM worker constructs its own Buffer independently; if
        # the workers don't all enter devCommCreate at the same wall-
        # clock window, the internal Gin barrier times out at 100s waiting
        # for cross-node signal -> tag:6 Gin barrier timeout + CUDA
        # 719 at dispatch.hpp:183.
        # Fix: torch.distributed.barrier() on the same group we're about
        # to hand V2 so all ranks synchronize BEFORE devCommCreate starts.
        try:
            if dist.is_initialized() and group is not None:
                dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        except Exception as _bar_err:
            # If device_ids isn't supported by the torch version, fall
            # back to untyped barrier (costs one torch.cuda.synchronize).
            try:
                torch.cuda.synchronize()
                dist.barrier(group=group)
            except Exception:
                pass

        # GPU-side Gin barrier timeout. V2 default is 100s, which is too
        # short when framework workers (vLLM DP=16 MultiprocExecutor)
        # enter Buffer() staggered by weight-load time or compilation
        # stalls. Raise to 600s so the barrier tolerates wall-clock
        # skew between cross-node peers. Override via EP_GPU_TIMEOUT_SECS.
        _gpu_timeout = int(os.environ.get("EP_GPU_TIMEOUT_SECS", "600"))
        _cpu_timeout = int(os.environ.get("EP_CPU_TIMEOUT_SECS", "900"))

        # Ctor shape selection. Proven 2026-04-29 on 2-node EFA
        # (docs/pr41183-pattern-validated.md): V2's MoE-shape ctor path
        # (num_max_tokens_per_rank+hidden+num_topk) successfully
        # registers hybrid-mode Gin transports at construction time,
        # while the byte-pool ctor (num_bytes=total_bytes) does NOT — it
        # hangs at the tag-6 Gin cross-node barrier during devCommCreate.
        # Prefer the MoE-shape ctor when the env var says to
        # (DEEP_EP_SHIM_MOE_CTOR=1); default on for EFA fabric.
        _prefer_moe_ctor = os.environ.get(
            "DEEP_EP_SHIM_MOE_CTOR",
            "1" if os.environ.get("FI_PROVIDER", "") == "efa" else "0",
        ) == "1"

        # Pinned ceiling for num_max_tokens_per_rank — load-bearing.
        # V2's csrc/kernels/elastic/dispatch.hpp:138,150 fmt-substitutes
        # `num_max_tokens_per_rank` into the JIT template instantiation
        # string (`hybrid_dispatch_impl<..., N, ...>`). If ranks call
        # dispatch with different per-batch token counts, they compile
        # different kernel binaries with different channel layouts; the
        # kHybridDispatchTag0 prologue barrier (hybrid_dispatch.cuh:82)
        # then polls signal slots that peers are writing to different
        # offsets and hangs with `signal: 1, target: 2`. Pin at ctor,
        # reuse on every dispatch call. Matches the reference harness
        # tests/elastic/test_ep.py:62 which fixes it once at test start.
        _mt = int(os.environ.get("DEEP_EP_SHIM_MAX_TOKENS_PER_RANK", "8192"))
        _h = int(os.environ.get("DEEP_EP_SHIM_HIDDEN", "7168"))
        _tk = int(os.environ.get("DEEP_EP_SHIM_NUM_TOPK", "8"))
        self._shim_num_max_tokens_per_rank = _mt
        self._shim_hidden = _h
        self._shim_num_topk = _tk

        if _prefer_moe_ctor:
            # Sane MoE defaults; frameworks that need smaller should set
            # DEEP_EP_SHIM_MAX_TOKENS_PER_RANK / DEEP_EP_SHIM_HIDDEN env
            # vars explicitly.
            self._elastic: ElasticBuffer = ElasticBuffer(
                group=group,
                num_max_tokens_per_rank=_mt,
                hidden=_h,
                num_topk=_tk,
                use_fp8_dispatch=False,
                allow_hybrid_mode=not low_latency_mode,
                num_allocated_qps=_effective_num_qps,
                num_cpu_timeout_secs=_cpu_timeout,
                num_gpu_timeout_secs=_gpu_timeout,
                explicitly_destroy=explicitly_destroy,
            )
        else:
            self._elastic: ElasticBuffer = ElasticBuffer(
                group=group,
                num_bytes=total_bytes,
                allow_hybrid_mode=not low_latency_mode,
                num_allocated_qps=_effective_num_qps,
                num_cpu_timeout_secs=_cpu_timeout,
                num_gpu_timeout_secs=_gpu_timeout,
                explicitly_destroy=explicitly_destroy,
            )

        # Post-construction barrier. After all ranks' devCommCreate
        # returns, ensure every rank has completed setup before the
        # first dispatch call (V1 callers don't synchronize across
        # Buffer() construction, but V2's next dispatch implicitly
        # assumes the buffer is globally live).
        try:
            if dist.is_initialized() and group is not None:
                dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        except Exception:
            try:
                torch.cuda.synchronize()
                dist.barrier(group=group)
            except Exception:
                pass

        # Cached handle for repeated dispatch with the same shapes (V2
        # optimization; V1 also reuses via `handle=` arg so this is a
        # natural match).
        self._cached_handle: Optional[Any] = None

    # -----------------------------------------------------------------
    # Static / class methods that frameworks call
    # -----------------------------------------------------------------

    @staticmethod
    def is_sm90_compiled() -> bool:
        # V1 exposes this; V2 has no equivalent. All deployments so far
        # are sm90 H100 or sm100 B200; best-effort answer from the runtime.
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9

    @classmethod
    def set_num_sms(cls, new_num_sms: int) -> None:
        assert new_num_sms % 2 == 0, "num_sms must be even (V1 contract)"
        cls.num_sms = new_num_sms

    @staticmethod
    def capture() -> "EventOverlap":
        # V1: wraps the current stream's recorded event.
        # V2: ElasticBuffer.capture() returns an EventHandle directly.
        try:
            handle = ElasticBuffer.capture()
            return EventOverlap(handle)
        except Exception:
            return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        # V2 ElasticBuffer has `get_buffer_size_hint` (with different sig).
        # Use it if available; else fall back to V1's formula (same as
        # DeepEP upstream).
        try:
            # V2 static: (group, num_max_tokens_per_rank, hidden, num_topk, ...)
            # We don't have a `group` at this point (static call), so use the
            # V1 formula directly. V1 formula matches V2 for low-latency mode.
            num_scales = max(hidden // 128, 1)
            per_token_bytes = hidden * 2 + num_scales * 4  # bf16 token + fp32 scales
            return num_max_dispatch_tokens_per_rank * num_ranks * num_experts * per_token_bytes
        except Exception:
            # Generous upper bound
            return int(2 * 1024 * 1024 * 1024)

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        # V1 returns a tuned Config. V2 passes num_sms/num_qps per call.
        # We return a default Config with num_sms=0 (auto) so the shim's
        # dispatch path picks V2 defaults.
        return Config(num_sms=0, nvl_chunk_size=0, nvl_buffer_size=0)

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        return Config(num_sms=0, nvl_chunk_size=0, nvl_buffer_size=0)

    # -----------------------------------------------------------------
    # Instance attributes / introspection
    # -----------------------------------------------------------------

    @property
    def runtime(self) -> Any:
        """V1 exposed `buf.runtime`; some callers poke it for diagnostic
        info. Return V2's underlying C++ runtime."""
        return self._elastic.runtime

    # -----------------------------------------------------------------
    # Dispatch / combine (normal / high-throughput V1 path)
    # -----------------------------------------------------------------

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional["EventOverlap"] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor], "EventOverlap"]:
        """V2 `ElasticBuffer.dispatch` computes layout internally and
        doesn't expose a separate layout call. We return sentinels so
        V1 callers that then pass them BACK into `dispatch` work — our
        shim's `dispatch` then ignores them and lets V2 recompute.

        Returns: (num_tokens_per_rank, num_tokens_per_rdma_rank,
                  num_tokens_per_expert, is_token_in_rank, event)
        The first four are `None`; V2 derives them from topk_idx inside
        dispatch. Empty event wraps current stream.

        Side effect: cache `num_experts` on self so the subsequent
        dispatch() call can retrieve it (V1 `dispatch` signature doesn't
        carry num_experts, but V2's buffer.hpp:733 asserts
        num_experts % num_ranks == 0, so an accurate value is required).
        """
        self._num_experts_hint = int(num_experts)
        # Wave 34.14: TRT-LLM 0.21 VariableLengthBuffer.dispatch at
        # deep_ep_utils.py:71 asserts `event.event is None` on sync path.
        # V2 ElasticBuffer returns EventOverlap with .event populated;
        # explicitly null it for the sync contract.
        evt = EventOverlap(EventHandle()) if async_finish else EventOverlap()
        try:
            evt.event = None
        except Exception:
            pass
        return None, None, None, None, evt

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Any] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional["EventOverlap"] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        # TRT-LLM extensions (ignored on V2 for now)
        global_expert_id_offset: int = 0,
        **_: Any,
    ) -> Tuple[Any, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[List[int]], Any, "EventOverlap"]:
        """V1 normal dispatch -> V2 ElasticBuffer.dispatch.

        V1 returns (recv_x, recv_topk_idx, recv_topk_weights,
                    num_recv_tokens_per_expert_list, handle, event)
        V2 returns (recv_x, recv_topk_idx, recv_topk_weights,
                    EPHandle, event) — note V2 has no
                    `num_recv_tokens_per_expert_list` explicitly, it's
                    stored inside the handle as
                    `handle.num_recv_tokens_per_expert_list`.

        We unpack and return the V1 shape.
        """
        # V1 layout tensors are ignored; V2 infers from topk_idx.
        _warn_drop("num_tokens_per_rank (V2 ignores)", num_tokens_per_rank is not None)
        _warn_drop("num_worst_tokens", num_worst_tokens)
        _warn_drop("global_expert_id_offset (TRT-LLM)", global_expert_id_offset)

        # NOTE: The previous empty-batch short-circuit (first_x.shape[0]==0
        # returned here without touching V2) was REMOVED 2026-04-29 after
        # tracer sub-agent (Opus) analysis of
        # hybrid_dispatch.cuh:82 + common/comm.cuh:135-181. The V2 hybrid
        # dispatch kernel starts with a collective gin_barrier on tag
        # kHybridDispatchTag0 (6) across all scaleup AND scaleout peers.
        # A rank that short-circuits never calls gin.signal on that tag,
        # and its peers wait forever — the exact `signal: 1, target: 2`
        # tag-6 timeout + all-zero CPU recv counts seen on Run #5.
        # With `num_max_tokens_per_rank` pinned below (architect Opus
        # analysis), the JIT template instantiation is identical across
        # ranks even when per-rank token count is zero, so every rank
        # enters the same kernel binary and participates in the barrier.
        first_x = x if isinstance(x, torch.Tensor) else x[0]

        # Derive num_experts. Priority order:
        # 1. Explicit kwarg `num_experts=` (vLLM's _do_dispatch passes it).
        # 2. `_num_experts_hint` cached from most recent get_dispatch_layout
        #    call (vLLM / SGLang / Megatron all call get_dispatch_layout
        #    right before dispatch; that call carries the true num_experts).
        # 3. num_tokens_per_expert tensor length (V1 normal path contract).
        # 4. topk_idx.max()+1 (fallback; undercounts if not all experts
        #    receive tokens — dangerous for V2's get_theoretical_num_sms
        #    which needs num_experts % num_groups == 0, and for
        #    buffer.hpp:733 which asserts num_experts % num_ranks == 0).
        num_experts = _.get('num_experts')
        if num_experts is None:
            num_experts = getattr(self, '_num_experts_hint', None)
        if num_experts is None and num_tokens_per_expert is not None:
            num_experts = num_tokens_per_expert.numel()
        if num_experts is None and topk_idx is not None:
            num_experts = int(topk_idx.max().item()) + 1

        num_sms = config.num_sms if (config is not None and config.num_sms > 0) else 4

        # Pin num_max_tokens_per_rank to the construction-time ceiling,
        # NOT the per-batch x.shape[0]. This is the load-bearing fix
        # identified by the 4-agent analysis on 2026-04-29 (Opus
        # architect trace of dispatch.hpp:138,150). `num_max_tokens_per_rank`
        # is a non-type template parameter baked into V2's JIT kernel
        # binary name. Floating it per-batch across ranks produces
        # different `hybrid_dispatch_impl<..., N, ...>` specializations
        # with incompatible channel layouts, and the gin barrier slots
        # don't line up — causing the `signal: 1, target: 2` tag-6 hang
        # on the 2nd+ dispatch. Match tests/elastic/test_ep.py:62 which
        # fixes the value once at test start.
        num_max_tokens = int(self._shim_num_max_tokens_per_rank)
        _cur_tokens = first_x.shape[0]
        assert _cur_tokens <= num_max_tokens, (
            f"[api_shim] batch {_cur_tokens} exceeds pinned "
            f"num_max_tokens_per_rank={num_max_tokens}; raise "
            f"DEEP_EP_SHIM_MAX_TOKENS_PER_RANK in the pod env.")

        # Per 3-model consensus (docs/deep-code-trace-playbook.md): V2's
        # hybrid_dispatch and hybrid_combine kernels have an unguarded
        # red_add_rel handshake that traps on zero-token ranks OR
        # unbalanced scaleout signals, surfacing as Gin barrier tag-6
        # (dispatch) or tag-8 (combine) timeout + CUDA 719 sticky. PR
        # #41183 avoids this STRUCTURALLY by using do_expand=True —
        # which routes through a different V2 code path that doesn't
        # have the defect. Mirror that here: pass do_expand=True.
        # Also attach previous_event via buffer.capture() to close the
        # stream-event handoff race Gemini/GPT-5.4 flagged.
        # V2 contract (buffer.hpp:483): if previous_event is set,
        # allocate_on_comm_stream MUST be True. So we flip both on together.
        # Seed previous_event unconditionally when caller didn't supply one:
        # PR #41183 (deepep_v2.py:99) always seeds via dbo_get_previous_event,
        # never gated on async_finish. Our prior gate (if ... and async_finish)
        # meant vLLM's default async_finish=False path never got the capture —
        # defeating commit 56d9218's purpose. Audit flagged this as BLOCKER.
        _prev_evt = getattr(previous_event, 'event', None)
        _alloc_on_comm = allocate_on_comm_stream
        if _prev_evt is None:
            try:
                _prev_evt = self._elastic.capture()
                _alloc_on_comm = True  # V2 contract
            except Exception:
                _prev_evt = None

        # do_expand=False per pr521gin/v2 reference harness (tests/elastic/
        # test_ep.py cycles BOTH modes but reference bench runs at 740us
        # p50 with default do_expand=False). Run #6 revealed CUDA 719 at
        # csrc/jit/handle.hpp:86 when caller (vLLM _do_combine) supplies
        # non-expanded tokens into a handle wired for expanded layout.
        # Paths diverge:
        #   do_expand=True  -> recv_x is [N_expanded, hidden], one slot
        #                      per expert per token; combine must receive
        #                      same expanded layout.
        #   do_expand=False -> recv_x is [N_unique, hidden], standard
        #                      V1-compatible layout; combine receives
        #                      same layout. Matches what vLLM HT supplies.
        # Upstream reference (Issue #604 + elastic.py:58 + test_ep.py:62)
        # confirm do_expand=False is the correct path for HT dispatch.
        recv_x, recv_topk_idx, recv_topk_weights, v2_handle, event = \
            self._elastic.dispatch(
                x,
                topk_idx=topk_idx if handle is None else None,
                topk_weights=topk_weights if handle is None else None,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_tokens,
                expert_alignment=(expert_alignment if expert_alignment != 1 else None),
                num_sms=num_sms,
                num_qps=0,  # V2 auto
                previous_event=_prev_evt,
                async_with_compute_stream=async_finish,
                allocate_on_comm_stream=_alloc_on_comm,
                handle=handle,
                do_handle_copy=True,
                do_expand=False,
            )

        # Extract num_recv_tokens_per_expert_list from the V2 handle. This
        # is the V1 4th-position return value that frameworks consume.
        num_recv_per_expert: Optional[List[int]] = None
        if v2_handle is not None and hasattr(v2_handle, 'num_recv_tokens_per_expert_list'):
            per_expert = v2_handle.num_recv_tokens_per_expert_list
            if isinstance(per_expert, torch.Tensor):
                num_recv_per_expert = per_expert.tolist()
            elif isinstance(per_expert, list):
                num_recv_per_expert = per_expert

        # Reconstruct recv_topk_idx when V2 returns None (happens with
        # do_expand=True). V1 callers like vLLM's deepep_ht.py:213 assert
        # `expert_topk_ids is not None`. Mirror PR #41183's
        # DeepEPV2PrepareAndFinalize._receiver logic: build from
        # num_recv_tokens_per_expert_list + rank_expert_offset.
        first_x_is_tensor = isinstance(recv_x, torch.Tensor)
        recv_tensor = recv_x if first_x_is_tensor else recv_x[0]
        if recv_topk_idx is None and num_recv_per_expert is not None:
            rank_expert_offset = self.rank * (
                (num_experts or len(num_recv_per_expert)) // self.group_size
            ) if num_experts else 0
            if recv_tensor.numel() > 0:
                rebuilt = torch.cat([
                    torch.full(
                        (count,), i + rank_expert_offset,
                        dtype=torch.int64, device=recv_tensor.device,
                    )
                    for i, count in enumerate(num_recv_per_expert)
                    if count > 0
                ]) if any(c > 0 for c in num_recv_per_expert) else torch.empty(
                    0, dtype=torch.int64, device=recv_tensor.device,
                )
                recv_topk_idx = rebuilt.unsqueeze(1)
            else:
                recv_topk_idx = torch.empty(
                    (0, 1), dtype=torch.int64, device=recv_tensor.device,
                )

        # Normalize recv_topk_weights to 2D [N, topk].
        # With do_expand=True, V2 returns 1D [N] (per-slot scalar) —
        # see csrc/elastic/buffer.hpp:1043. vLLM/flashinfer's
        # `token_final_scales` check (flashinfer_cutlass_fused_moe) asserts
        # `ndim == 2`. Mirror that contract by unsqueezing the trailing dim.
        if recv_topk_weights is not None and recv_topk_weights.dim() == 1:
            recv_topk_weights = recv_topk_weights.unsqueeze(-1)
        # If V2 returned None (topk_weights wasn't supplied), fabricate
        # ones shaped to match recv_topk_idx (V1 consumers expect 2D).
        if recv_topk_weights is None and recv_topk_idx is not None:
            recv_topk_weights = torch.ones(
                recv_topk_idx.shape, dtype=torch.float32,
                device=recv_topk_idx.device,
            )

        self._cached_handle = v2_handle

        # Wave 34.14: TRT-LLM 0.21 VariableLengthBuffer.dispatch at
        # deep_ep_utils.py:80 asserts `event.event is None` on sync path.
        if not async_finish:
            try:
                event.event = None
            except Exception:
                event = EventOverlap()
                try:
                    event.event = None
                except Exception:
                    pass

        return (recv_x, recv_topk_idx, recv_topk_weights,
                num_recv_per_expert, v2_handle, event)

    def combine(
        self,
        x: torch.Tensor,
        handle: Any,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        config: Optional[Config] = None,
        previous_event: Optional["EventOverlap"] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        **_: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], "EventOverlap"]:
        """V1 normal combine -> V2 ElasticBuffer.combine.

        V1 returns (combined_x, combined_topk_weights_or_None, event)
        V2 returns the same (combined_x, combined_topk_weights, event)
        """
        # Let V2 reuse handle.num_sms (set by dispatch) instead of forcing a
        # shim-chosen value. Mismatched num_sms between dispatch and combine
        # triggers sticky CUDA 719 at csrc/jit/handle.hpp:86 on the 2nd
        # combine (first trap'd asynchronously via
        # hybrid_combine.cuh:410/592 -> comm::timeout_while -> ptx::trap).
        # See docs/handle-hpp-86-analysis.md. Zero means "reuse handle's".
        num_sms = 0

        # NOTE: The previous empty-batch short-circuit (x.shape[0]==0
        # returned here without touching V2) was REMOVED 2026-04-29.
        # Same reasoning as dispatch: V2 hybrid_combine.cuh begins with
        # a collective gin barrier on kHybridCombineTag0 (8) across all
        # scaleup + scaleout peers. A rank that short-circuits never
        # signals, peers time out waiting. Let every rank participate
        # in the collective — if the underlying combine handles zero
        # tokens, great; if not, the V2 bug surfaces directly.

        # Shim5c instrumentation: dump handle fields the first time combine
        # runs so we can see *why* V2's combine.hpp:95 kernel launch fails.
        # The multi-node kernel path references token_metadata_at_forward
        # and channel_linked_list, which V2 documents as 'hybrid-mode only'.
        if not getattr(type(self), '_combine_dumped', False):
            import sys
            try:
                elastic = self._elastic
                print(f"[shim5c combine] x.shape={tuple(x.shape)} "
                      f"x.dtype={x.dtype} num_sms={num_sms} "
                      f"handle.num_sms={getattr(handle, 'num_sms', '?')} "
                      f"handle.num_experts={getattr(handle, 'num_experts', '?')} "
                      f"handle.num_max_tokens_per_rank="
                      f"{getattr(handle, 'num_max_tokens_per_rank', '?')} "
                      f"tm_at_forward_is_none="
                      f"{getattr(handle, 'token_metadata_at_forward', None) is None} "
                      f"channel_ll_is_none="
                      f"{getattr(handle, 'channel_linked_list', None) is None} "
                      f"num_scaleout_ranks="
                      f"{getattr(elastic, 'num_scaleout_ranks', '?')} "
                      f"num_scaleup_ranks="
                      f"{getattr(elastic, 'num_scaleup_ranks', '?')} "
                      f"allow_hybrid_mode="
                      f"{getattr(elastic, 'allow_hybrid_mode', '?')}",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[shim5c combine] dump failed: {e}", file=sys.stderr, flush=True)
            type(self)._combine_dumped = True

        # Stream-event handoff: close the race Gemini + GPT-5.4 flagged
        # when async_with_compute_stream=True and no previous_event is
        # threaded. PR #41183 always seeds with buffer.capture()
        # unconditionally (deepep_v2.py:313).
        # V2 contract (buffer.hpp:483): previous_event requires
        # allocate_on_comm_stream=True. Flip both together.
        # Gate on async_finish removed (matches dispatch path above).
        _prev_evt = getattr(previous_event, 'event', None)
        _alloc_on_comm = allocate_on_comm_stream
        if _prev_evt is None:
            try:
                _prev_evt = self._elastic.capture()
                _alloc_on_comm = True  # V2 contract
            except Exception:
                _prev_evt = None

        combined_x, combined_weights, event = self._elastic.combine(
            x,
            handle,
            topk_weights=topk_weights,
            bias=bias,
            num_sms=num_sms,
            num_qps=0,
            previous_event=_prev_evt,
            async_with_compute_stream=async_finish,
            allocate_on_comm_stream=_alloc_on_comm,
        )
        # Wave 34.14: TRT-LLM 0.21 VariableLengthBuffer.combine at
        # deep_ep_utils.py:89 asserts `event.event is None` on sync path.
        if not async_finish:
            try:
                event.event = None
            except Exception:
                event = EventOverlap()
                try:
                    event.event = None
                except Exception:
                    pass
        return combined_x, combined_weights, event

    # -----------------------------------------------------------------
    # Low-latency path (V1 LL API over V2 all-RDMA mode)
    # -----------------------------------------------------------------

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        use_nvfp4: bool = False,
        x_global_scale: Optional[torch.Tensor] = None,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        **_: Any,
    ) -> Tuple[Any, torch.Tensor, Any, "EventOverlap", Callable[[], None]]:
        """V1 low_latency_dispatch -> V2 ElasticBuffer.dispatch with
        allow_hybrid_mode=False (force-RDMA).

        **Output reshape**: V1 LL returns `recv_x` packed as
        `[num_local_experts, num_max_tokens*num_ranks, hidden]`. V2 dispatch
        returns flat `[num_recv_tokens, hidden]`. We reshape + pad to the V1
        packed shape. Frameworks (vLLM deepep_ll, SGLang) consume the packed
        layout directly.

        Returns V1 5-tuple: `(recv_x_or_tuple, recv_count, handle, event, hook)`
        """
        if use_nvfp4:
            raise NotImplementedError(
                "[api_shim] low_latency_dispatch(use_nvfp4=True) is a V1-only "
                "feature; V2's fp4 path is not yet wired into the shim. "
                "See docs/V1-to-V2-API-migration.md."
            )
        if x_global_scale is not None:
            raise NotImplementedError(
                "[api_shim] low_latency_dispatch(x_global_scale=...) requires "
                "nvfp4 V2 support; not wired."
            )
        _warn_drop("round_scale", round_scale)
        _warn_drop("use_ue8m0", use_ue8m0)

        # V2 dispatch — force all-RDMA path (we constructed with
        # allow_hybrid_mode=False when low_latency_mode=True).
        recv_x, recv_topk_idx, _recv_weights, v2_handle, event = \
            self._elastic.dispatch(
                x if not use_fp8 else x,
                topk_idx=topk_idx,
                topk_weights=None,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                async_with_compute_stream=async_finish,
            )

        # Reshape recv_x to V1 LL packed layout:
        #   [num_local_experts, num_max_tokens*num_ranks, hidden]
        num_local_experts = num_experts // self.group_size
        hidden = x.shape[1]
        packed_tokens = num_max_dispatch_tokens_per_rank * self.group_size

        def _pack(t: torch.Tensor) -> torch.Tensor:
            # V2 returned [num_recv_tokens, hidden]; we need to scatter into
            # [num_local_experts, packed_tokens, hidden] shape, zero-padded.
            # This uses handle.num_recv_tokens_per_expert_list if present.
            per_expert = getattr(v2_handle, 'num_recv_tokens_per_expert_list', None)
            out = torch.zeros(
                (num_local_experts, packed_tokens, t.shape[-1]),
                dtype=t.dtype, device=t.device,
            )
            if per_expert is None:
                return out
            if isinstance(per_expert, torch.Tensor):
                per_expert = per_expert.tolist()
            offset = 0
            for i, count in enumerate(per_expert):
                if count > 0 and offset + count <= t.shape[0]:
                    out[i, :count] = t[offset:offset + count]
                    offset += count
            return out

        # Handle FP8 return tuple shape
        if use_fp8 and isinstance(recv_x, tuple):
            recv_x_packed = (_pack(recv_x[0]), _pack(recv_x[1]))
        elif isinstance(recv_x, torch.Tensor):
            recv_x_packed = _pack(recv_x)
        else:
            recv_x_packed = recv_x

        # recv_count: `[num_local_experts]` int — count per local expert
        per_expert = getattr(v2_handle, 'num_recv_tokens_per_expert_list', None)
        if per_expert is None:
            recv_count = torch.zeros(num_local_experts, dtype=torch.int32, device=x.device)
        elif isinstance(per_expert, torch.Tensor):
            recv_count = per_expert.to(torch.int32)
        else:
            recv_count = torch.tensor(per_expert, dtype=torch.int32, device=x.device)

        # V1's hook: deferred data receipt. V2 doesn't split kernel from
        # receipt, but we can synthesize hook semantics with event.sync().
        # LLM consensus (Gemini, Claude): synthesize the hook so framework
        # overlap logic still waits at the right point.
        def _hook() -> None:
            if event is not None and hasattr(event, 'synchronize'):
                event.synchronize()

        # V1 handle was a tuple; we return the V2 EPHandle wrapped in a
        # tuple-like structure so frameworks that unpack a 5-element tuple
        # (src_info, layout_range, max_tokens, hidden, num_experts) still
        # work. Attach a .v2_handle attribute for the shim's combine path.
        shim_handle = _LLHandle(v2_handle, num_max_dispatch_tokens_per_rank,
                                hidden, num_experts)

        return recv_x_packed, recv_count, shim_handle, event, _hook

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Any,
        use_logfmt: bool = False,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
        combine_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        # SGLang-only overlap kwargs — silently drop
        overlap: bool = False,
        src_signals: Any = None,
        src_signal_expect_value: Any = None,
        packed_recv_count: Any = None,
        comp_signal: Any = None,
        block_m: Any = None,
        threshold: Any = None,
        num_sms: int = 0,
        **_: Any,
    ) -> Tuple[torch.Tensor, "EventOverlap", Callable[[], None]]:
        """V1 low_latency_combine -> V2 ElasticBuffer.combine.

        Input `x` is V1 LL packed `[num_local_experts, packed_tokens, hidden]`.
        V2 expects flat `[num_tokens, hidden]`. We un-pack before calling V2.

        Returns V1 3-tuple: `(combined_x, event, hook)`.
        """
        _warn_drop("use_logfmt", use_logfmt)
        _warn_drop("zero_copy", zero_copy)
        _warn_drop("overlap", overlap)

        # Unwrap shim handle
        v2_handle = handle.v2_handle if isinstance(handle, _LLHandle) else handle

        # Un-pack: take the first `sum(per_expert)` valid tokens from packed
        # layout. The V2 handle knows how many tokens each expert received.
        per_expert = getattr(v2_handle, 'num_recv_tokens_per_expert_list', None)
        if per_expert is not None and x.dim() == 3:
            if isinstance(per_expert, torch.Tensor):
                per_expert = per_expert.tolist()
            flat_chunks = []
            for i, count in enumerate(per_expert):
                if count > 0:
                    flat_chunks.append(x[i, :count])
            x_flat = torch.cat(flat_chunks, dim=0) if flat_chunks else \
                     torch.zeros(0, x.shape[-1], dtype=x.dtype, device=x.device)
        else:
            x_flat = x

        combined_x, _weights, event = self._elastic.combine(
            x_flat,
            v2_handle,
            topk_weights=topk_weights,
            num_sms=num_sms,
            num_qps=0,
            async_with_compute_stream=async_finish,
        )

        if out is not None:
            out.copy_(combined_x)
            combined_x = out

        def _hook() -> None:
            if event is not None and hasattr(event, 'synchronize'):
                event.synchronize()

        return combined_x, event, _hook

    # -----------------------------------------------------------------
    # LL buffer lifecycle (V1 specifics)
    # -----------------------------------------------------------------

    def clean_low_latency_buffer(
        self,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
    ) -> None:
        # V2 doesn't require explicit LL buffer cleanup; it's part of the
        # ElasticBuffer lifecycle. No-op here; the buffer will be cleaned
        # on the next dispatch or on destroy().
        pass

    def destroy(self) -> None:
        if hasattr(self._elastic, 'destroy'):
            self._elastic.destroy()

    # -----------------------------------------------------------------
    # TRT-LLM-specific extensions — raise with migration hint
    # -----------------------------------------------------------------

    def low_latency_dispatch_fp4(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "[api_shim] low_latency_dispatch_fp4 is a TRT-LLM extension over "
            "V1 Buffer. V2 ElasticBuffer has a different nvfp4 path; "
            "see docs/V1-to-V2-API-migration.md for the migration plan."
        )

    def low_latency_combine_low_precision(
        self,
        precision: str,
        hidden_states: torch.Tensor,
        global_scales: Optional[torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, "EventOverlap", Callable[[], None]]:
        """TRT-LLM extension over V1 `Buffer`. V2 `ElasticBuffer` handles
        FP8 combine natively (the fp8 cast happens inside the combine
        kernel when `use_fp8_dispatch=True` was set at buffer
        construction), so the `precision="fp8"` case degenerates to a
        pass-through into `low_latency_combine`.

        `precision="nvfp4"` would need a different kernel path and is
        not wired yet.

        `global_scales` is silently dropped for fp8: V2's elastic combine
        applies global fp8 scales via the buffer's internal state, not a
        per-call kwarg. See
        /home/ubuntu/deepep-intergration/deepep-v2/DeepEP/deep_ep/buffers/elastic.py
        around the combine signature for the V2 contract.
        """
        if precision == "fp8":
            _warn_drop("global_scales", global_scales is not None)
            return self.low_latency_combine(
                hidden_states,
                topk_idx,
                topk_weights,
                handle,
                *args,
                **kwargs,
            )
        raise NotImplementedError(
            f"[api_shim] low_latency_combine_low_precision(precision={precision!r}) "
            "is not wired to V2 yet. Only precision='fp8' pass-through is "
            "implemented; nvfp4 needs a dedicated V2 kernel path."
        )


class _LLHandle:
    """Shim handle for V1 low-latency path. V1 returns a 5-tuple
    (src_info, layout_range, max_tokens, hidden, num_experts). Frameworks
    (vLLM, SGLang, TRT-LLM) unpack this for low_latency_combine. We wrap
    V2's EPHandle and expose the V1 tuple shape on attribute access."""

    def __init__(self, v2_handle: Any, num_max_tokens: int,
                 hidden: int, num_experts: int) -> None:
        self.v2_handle = v2_handle
        self.num_max_tokens = num_max_tokens
        self.hidden = hidden
        self.num_experts = num_experts

    # Tuple-style unpacking (V1 contract)
    def __iter__(self):
        src_info = getattr(self.v2_handle, 'recv_src_metadata', None)
        layout_range = getattr(self.v2_handle, 'num_recv_tokens_per_expert_list', None)
        yield src_info
        yield layout_range
        yield self.num_max_tokens
        yield self.hidden
        yield self.num_experts

    def __getitem__(self, idx: int) -> Any:
        return list(self)[idx]

    def __len__(self) -> int:
        return 5


# -----------------------------------------------------------------------------
# Install: monkey-patch deep_ep.Buffer (and Config) with CompatBuffer.
# -----------------------------------------------------------------------------

def install(override_v2: bool = False) -> None:
    """Monkey-patch `deep_ep.Buffer` -> `CompatBuffer`. Idempotent.

    After install, any `from deep_ep import Buffer` returns CompatBuffer,
    so framework code using the V1 API transparently flows through V2.

    Args:
        override_v2: if True, also patches even if `deep_ep.Buffer` is
            already the V2-native Buffer. Default False — safest when
            upstream lands native V2-aware Buffer.
    """
    if getattr(deep_ep, '_api_shim_installed', False):
        return
    if not override_v2 and getattr(deep_ep, 'Buffer', None) is not None:
        # V1 Buffer already present; wrap it.
        pass
    deep_ep.Buffer = CompatBuffer  # type: ignore[attr-defined]
    # Also expose Config if not already there
    if not hasattr(deep_ep, 'Config'):
        deep_ep.Config = Config  # type: ignore[attr-defined]

    # V1 Megatron-LM imports:
    #   from deep_ep.utils import EventHandle, EventOverlap
    # V2's deep_ep.utils doesn't export EventOverlap by name. Inject our
    # shim EventOverlap + EventHandle onto deep_ep.utils so V1-shaped
    # imports succeed.
    try:
        import deep_ep.utils as _deep_ep_utils
        if not hasattr(_deep_ep_utils, 'EventOverlap'):
            _deep_ep_utils.EventOverlap = EventOverlap  # type: ignore[attr-defined]
        if not hasattr(_deep_ep_utils, 'EventHandle'):
            _deep_ep_utils.EventHandle = EventHandle  # type: ignore[attr-defined]
    except Exception:
        pass

    deep_ep._api_shim_installed = True  # type: ignore[attr-defined]
