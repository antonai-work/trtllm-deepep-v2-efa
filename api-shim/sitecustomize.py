"""sitecustomize: auto-install the V1->V2 api_shim on EVERY Python interpreter
startup, so vllm's multiproc worker subprocesses also get deep_ep.Buffer ->
CompatBuffer. Activated by env DEEP_EP_USE_V2_SHIM=1.

This file lives at /opt/api-shim/sitecustomize.py. PYTHONPATH=/opt/api-shim
puts it on the import path; Python's site initialization imports
'sitecustomize' automatically if found.

A second, independently-guarded install step installs a sys.meta_path import
hook that monkey-patches `tensorrt_llm._torch.modules.fused_moe.deep_ep_utils`
so its two `Buffer(..., comm=<mpi4py>)` call sites translate the MPI comm
into a torch.distributed.ProcessGroup before reaching CompatBuffer.
Activated by DEEP_EP_SHIM_TRTLLM_COMM_BRIDGE=1 (default "1"). The hook is a
no-op unless TRT-LLM actually gets imported, so vLLM/sglang/megatron/nemo-rl
containers are unaffected.
"""
import os
import sys


def _install_trtllm_comm_bridge() -> None:
    """Install an import hook that rewrites the `reserve()` methods on
    `tensorrt_llm._torch.modules.fused_moe.deep_ep_utils.VariableLengthBuffer`
    and `VariableLengthLowLatencyBuffer` the first time that module is
    imported.

    The patched reserve() wraps the module's local `Buffer` symbol with a
    proxy that:
      * strips `comm=<mpi4py_comm>` from kwargs,
      * produces a torch.distributed ProcessGroup covering the MPI peers
        via `torch.distributed.new_group(ranks=list(range(comm.Get_size())))`
        (or reuses `dist.group.WORLD` when it already spans those ranks),
      * passes the PG through `group=` so `CompatBuffer.__init__` takes
        the torch-PG path rather than raising NotImplementedError on
        mpi4py comm.
    """
    import importlib.abc
    import importlib.machinery

    TARGET = "tensorrt_llm._torch.modules.fused_moe.deep_ep_utils"

    class _TRTLLMDeepEPLoader(importlib.abc.Loader):
        def __init__(self, real_loader: importlib.abc.Loader) -> None:
            self._real_loader = real_loader

        def create_module(self, spec):  # type: ignore[override]
            return self._real_loader.create_module(spec)

        def exec_module(self, module) -> None:  # type: ignore[override]
            self._real_loader.exec_module(module)
            try:
                _patch_deep_ep_utils_module(module)
            except Exception as e:  # pragma: no cover - defensive
                print(
                    f"[sitecustomize] TRT-LLM deep_ep_utils patch failed: {e}",
                    file=sys.stderr,
                )

    class _TRTLLMDeepEPFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # type: ignore[override]
            if fullname != TARGET:
                return None
            # Delegate to the remaining finders to locate the real spec,
            # then wrap its loader so we patch after exec_module().
            finders = [f for f in sys.meta_path if f is not self]
            for finder in finders:
                try:
                    spec = finder.find_spec(fullname, path, target)
                except AttributeError:
                    spec = None
                if spec is not None and spec.loader is not None:
                    spec.loader = _TRTLLMDeepEPLoader(spec.loader)
                    return spec
            return None

    # Idempotent: don't stack multiple finders across reloads.
    for existing in sys.meta_path:
        if existing.__class__.__name__ == "_TRTLLMDeepEPFinder":
            return
    sys.meta_path.insert(0, _TRTLLMDeepEPFinder())

    # If TRT-LLM was already imported (e.g. sitecustomize ran after TRT-LLM
    # at interpreter init time), patch the live module in place.
    already = sys.modules.get(TARGET)
    if already is not None:
        try:
            _patch_deep_ep_utils_module(already)
        except Exception as e:  # pragma: no cover
            print(
                f"[sitecustomize] TRT-LLM deep_ep_utils live patch failed: {e}",
                file=sys.stderr,
            )


def _lazy_init_torch_dist_from_mpi():
    """Initialize torch.distributed NCCL PG on this rank using MPI rank/size.

    TRT-LLM 0.21 fires a MoE warmup before torch.distributed.init_process_group
    is called (it drives MPI directly). For the V2 shim bridge to translate
    comm->ProcessGroup we need a real NCCL PG. We bootstrap one here using
    MPI for rendezvous:
      * RANK/WORLD_SIZE from MPI.COMM_WORLD
      * MASTER_ADDR from MPI rank-0 broadcast of socket.gethostname()
      * MASTER_PORT from env or default 29555 (distinct from trtllm port)

    Safe to call from any rank: all ranks must hit this before anyone
    returns a PG. torch.distributed handshakes across all ranks.
    """
    import socket
    import torch
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return

    try:
        from mpi4py import MPI
    except ImportError as e:
        raise RuntimeError(
            f"[trtllm-comm-bridge] mpi4py unavailable for lazy PG init: {e}"
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Pick master addr/port. Prefer env-provided MASTER_ADDR (so the caller
    # can force a specific pod IP that is routable cross-pod). Fall back
    # to rank-0 broadcasting socket.gethostname(), which only works when
    # the pod's hostname resolves to a cross-pod-reachable IP.
    env_master_addr = os.environ.get("MASTER_ADDR")
    if env_master_addr:
        master_addr = env_master_addr
    else:
        master_addr = socket.gethostname() if rank == 0 else None
        master_addr = comm.bcast(master_addr, root=0)
    master_port = os.environ.get("MASTER_PORT") or os.environ.get(
        "TRTLLM_SHIM_PG_MASTER_PORT", "29555"
    )

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Bind this rank to its GPU before NCCL init.
    if torch.cuda.is_available():
        local_rank_env = (
            os.environ.get("LOCAL_RANK")
            or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
            or os.environ.get("SLURM_LOCALID")
        )
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
        else:
            local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

    print(
        f"[trtllm-comm-bridge] lazy-initializing torch.distributed NCCL "
        f"PG rank={rank}/{world_size} master={master_addr}:{master_port}",
        file=sys.stderr,
        flush=True,
    )
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


def _patch_deep_ep_utils_module(module) -> None:
    """Wrap `module.Buffer` so mpi4py `comm=` is translated to a torch PG
    before reaching deep_ep.Buffer (CompatBuffer). This leaves the upstream
    `reserve()` implementations unchanged; only the module-local `Buffer`
    symbol they call is proxied.
    """
    import functools
    import torch.distributed as dist

    real_buffer = getattr(module, "Buffer", None)
    if real_buffer is None:
        return
    if getattr(real_buffer, "__wrapped_by_api_shim__", False):
        return

    def _comm_to_pg(comm):
        """Translate an mpi4py comm to a torch PG usable by ElasticBuffer.

        The returned PG must be a NCCL group over the MPI peers. Reuse
        `dist.group.WORLD` when it already spans the same ranks (common
        case: one PP/EP group per interpreter); otherwise build a new
        NCCL subgroup.
        """
        if not dist.is_initialized():
            # TRT-LLM 0.21 fires MoE warmup before calling
            # torch.distributed.init_process_group. Bootstrap a NCCL PG
            # over MPI ranks so the shim can translate comm -> PG.
            _lazy_init_torch_dist_from_mpi()

        world_size = comm.Get_size()
        if dist.get_world_size() == world_size:
            wg = dist.group.WORLD
            try:
                backend = dist.get_backend(wg).lower()
            except Exception:
                backend = ""
            if backend == "nccl":
                return wg
        return dist.new_group(ranks=list(range(world_size)), backend="nccl")

    class _MetaInitBufferStub:
        """Lightweight stand-in for a Buffer created while TRT-LLM is in
        MetaInitMode. Exposes the attributes VariableLengthBuffer.reserve
        later checks (`num_nvl_bytes`, `num_rdma_bytes`) so the second
        (real) pass can tell the first-pass stub from a real Buffer and
        replace it. ElasticBuffer is NOT constructed here, because
        ElasticBuffer.__init__ calls dist.all_gather_object which trips
        MetaInitMode's aten.set_.source_Storage guard.
        """
        __wrapped_by_api_shim__ = True

        def __init__(self, *, num_nvl_bytes=0, num_rdma_bytes=0, **_kw):
            self.num_nvl_bytes = int(num_nvl_bytes)
            self.num_rdma_bytes = int(num_rdma_bytes)
            self._is_meta_stub = True

    def _in_meta_init_mode() -> bool:
        try:
            from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
        except Exception:
            return False
        try:
            import torch.utils._python_dispatch as _pd
            stack = _pd._get_current_dispatch_mode_stack()
        except Exception:
            # Old torch without stack introspection: best effort.
            return False
        return any(isinstance(m, MetaInitMode) for m in stack)

    @functools.wraps(real_buffer)
    def Buffer_proxy(*args, **kwargs):
        # Under TRT-LLM MetaInitMode, constructing ElasticBuffer trips
        # the meta-tensor guard inside dist.all_gather_object (torch
        # ByteTensor from Storage). Return a stub so the MetaInit pass
        # finishes; TRT-LLM's fallback path re-enters without MetaInitMode
        # and the real Buffer gets built.
        if _in_meta_init_mode():
            # TRT-LLM call: Buffer(None, num_nvl_bytes, num_rdma_bytes,
            # num_nvl_peers=..., comm=...). Pull sizes positionally
            # when present, else from kwargs.
            nvl = args[1] if len(args) > 1 else kwargs.get("num_nvl_bytes", 0)
            rdma = args[2] if len(args) > 2 else kwargs.get("num_rdma_bytes", 0)
            return _MetaInitBufferStub(num_nvl_bytes=nvl, num_rdma_bytes=rdma)

        comm = kwargs.pop("comm", None)
        if comm is not None:
            pg = _comm_to_pg(comm)
            # TRT-LLM calls Buffer(None, num_nvl_bytes, num_rdma_bytes,
            # num_nvl_peers=..., comm=...). Arg 0 is the positional
            # `group=None`. Replace it with the translated PG so we
            # preserve the argument positions of num_nvl_bytes and
            # num_rdma_bytes that follow.
            if args:
                args = (pg,) + args[1:]
            elif "group" not in kwargs:
                kwargs["group"] = pg
        return real_buffer(*args, **kwargs)

    Buffer_proxy.__wrapped_by_api_shim__ = True  # type: ignore[attr-defined]
    module.Buffer = Buffer_proxy

    # Wrap `reserve` so that if a MetaInit stub landed in self.buffer from
    # the first (meta) pass, we drop it before the upstream `reserve` body
    # runs — forcing it to take the "buffer is None" branch and construct
    # a real Buffer under the (now non-meta) torch dispatch state.
    #
    # Also wrap `dispatch` and friends so that if the stub is still in
    # self.buffer by the time a real forward fires (common — the meta
    # pass does not trigger the fallback), we upgrade to a real Buffer
    # on demand by re-calling the upstream reserve with cached sizes.
    def _upgrade_stub_if_needed(self):
        buf = getattr(self, "buffer", None)
        if buf is None or not getattr(buf, "_is_meta_stub", False):
            return
        nvl_hint = int(buf.num_nvl_bytes)
        rdma_hint = int(buf.num_rdma_bytes)
        # Drop the stub and rebuild a real Buffer. We reach into the
        # module-level `Buffer` symbol (our proxy) which, with no active
        # MetaInitMode, will construct a real ElasticBuffer.
        self.buffer = None
        Buffer_sym = module.Buffer
        import tensorrt_llm._utils as _tu
        self.buffer = Buffer_sym(
            None,
            max(1, nvl_hint),
            max(1, rdma_hint),
            num_nvl_peers=_tu.local_mpi_size(),
            comm=self.comm,
        )

    for cls_name in ("VariableLengthBuffer", "VariableLengthLowLatencyBuffer"):
        cls = getattr(module, cls_name, None)
        if cls is None:
            continue

        reserve = getattr(cls, "reserve", None)
        if reserve is not None and not getattr(
            reserve, "__wrapped_by_api_shim__", False
        ):
            @functools.wraps(reserve)
            def _reserve_tag(self, *a, __orig=reserve, **kw):
                _upgrade_stub_if_needed(self)
                return __orig(self, *a, **kw)

            _reserve_tag.__wrapped_by_api_shim__ = True  # type: ignore[attr-defined]
            _reserve_tag.__wrapped__ = reserve  # type: ignore[attr-defined]
            setattr(cls, "reserve", _reserve_tag)

        # Also wrap forward-facing buffer users so stubs are upgraded
        # lazily on first real forward.
        for meth_name in (
            "dispatch",
            "combine",
            "low_latency_dispatch",
            "low_latency_combine",
            "get_dispatch_layout",
        ):
            meth = getattr(cls, meth_name, None)
            if meth is None or getattr(meth, "__wrapped_by_api_shim__", False):
                continue

            @functools.wraps(meth)
            def _meth_tag(self, *a, __orig=meth, **kw):
                _upgrade_stub_if_needed(self)
                return __orig(self, *a, **kw)

            _meth_tag.__wrapped_by_api_shim__ = True  # type: ignore[attr-defined]
            _meth_tag.__wrapped__ = meth  # type: ignore[attr-defined]
            setattr(cls, meth_name, _meth_tag)


if os.environ.get("DEEP_EP_USE_V2_SHIM", "0") == "1":
    try:
        import api_shim
        api_shim.install()
    except Exception as e:
        print(f"[sitecustomize] api_shim.install() failed: {e}", file=sys.stderr)

if os.environ.get("DEEP_EP_SHIM_TRTLLM_COMM_BRIDGE", "1") == "1":
    try:
        _install_trtllm_comm_bridge()
    except Exception as e:
        print(
            f"[sitecustomize] TRT-LLM comm bridge install failed: {e}",
            file=sys.stderr,
        )
