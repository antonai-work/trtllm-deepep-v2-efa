#!/bin/bash
# Entrypoint for trtllm-deepep-v2.
#
# Responsibilities:
#   * Install the V1->V2 api-shim if DEEP_EP_USE_V2_SHIM=1 (so TRT-LLM's
#     deep_ep_utils.py still resolves `deep_ep.Buffer` on imports).
#   * Exec the caller's command (typically trtllm-serve or mpirun ...).
#
# The comm-bridge decision is Option B (docs/trtllm-comm-decision.md):
# shim requires a torch ProcessGroup. Consumer code that still calls
# `Buffer(None, ..., comm=mpi_comm)` will hit NotImplementedError. See
# docs/trtllm-comm-decision.md for the one-line migration.
#
# Usage (inside a pod):
#   trtllm-deepep-v2-entrypoint.sh trtllm-serve --model ...
#   DEEP_EP_USE_V2_SHIM=1 trtllm-deepep-v2-entrypoint.sh python3 /opt/smoke_test_shim.py
set -euo pipefail

# Startup accelerator: reuse JIT caches on FSX across pod restarts.
# Drops FlashInfer CUTLASS + torch inductor + DeepEP .so JIT time from
# ~8 min to ~0 on second-and-later runs.
if [ -d /mnt/fsx ] && [ -w /mnt/fsx ]; then
  export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-/mnt/fsx/flashinfer}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/fsx/torch-inductor}"
  export DEEP_EP_JIT_CACHE_DIR="${DEEP_EP_JIT_CACHE_DIR:-/mnt/fsx/deepep-jit}"
  mkdir -p "$FLASHINFER_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$DEEP_EP_JIT_CACHE_DIR" 2>/dev/null || true
fi

# OpenMPI wants these at every invocation (root in a pod is normal).
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_plm_rsh_agent=${OMPI_MCA_plm_rsh_agent:-/bin/false}

if [ "${DEEP_EP_USE_V2_SHIM:-0}" = "1" ]; then
  echo "[trtllm-deepep-v2] Installing V1 -> V2 api-shim"
  python3 -c "import api_shim; api_shim.install()" || {
    echo "[trtllm-deepep-v2] Shim install failed; import path likely missing";
    exit 1;
  }
fi

exec "$@"
