# syntax=docker/dockerfile:1.10
# =============================================================================
# trtllm-deepep-v2-efa - TensorRT-LLM + DeepEP V2 + AWS EFA  (Wave 33)
# =============================================================================
#
# Wave 33: switched BASE_IMAGE_FAST from
#   ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.5-sm90a (PyPI torch 2.11.0 final)
# to
#   ghcr.io/antonai-work/deepep-v2-efa-base-ngc:v0.3.0-sm90a-ngc
#       (NGC pytorch 26.02 = torch 2.11.0a0+eb65b36914.nv26.2)
#
# Why: Wave 32 proved the PyPI torch 2.11.0 final lacks the 4-arg
# `c10::MessageLogger(const char*, int, int, bool)` constructor that
# trtllm 1.3.0rc14's libth_common.so requires. NGC's torch 2.11 alpha
# has it (verified via nm -D on both). See parent repo
# docs/wave32-evidence/trtllm-ngc-pivot-analysis.md.
#
# The NGC base already has:
#   - torch 2.11.0a0+eb65b36914.nv26.2 with Intel MKL
#   - cuda 13.1 runtime, nvcc, cudart, NVSHMEM
#   - patched NCCL 2.30.4 (256 -> 2048 XML cap)
#   - aws-ofi-nccl 206c02c4 + EFA userspace
#   - DeepEP V2 (PR #612 trio) at /opt/DeepEP
#
# This overlay only has to install tensorrt_llm (and force-preserve the
# base's nccl / nvshmem versions after trtllm's transitive-deps install).
# =============================================================================

ARG BASE_IMAGE_FAST=ghcr.io/antonai-work/deepep-v2-efa-base-ngc:v0.3.0-sm90a-ngc

FROM ${BASE_IMAGE_FAST} AS release

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# TensorRT-LLM 1.3.0rc14 from NVIDIA PyPI.
#
# wheel: tensorrt_llm-1.3.0rc14-cp312-cp312-linux_x86_64.whl  (2.6 GB)
#   compiled against NGC pytorch 26.02 torch 2.11.0a0 (verified ABI match)
#   requires-dist says `torch<=2.11.0a0,>=2.10.0` — our base ships
#   torch==2.11.0a0+eb65b36914.nv26.2 which satisfies it.
# -----------------------------------------------------------------------------
ARG TRTLLM_VERSION=1.3.0rc14
ARG TRTLLM_INDEX=https://pypi.nvidia.com

# Recipe (restored from be05ea0 — pre-Wave-33 working shape):
#   1. Install tensorrt_llm code only with --no-deps from NVIDIA PyPI
#   2. Extract declared deps, filter out torch/nccl/nvshmem/triton/flashinfer
#      (ecosystem packages we need to keep at base versions; bad set includes
#       hyphenated AND underscore forms because requires-dist may use either)
#   3. Install the filtered deps with NORMAL pip resolution (no --ignore-installed).
#      Key insight: --ignore-installed triggers uninstall-no-record-file on
#      apt-managed packages in NGC base (Pygments 2.17.2, etc). Since torch
#      and nvidia-* are already in the base and satisfy constraints, normal
#      pip resolution preserves them and only fills missing transitives.
#   4. Force-restore the patched NCCL 2.30.4 and NVSHMEM 3.6.5 after, since
#      tensorrt_llm declares nvidia-nccl-cu13<=2.29.2 which would downgrade
#      our patched libnccl.so.2.
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      --index-url "${TRTLLM_INDEX}" \
      --extra-index-url https://pypi.org/simple \
      "tensorrt_llm==${TRTLLM_VERSION}"

RUN set -e; \
    python3 -c "\
from importlib.metadata import requires; \
import re; \
reqs = requires('tensorrt_llm'); \
bad = {'torch','torchvision','torchaudio','torchcodec','torch-memory-saver','torchao','torch-c-dlpack-ext','flashinfer-python','flashinfer-cubin','triton','nvidia-nccl-cu13','nvidia-nvshmem-cu13'}; \
normdist = lambda s: re.split(r'[<>=! ;,\[]',s,1)[0].strip().lower().replace('_','-'); \
good = [r.split(';')[0].strip() for r in reqs if normdist(r) not in bad]; \
print('\n'.join(good))" > /tmp/trtllm-deps.txt && \
    cat /tmp/trtllm-deps.txt && echo "--- deps end ---" && \
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)") && \
    echo "[pre-deps] preserving torch=${TORCH_VER}" && \
    pip install --no-cache-dir --break-system-packages \
      --extra-index-url "${TRTLLM_INDEX}" \
      -r /tmp/trtllm-deps.txt && \
    python3 -c "import torch; v=torch.__version__; print('[post-deps] torch=', v); assert v.startswith('2.11.0a0'), 'torch downgraded to '+v+' after deps install'"

# Force-restore patched NCCL + NVSHMEM (pre-Wave-33 pattern from be05ea0).
# tensorrt_llm declares nvidia-nccl-cu13<=2.29.2 which the filtered-deps
# install above skipped (nvidia-nccl-cu13 in bad set), so NGC's bundled
# libnccl.so.2 should still be intact. This --no-deps --force-reinstall
# restores the specific pinned versions explicitly in case anything in
# the filtered deps pulled a different version.
ARG NVIDIA_NCCL_PIN=2.30.4
ARG NVSHMEM_PIN=3.6.5
RUN pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NVIDIA_NCCL_PIN}" \
      "nvidia-nvshmem-cu13==${NVSHMEM_PIN}"

# Build-time probes. CodeBuild has no NVIDIA driver (no libcuda.so.1) so
# we cannot exercise the CUDA ABI path (tensorrt_llm.bindings loads
# libth_common.so which dlopens libcuda). Limit the probe to version
# strings + wheel presence; the real ABI check runs on the GPU pod
# entrypoint (see k8s manifest import-probe block).
RUN python3 -c "import torch; v=torch.__version__; print('[trtllm-ngc] torch=', v, 'cuda=', torch.version.cuda); assert v.startswith('2.11.0a0'), 'expected NGC torch 2.11.0a0, got '+v; print('[trtllm-ngc] NGC torch alpha confirmed')" && \
    python3 -c "from importlib.metadata import version; print('[trtllm-ngc] tensorrt_llm=', version('tensorrt_llm'))" && \
    find /usr/local/lib/python3.12/dist-packages/tensorrt_llm -maxdepth 3 \( -name "libth_common.so" -o -name "bindings*.so" \) | head -5 && \
    echo "[trtllm-ngc] wheel shared objects located"

ENV TRTLLM_CAN_USE_DEEP_EP=1

WORKDIR /opt

LABEL org.opencontainers.image.title="trtllm-deepep-v2-efa" \
      org.opencontainers.image.description="TensorRT-LLM 1.3.0rc14 on NGC PyTorch 26.02 + DeepEP V2 + AWS EFA (Wave 33: NGC base for torch 2.11.0a0 ABI match)" \
      org.opencontainers.image.source="https://github.com/antonai-work/trtllm-deepep-v2-efa" \
      org.opencontainers.image.licenses="MIT" \
      status.wave="33" \
      base.variant="ngc" \
      base.image="deepep-v2-efa-base-ngc:v0.3.0-sm90a-ngc" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      deepep.pr="612" \
      trtllm.version="1.3.0rc14" \
      trtllm.source="https://pypi.nvidia.com/tensorrt-llm/"
