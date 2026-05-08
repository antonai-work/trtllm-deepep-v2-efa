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

# Recipe (venv isolation — Wave 33 architectural fix):
#   Installing tensorrt_llm into the NGC base's system python triggers pip's
#   uninstall-no-record-file death on apt-managed packages (wheel 0.42.0,
#   Pygments 2.17.2, setuptools, safetensors, pyzmq — all shipped via apt in
#   NGC pytorch:26.02). Builds #11-#19 all hit this wall regardless of
#   --ignore-installed / --no-deps / --force-reinstall flag permutations.
#
#   Root cause: apt-managed packages have no RECORD file; pip cannot uninstall
#   them. Any filtered-deps install that wants to upgrade e.g. Pygments>=2.18
#   tries to uninstall 2.17.2 first and dies. Local repro confirmed the N+1th
#   conflicting apt package surfaces after every single-package exclusion.
#
#   Fix: install tensorrt_llm into /opt/trtllm-venv with --system-site-packages.
#   The venv SEES the base's torch 2.11.0a0 + nvidia-nccl-cu13 + nvidia-nvshmem-cu13
#   (satisfying tensorrt_llm's dep constraints without triggering upgrades).
#   New installs land in the venv's own site-packages with proper RECORD files.
#   apt packages are SHADOWED by venv packages, not uninstalled.
#
#   Verified locally (2026-05-08): scratch/wave33-trtllm-local-verify/03-verify-venv-fix.sh
#   PASSED all 4 steps against the real NGC base in 2m45s.
ENV TRTLLM_VENV=/opt/trtllm-venv
ENV PATH=${TRTLLM_VENV}/bin:${PATH}

RUN python3 -m venv --system-site-packages "${TRTLLM_VENV}" && \
    "${TRTLLM_VENV}/bin/python3" -c "import torch; v=torch.__version__; print('[venv-create] torch=', v); assert v.startswith('2.11.0a0'), 'base torch lost: '+v"

RUN "${TRTLLM_VENV}/bin/pip" install --no-cache-dir --no-deps \
      --index-url "${TRTLLM_INDEX}" \
      --extra-index-url https://pypi.org/simple \
      "tensorrt_llm==${TRTLLM_VERSION}"

RUN set -e; \
    "${TRTLLM_VENV}/bin/python3" -c "\
from importlib.metadata import requires; \
import re; \
reqs = requires('tensorrt_llm'); \
bad = {'torch','torchvision','torchaudio','torchcodec','torch-memory-saver','torchao','torch-c-dlpack-ext','flashinfer-python','flashinfer-cubin','triton','nvidia-nccl-cu13','nvidia-nvshmem-cu13'}; \
normdist = lambda s: re.split(r'[<>=! ;,\[]',s,1)[0].strip().lower().replace('_','-'); \
good = [r.split(';')[0].strip() for r in reqs if normdist(r) not in bad]; \
print('\n'.join(good))" > /tmp/trtllm-deps.txt && \
    cat /tmp/trtllm-deps.txt && echo "--- deps end ---" && \
    TORCH_VER=$("${TRTLLM_VENV}/bin/python3" -c "import torch; print(torch.__version__)") && \
    echo "[pre-deps] preserving torch=${TORCH_VER}" && \
    "${TRTLLM_VENV}/bin/pip" install --no-cache-dir \
      --extra-index-url "${TRTLLM_INDEX}" \
      -r /tmp/trtllm-deps.txt && \
    "${TRTLLM_VENV}/bin/python3" -c "import torch; v=torch.__version__; print('[post-deps] torch=', v); assert v.startswith('2.11.0a0'), 'torch downgraded to '+v+' after deps install'"

# Force-restore patched NCCL + NVSHMEM. tensorrt_llm declares
# nvidia-nccl-cu13<=2.29.2 which the filtered-deps install above skipped
# (in bad set). The venv with --system-site-packages inherits the base's
# libnccl.so.2 (2.30.4-patched) and libnvshmem_host.so (3.6.5) automatically,
# so this reinstall is defensive only — ensures pinned versions survive
# even if a transitive dep attempts a version flip inside the venv.
ARG NVIDIA_NCCL_PIN=2.30.4
ARG NVSHMEM_PIN=3.6.5
RUN "${TRTLLM_VENV}/bin/pip" install --no-cache-dir --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NVIDIA_NCCL_PIN}" \
      "nvidia-nvshmem-cu13==${NVSHMEM_PIN}"

# Wave 33.2: restore flashinfer after the filtered-deps install.
# The parent bad-set in /tmp/trtllm-deps.txt deliberately skipped
# {flashinfer-python, flashinfer-cubin} to prevent them from pulling torch
# as a hard dep, but tensorrt_llm 1.3.0rc14 HARD-IMPORTS flashinfer at
# module load:
#   tensorrt_llm/_torch/flashinfer_utils.py:20   import flashinfer
#   tensorrt_llm/_torch/attention_backend/__init__.py  exports
#   FlashInferAttentionMetadata
# Pod run at 2026-05-08 13:34 UTC hit:
#   ImportError: cannot import name 'FlashInferAttentionMetadata' from
#   'tensorrt_llm._torch.attention_backend'
# Install with --no-deps so torch is NOT re-pinned to the PyPI build.
ARG FLASHINFER_PIN=0.4.1
RUN "${TRTLLM_VENV}/bin/pip" install --no-cache-dir --no-deps \
      "flashinfer-python==${FLASHINFER_PIN}" && \
    "${TRTLLM_VENV}/bin/python3" -c "import flashinfer; print('[wave33.2] flashinfer=', flashinfer.__version__)" && \
    "${TRTLLM_VENV}/bin/python3" -c "from tensorrt_llm._torch.attention_backend import FlashInferAttentionMetadata; print('[wave33.2] tensorrt_llm import chain clean')"

# Build-time probes. CodeBuild has no NVIDIA driver (no libcuda.so.1) so
# we cannot exercise the CUDA ABI path (tensorrt_llm.bindings loads
# libth_common.so which dlopens libcuda). Limit the probe to version
# strings + wheel presence; the real ABI check runs on the GPU pod
# entrypoint (see k8s manifest import-probe block).
RUN "${TRTLLM_VENV}/bin/python3" -c "import torch; v=torch.__version__; print('[trtllm-ngc] torch=', v, 'cuda=', torch.version.cuda); assert v.startswith('2.11.0a0'), 'expected NGC torch 2.11.0a0, got '+v; print('[trtllm-ngc] NGC torch alpha confirmed')" && \
    "${TRTLLM_VENV}/bin/python3" -c "from importlib.metadata import version; print('[trtllm-ngc] tensorrt_llm=', version('tensorrt_llm'))" && \
    find "${TRTLLM_VENV}/lib/python3.12/site-packages/tensorrt_llm" -maxdepth 3 \( -name "libth_common.so" -o -name "bindings*.so" \) | head -5 && \
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
