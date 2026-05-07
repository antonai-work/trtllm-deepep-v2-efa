# syntax=docker/dockerfile:1.10
# =============================================================================
# trtllm-deepep-v2-efa - TensorRT-LLM + DeepEP V2 + AWS EFA
# =============================================================================
#
# STATUS: SCAFFOLD. TRT-LLM DeepEP fast path was validated in Wave 4 on
# the v0.2.3 base (4x512 tokens, AlltoallMethodType=DeepEP); revalidation
# on the current v0.2.5 base is Wave 30 in the private dev tree
# (deepep-v2-integration). This Dockerfile lands the cascade-symmetric
# source shape; the TRT-LLM pip body will be materialized once
# Wave 30 PROVEN evidence is in hand.
#
# Dual-path build (sibling pattern from vllm-deepep-v2-efa):
#   BUILD_MODE=fast    -> FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.5-sm90a
#   BUILD_MODE=vanilla -> TODO (Wave 30). See sibling repos for inline pattern.
#
# Build:
#   docker build -f docker/Dockerfile --build-arg BUILD_MODE=fast \
#                -t trtllm-deepep-v2-efa:fast .
#
# Sibling repos:
#   - antonai-work/deepep-v2-efa-base      (base substrate)
#   - antonai-work/vllm-deepep-v2-efa      (reference inference overlay)
#   - antonai-work/sglang-deepep-v2-efa    (sibling inference overlay)
# =============================================================================

ARG BUILD_MODE=fast
ARG BASE_IMAGE_FAST=ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.5-sm90a

FROM ${BASE_IMAGE_FAST} AS base-fast

FROM base-fast AS release

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# TensorRT-LLM overlay — Wave 32 version bump.
#
# Why 1.3.0rc14: NVIDIA PyPI (https://pypi.nvidia.com/tensorrt-llm/) ships
# tensorrt_llm-1.3.0rc14-cp312-cp312-linux_x86_64.whl (published 2026-05-06)
# with cu13 runtime and torch 2.11 ABI. Empirically verified:
# `nm -D libth_common.so | grep c10_cuda_check` → `_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`
# which is the torch 2.11 `j`-variant (unsigned int line number). Our base
# has torch 2.11.0+cu130 on cp312, so wheel + ABI match.
#
# The requires_dist claims `torch<=2.11.0a0,>=2.10.0`, which is a conservative
# metadata pin that doesn't reflect the actual compiled ABI. `--no-deps` bypass
# is required to install on torch 2.11.0 (final), which is safe because the
# compiled extensions are already torch 2.11-ABI-compatible.
#
# Prior attempt (v0.2.5): tensorrt_llm 0.21.0 shipped torch-2.6 C++ extensions
# incompatible with torch 2.11. 1.3.0rc14 is the NVIDIA cu13 wheel line.
# -----------------------------------------------------------------------------
ARG TRTLLM_VERSION=1.3.0rc14
ARG TRTLLM_INDEX=https://pypi.nvidia.com

# Recipe (same shape as sglang; bumped wheel source to NVIDIA PyPI):
#   1. Install tensorrt_llm code only (--no-deps) from NVIDIA PyPI cu13 wheel
#   2. Compute declared deps MINUS torch/flashinfer/tensorrt via importlib
#   3. Install the filtered list from NVIDIA + PyPI (tensorrt is on NVIDIA)
#   4. Force-restore nvidia-nccl-cu13 + nvidia-nvshmem-cu13 to v0.2.5 versions
#      (tensorrt_llm declares nvidia-nccl-cu13<=2.29.2 which would downgrade
#       our patched 2.30.4 — must force-reinstall after)
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      --index-url "${TRTLLM_INDEX}" \
      --extra-index-url https://pypi.org/simple \
      "tensorrt_llm==${TRTLLM_VERSION}"

RUN set -e; \
    python3 -c "\
from importlib.metadata import requires; \
reqs = requires('tensorrt_llm'); \
bad = {'torch','torchvision','torchaudio','torchcodec','torch_memory_saver','torchao','flashinfer_python','flashinfer_cubin','triton'}; \
good = [r.split(';')[0].strip() for r in reqs if not any(r.lower().startswith(b.lower()) for b in bad)]; \
print('\n'.join(good))" > /tmp/trtllm-deps.txt && \
    pip install --no-cache-dir --break-system-packages \
      --extra-index-url "${TRTLLM_INDEX}" \
      -r /tmp/trtllm-deps.txt

# Force-restore v0.2.5 cu13 stack. tensorrt_llm declares
# `nvidia-nccl-cu13<=2.29.2` which would downgrade our patched 2.30.4.
# Must run AFTER the dep install to win the version race.
ARG NVIDIA_NCCL_PIN=2.30.4
ARG NVSHMEM_PIN=3.6.5
RUN pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NVIDIA_NCCL_PIN}" \
      "nvidia-nvshmem-cu13==${NVSHMEM_PIN}"



ENV TRTLLM_CAN_USE_DEEP_EP=1

WORKDIR /opt

LABEL org.opencontainers.image.title="trtllm-deepep-v2-efa" \
      org.opencontainers.image.description="TensorRT-LLM + DeepEP V2 + AWS EFA MoE inference cascade (Wave 30 pending)." \
      org.opencontainers.image.source="https://github.com/antonai-work/trtllm-deepep-v2-efa" \
      org.opencontainers.image.licenses="MIT" \
      status.wave="32" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      deepep.pr="612" \
      trtllm.version="1.3.0rc14" \
      trtllm.source="https://pypi.nvidia.com/tensorrt-llm/"
