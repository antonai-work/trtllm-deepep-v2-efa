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
# TensorRT-LLM overlay. Wave 30 will:
#   1. Confirm 0.21.0 still installs cleanly on v0.2.5 cu13 base.
#   2. Re-drive the CutlassFusedMoE DeepEP fast path via a YAML-driven
#      `trtllm-serve --extra_llm_api_options=<yaml>` with
#      `enable_attention_dp: true`, `TRTLLM_CAN_USE_DEEP_EP=1`.
#   3. Confirm EFA counter delta >= 1 GB on a 2-pod 4x512-token run.
# -----------------------------------------------------------------------------
ARG TRTLLM_VERSION=0.21.0

# Recipe (proven via docker-run smoke on v0.2.5 base, same as sglang):
#   1. Install tensorrt_llm code only (--no-deps)
#   2. Compute tensorrt_llm's declared deps MINUS torch/flashinfer/etc
#   3. Install that filtered list normally
#   4. Force-restore nvidia-nccl-cu13 + nvidia-nvshmem-cu13 to v0.2.5 versions
#
# Why: tensorrt_llm 0.21.0 hard-pins torch 2.6+cu124 which would shatter
# the v0.2.5 cu13 invariant (EVIDENCE-LEDGER Wave 7d #1).
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      "tensorrt_llm==${TRTLLM_VERSION}"

RUN set -e; \
    python3 -c "\
from importlib.metadata import requires; \
reqs = requires('tensorrt_llm'); \
bad = {'torch','torchvision','torchaudio','torchcodec','torch_memory_saver','torchao','flashinfer_python','flashinfer_cubin','triton','tensorrt'}; \
good = [r.split(';')[0].strip() for r in reqs if not any(r.lower().startswith(b.lower()) for b in bad)]; \
print('\n'.join(good))" > /tmp/trtllm-deps.txt && \
    pip install --no-cache-dir --break-system-packages -r /tmp/trtllm-deps.txt

# Force-restore v0.2.5 cu13 stack
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
      status.wave="30-pending" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      deepep.pr="612" \
      trtllm.version="0.21.0"
