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

# tensorrt_llm 0.21.0 pins torch 2.6.0+cu124 which would shatter v0.2.5's
# cu13 invariant (EVIDENCE-LEDGER Wave 7d #1). Install --no-deps and
# curate runtime deps to keep torch 2.11.0+cu130 intact.
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      "tensorrt_llm==${TRTLLM_VERSION}"

# Non-torch runtime deps. Curated from tensorrt_llm 0.21.0 pyproject.toml
# minus {torch*, nvidia-*-cu12, tensorrt (wrong CUDA), flashinfer*}.
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      mpi4py accelerate build colored cuda-python diffusers lark \
      onnx onnx_graphsurgeon openai polygraphy psutil \
      'nvidia-ml-py>=12' 'pynvml>=12.0.0' pulp pandas 'h5py==3.12.1' \
      StrEnum 'sentencepiece>=0.1.99' transformers tokenizers safetensors \
      huggingface_hub requests urllib3 charset-normalizer idna certifi \
      tqdm pydantic pydantic_core fastapi 'uvicorn[standard]' \
      starlette sniffio anyio h11 click numpy scipy einops \
      janus filelock packaging pyyaml jinja2 markupsafe typing_extensions \
      'email-validator' annotated-types aiohttp httpx

# NOTE: no nvidia-nccl-cu13 or nvidia-nvshmem-cu13 guard needed. The
# --no-deps install never touched them; base's cu13 stack is untouched.

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
