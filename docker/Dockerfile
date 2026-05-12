# syntax=docker/dockerfile:1.10
# =============================================================================
# trtllm-deepep-v2-efa - TensorRT-LLM + DeepEP V2 + AWS EFA (Wave 34.8)
# =============================================================================
#
# STATUS: Wave 34.8 — TRT-LLM 0.21.0 + api-shim V1->V2 bridge. This is the
# PROVEN recipe from the private integration tree
# (deepep-v2-integration/integrations/trtllm-deepep-v2/docker/Dockerfile, 190
# lines). Replaces the prior public scaffold (106 lines, no shim, TRT-LLM
# 1.3.0rc14 — blocked by rc14 mgmn_worker_node eager nvshmem_init on EFA).
#
# Why 0.21.0 (not 1.3.0rc14): upstream TRT-LLM 1.3.0rc14 eagerly calls
# `nvshmem_init` during `mgmn_worker_node` startup and segfaults on AWS
# EFA 32-NIC configurations (verified Wave 34.5/6/7 2026-05-12). TRT-LLM
# 0.21.0 uses the deep_ep V1 `Buffer(..., comm=)` surface which the api-shim
# intercepts and routes through `deep_ep.buffers.elastic.ElasticBuffer` —
# pure NCCL-Gin, zero NVSHMEM. Run #5 (2026-04-29) with this recipe hit
# 4.58 GB EFA TX on 2×H100 with `AlltoallMethodType=DeepEP` selected.
#
# Upstream TRT-LLM has ZERO work toward native `ElasticBuffer` adoption
# as of 2026-05-12 (main HEAD 0119a23783 still imports `from
# tensorrt_llm.deep_ep import Buffer`, V1 API). We ship the shim
# indefinitely.
#
# Base cascade:
#   nvidia/cuda:13.0.0 -> deepep-v2-efa-base:v0.2.5-sm90a (patched NCCL + EFA + DeepEP V2)
#   -> trtllm-deepep-v2-efa:<tag> (this repo, Wave 34.8)
#
# Build:
#   docker build -f docker/Dockerfile -t trtllm-deepep-v2-efa:fast .
#
# Sibling repos:
#   - antonai-work/deepep-v2-efa-base      (base substrate)
#   - antonai-work/vllm-deepep-v2-efa      (PROVEN native V2 via PR #41183)
#   - antonai-work/sglang-deepep-v2-efa    (fork V2 via dmvevents/sglang PR #24443)
# =============================================================================

ARG BASE_IMAGE=ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.5-sm90a

FROM ${BASE_IMAGE} AS release

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# TRT-LLM 0.21 is the last release with the stable deep_ep V1 Buffer
# surface the api-shim targets. See rc14 NVSHMEM-eager-init research in
# the private dev tree (docs/wave34-evidence/).
ARG TRTLLM_VERSION=0.21.0

USER root
WORKDIR /workspace

# -----------------------------------------------------------------------------
# 1. PEP 668 + OpenMPI (TRT-LLM launches through mpirun even in single-node)
# -----------------------------------------------------------------------------
RUN find /usr/lib/python3*/EXTERNALLY-MANAGED -delete 2>/dev/null || true; \
    find /usr/local/lib/python3*/EXTERNALLY-MANAGED -delete 2>/dev/null || true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      openmpi-bin libopenmpi-dev openmpi-common openssh-client openssh-server netcat-openbsd \
      libfmt-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ompi_info --version 2>&1 | head -3 || true

# -----------------------------------------------------------------------------
# 1.5. Stash base image fmt headers (bundled with torch 2.11, including
#      base.h which Ubuntu 24.04's libfmt-dev 9.1 does NOT ship). TRT-LLM
#      0.21 downgrades torch to 2.7.1 which drops the bundled fmt, so we
#      need to restore these headers before rebuilding DeepEP.
# -----------------------------------------------------------------------------
RUN cp -r /usr/local/lib/python3.12/dist-packages/torch/include/fmt /opt/fmt_headers_from_torch2_11 \
 && ls /opt/fmt_headers_from_torch2_11/base.h >/dev/null && echo "fmt backup OK"

# -----------------------------------------------------------------------------
# 2. TensorRT-LLM 0.21
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --break-system-packages \
      "tensorrt_llm==${TRTLLM_VERSION}" \
      "mpi4py" \
 && (python3 -c "import tensorrt_llm, torch; print(f'trtllm={tensorrt_llm.__version__} torch={torch.__version__}')" \
      || echo "!! tensorrt_llm import deferred (likely needs libcuda.so.1 at runtime); will retry inside pod")

# TRT-LLM wants legacy `from cuda import cuda, cudart` from cuda-python 12.x.
RUN pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps \
      "cuda-python==12.9.0" "cuda-bindings==12.9.0" && \
    python3 -c "from cuda import cuda, cudart; print('legacy cuda API ok')" || \
    echo "WARN: legacy cuda import still failing; trtllm-serve may fault at model compile time"

RUN pip install --no-cache-dir --break-system-packages "opencv-python-headless>=4.10,<4.12" || true

# -----------------------------------------------------------------------------
# 3. OpenMPI root-exec env
# -----------------------------------------------------------------------------
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_plm_rsh_agent=/bin/false

# -----------------------------------------------------------------------------
# 4. NCCL + DeepEP rebuild. TRT-LLM 0.21 pins torch 2.7.1+cu126 and pulls
#    nvidia-nccl-cu12 2.26.2, replacing the base image's torch 2.11+cu130
#    and nvidia-nccl-cu13 2.30.4. DeepEP V2's elastic.py binds against
#    ncclTeamWorld (new in NCCL 2.30), so we need to keep cu13 NCCL 2.30.4
#    alongside TRT's cu12 NCCL 2.26.
# -----------------------------------------------------------------------------
RUN set -eux; \
    pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "nvidia-nccl-cu13>=2.30.4"; \
    NCCL_LIB="$(find /usr/local/lib /usr/lib -path '*/nvidia/nccl/lib' -type d 2>/dev/null | head -1)"; \
    SO2="$(ls "${NCCL_LIB}"/libnccl.so.* 2>/dev/null | head -1)"; \
    ln -sf "$(basename "${SO2}")" "${NCCL_LIB}/libnccl.so"; \
    ldconfig; \
    TORCH_INC="$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__), "include"))')"; \
    test -d "${TORCH_INC}/fmt" || cp -r /opt/fmt_headers_from_torch2_11 "${TORCH_INC}/fmt"; \
    ls "${TORCH_INC}/fmt/base.h" >/dev/null && echo "fmt headers OK"; \
    pip uninstall -y --break-system-packages nvidia-cuda-runtime-cu12 || true; \
    echo "/usr/local/cuda-12.9/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/00-system-cuda.conf; \
    ldconfig; \
    cd /opt/DeepEP; \
    NCCL_LIB_DIR="${NCCL_LIB}"; \
    export LIBRARY_PATH="${NCCL_LIB_DIR:-}:${LIBRARY_PATH:-}"; \
    export TORCH_CUDA_ARCH_LIST="9.0"; \
    export MAX_JOBS="$(nproc)"; \
    set +e; \
    pip install --no-build-isolation --break-system-packages --force-reinstall --no-deps -e . > /tmp/deep_ep_build.log 2>&1; \
    DEEP_EP_RC=$?; \
    tail -30 /tmp/deep_ep_build.log; \
    echo "deep_ep rebuild rc=${DEEP_EP_RC}"; \
    python3 -c "import deep_ep; print('ElasticBuffer:', deep_ep.ElasticBuffer)" || \
      echo "WARN: deep_ep import deferred to runtime (pod will recompile against torch 2.7.1 + cuda)"; \
    true

# -----------------------------------------------------------------------------
# 5. api-shim + harness. The shim monkeypatches deep_ep.Buffer ->
#    CompatBuffer which delegates to ElasticBuffer. TRT-LLM 0.21's
#    `tensorrt_llm.deep_ep.Buffer(None, ..., comm=)` call lands on the
#    shim, bypasses NVSHMEM, routes cross-node via NCCL-Gin.
# -----------------------------------------------------------------------------
RUN mkdir -p /opt/api-shim/api_shim /opt/harness
COPY api-shim/api_shim/__init__.py          /opt/api-shim/api_shim/__init__.py
COPY api-shim/api_shim/buffer_v1_compat.py  /opt/api-shim/api_shim/buffer_v1_compat.py
COPY api-shim/sitecustomize.py              /opt/api-shim/sitecustomize.py
COPY scripts/shared/test_shim_v1_surface.py /opt/harness/test_shim_v1_surface.py
COPY scripts/shared/verify_efa_traffic.sh   /opt/harness/verify_efa_traffic.sh
RUN chmod +x /opt/harness/verify_efa_traffic.sh

COPY integrations/trtllm-deepep-v2/docker/entrypoint.sh /usr/local/bin/trtllm-deepep-v2-entrypoint.sh
RUN chmod +x /usr/local/bin/trtllm-deepep-v2-entrypoint.sh

COPY integrations/trtllm-deepep-v2/smoke_test_shim.py /opt/smoke_test_shim.py

# -----------------------------------------------------------------------------
# V2 runtime ENV block (mirrors private tree). DEEP_EP_USE_V2_SHIM=1
# activates sitecustomize.py shim install at Python interpreter start.
# -----------------------------------------------------------------------------
ENV PYTHONPATH=/opt/api-shim \
    DEEP_EP_USE_V2_SHIM=1 \
    DEEP_EP_BACKEND=nccl \
    TRTLLM_CAN_USE_DEEP_EP=1 \
    NCCL_NET_PLUGIN=ofi \
    NCCL_GIN_TYPE=2 \
    NCCL_GIN_ENABLE=1 \
    NCCL_NVLS_ENABLE=0 \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_EFA_FORK_SAFE=1 \
    FI_EFA_ENABLE_SHM_TRANSFER=0 \
    OFI_NCCL_PROTOCOL=RDMA \
    OFI_NCCL_GIN_MAX_REQUESTS=512 \
    EP_EFA_MAX_QPS=2 \
    EP_EFA_RDMA_GBS=25.0

LABEL org.opencontainers.image.title="trtllm-deepep-v2-efa" \
      org.opencontainers.image.description="TensorRT-LLM 0.21 + DeepEP V2 via api-shim + AWS EFA (Wave 34.8)." \
      org.opencontainers.image.source="https://github.com/antonai-work/trtllm-deepep-v2-efa" \
      org.opencontainers.image.licenses="MIT" \
      status.wave="34.8" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      deepep.pr="612" \
      trtllm.version="${TRTLLM_VERSION}" \
      shim.decision="Option B - torch PG, see docs/trtllm-comm-decision.md in private tree"

CMD ["/bin/bash"]
