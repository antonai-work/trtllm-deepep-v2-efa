# trtllm-deepep-v2-efa

TensorRT-LLM + DeepEP V2 + AWS EFA MoE inference cascade.

**Status:** SCAFFOLD. TRT-LLM DeepEP fast path was validated in Wave 4
on the v0.2.3 base (4x512 tokens, `AlltoallMethodType=DeepEP`);
revalidation on the current v0.2.5 base is Wave 30 in the private
`deepep-v2-integration` dev tree. This repo lands the cascade-symmetric
source shape so CI/CodeBuild pipelines can hydrate once Wave 30 produces
PROVEN evidence.

## Image cascade

```
nvidia/cuda:13.0.0-devel-ubuntu24.04                   (public)
  └── deepep-v2-efa-base:v0.2.5-sm90a                  (base: NCCL patched, EFA, DeepEP V2)
        └── trtllm-deepep-v2-efa:<tag>                 (this repo, Wave 30 target)
```

Parent base: https://github.com/antonai-work/deepep-v2-efa-base

## Sibling repos (reproducibility cascade)

| Repo | Purpose | Status |
|---|---|---|
| [deepep-v2-efa-base](https://github.com/antonai-work/deepep-v2-efa-base) | Base substrate (EFA + NCCL + DeepEP V2) | v0.2.5-sm90a released |
| [vllm-deepep-v2-efa](https://github.com/antonai-work/vllm-deepep-v2-efa) | vLLM inference stack | Wave 26b PROVEN |
| [megatron-deepep-v2-efa](https://github.com/antonai-work/megatron-deepep-v2-efa) | Megatron-LM training | Wave 27 PROVEN |
| [nemo-rl-deepep-v2-efa](https://github.com/antonai-work/nemo-rl-deepep-v2-efa) | NeMo-RL + Megatron full-stack | Wave 28 PROVEN |
| [sglang-deepep-v2-efa](https://github.com/antonai-work/sglang-deepep-v2-efa) | SGLang inference | Wave 29 pending |
| **trtllm-deepep-v2-efa** | **TRT-LLM inference (this repo)** | **Wave 30 pending** |

## Upstream PRs consumed

| Upstream | PR / Issue | HEAD SHA | Status |
|---|---|---|---|
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | [#612](https://github.com/deepseek-ai/DeepEP/pull/612) | `146cc356` | OPEN (baked into base) |
| TensorRT-LLM release | `v0.21.0` | (pip) | GA |

Full pinned versions in [`pins.env`](pins.env).

## Build

```bash
docker build -f docker/Dockerfile --build-arg BUILD_MODE=fast \
             -t trtllm-deepep-v2-efa:fast .
```

Image will be rebuilt by CodeBuild once Wave 30 lands the TRT-LLM
overlay body.

## Licensing

MIT. TensorRT-LLM itself is under Apache-2.0 (NVIDIA). DeepEP under MIT (DeepSeek).
