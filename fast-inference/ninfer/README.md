# fast-inference/ninfer

[NInfer](https://github.com/Neroued/ninfer) — from-scratch C++/CUDA inference engine for
registered Qwen checkpoints on a single NVIDIA GeForce RTX 5090 (`sm_120a`, Blackwell).
Serves OpenAI-compatible (and Anthropic Messages) HTTP via `ninfer-serve`.

## Build

```bash
docker build --tag llama-nexus-ninfer-api:latest ./fast-inference/ninfer
```

Build args: `NINFER_REF` (git ref, default `master`), `CUDA_VERSION` (default `13.1.2`).
Requires a host driver reporting CUDA 13.1+ (590.44.01+; host now runs 595.84).

The image clones upstream at build time so picks up the latest fixes;
no vendored source is kept in this repo.

## Artifacts

NInfer runs only its registered version-2 `.ninfer` artifacts (not GGUF):

| Artifact | Repo |
|---|---|
| `qwen3_8_27b_nvfp4.ninfer` (20.02 GiB) | `neroued/Qwen3.8-27B-nvfp4-NInfer` |
| `qwen3_8_27b.ninfer` (16.96 GiB) | `neroued/Qwen3.8-27B-NInfer` |
| `qwen3_6_27b_nvfp4.ninfer` (17.07 GiB) | `neroued/Qwen3.6-27B-nvfp4-NInfer` |
| `qwen3_6_27b.ninfer` (16.29 GiB) | `neroued/Qwen3.6-27B-NInfer` |
| `qwen3_6_35b_a3b.ninfer` (21.22 GiB) | `neroued/Qwen3.6-35B-A3B-NInfer` |

Download one via the Models page (`.ninfer` is allowed) or:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models
```

Artifacts land in the shared `gpt_oss_models` volume (backend sees
`/home/llamacpp/models`, this container mounts it at `/models`).

## Custom artifacts (Huihui abliterated)

The tools in this directory build a `.ninfer` artifact from an arbitrary
same-architecture BF16 checkpoint (tested with
`huihui-ai/Huihui-Qwen3.8-27B-abliterated`). The engine only checks the
identity strings (`qwen3.8-27b` + `nvfp4`), so custom weights load unchanged.

Pipeline (all inside the `ninfer-convert` image, `fast-inference/ninfer/Dockerfile.convert`):

1. `fetch_unsloth_scalars.py` — HTTP-range-fetches the calibrated
   `input_global_scale` scalars from `unsloth/Qwen3.8-27B-NVFP4` (no 16 GB download).
2. `requant_huihui_ninfer_layout.py` — requantizes BF16 into the registered layout:
   FP8 row-scaled attention/GDN/lm_head/late-MLP + NVFP4 group-16 MLP (layers 0-55,
   joint gate/up global scale). Semantics verified bit-level against unsloth.
3. `convert_huihui_nvfp4.py` — runs the upstream dual-source converter unmodified
   for all numeric work (only provenance constants are repointed).

Result on this host: `qwen3_8_27b_nvfp4.ninfer` (20.02 GiB) with Huihui weights —
162 TPS single-stream, 839 TPS at C=8. The official artifact is preserved at
`qwen3_8_27b_nvfp4.ninfer.official-backup` in the models volume.


## Run

Managed by llama-nexus (Deploy page → backend **ninfer**, port 8603) or manually:

```bash
docker compose --profile ninfer up -d ninfer-api
curl http://127.0.0.1:8603/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with one short sentence."}],"max_tokens":64}'
```

Vision (image/video) requires the server to start with `--vision` — the Deploy
page wires this to the **Vision** toggle and it cannot be enabled later.
