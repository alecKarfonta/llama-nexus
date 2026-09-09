# Notes

## 2026-08-27 Qwen3.8-Flash-Next (qwen4exp) deploy prep

**Model:** `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-Q4_K_XL` (~111 GB, 4-shard split) + `mmproj-F16.gguf` (vision, 0.9 GB). New `qwen4exp` arch (Qwen4 preview: 125B MoE A6B + 51B n-gram PLE + QSA sparse attention, 262K native ctx).

**No `Q4_K_M` exists** in this repo — Unsloth ships Dynamic 2.0 quants only. 4-bit choices: `UD-Q4_K_XL` (111 GB) or `UD-IQ4_XS` (94 GB). K-quant chosen over IQ because IQ runs slow on CPU and this deploy offloads some experts to CPU.

**llama.cpp support is PR-only:** [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) (unsloth, opened 2026-08-26, still open). Not in any b-tagged release (latest b10656 is 2026-08-27 and does not contain it). `Dockerfile` `LLAMA_CPP_REF` now supports `pull/N/head` refs; pinned to `pull/27742/head`, build tag `b10631-qwen4exp`. Images must be rebuilt from source — prebuilt release download path cannot work for this model.

**Offload design (the "cpu offload" ask):**
- The 51B n-gram (PLE) table (~33 GB in UD-Q4_K_XL) is *designed* to live in **host RAM via mmap** — host-side row-index lookups, near-zero compute. Don't pass `--no-mmap`/`--mlock`; the pages stay cold/evictable.
- Remaining ~78 GB GPU weights + <1 GB KV (q8_0 at 64K ctx; only 12/48 layers hold KV thanks to QSA sparse attention + GDN linear attention).
- Rig state at deploy: GPU2 held by moss-tts (16.2 GB) → default deploy uses **GPUs 0,1,3** (~71 GB free) + `n_cpu_moe=4` (~6 GB experts on CPU). Free GPU2 (stop moss-tts) to go all-4-GPU with `n_cpu_moe=0`.
- `.env` `N_CPU_MOE=21` is the backend default for the old 27B model — override per-deploy via Deploy UI or the config PUT; don't rely on env default.

**Day-0 PR gotchas (from PR thread):** `--split-mode tensor` aborts on this arch (keep `layer`); multi-slot `--parallel` hits an indexer-cache assert (keep `parallel_slots=1`); `--fit on` hits `GGML_ASSERT(obj_new)` (don't use); garbage output on some rigs → reduce `gpu_layers`; load aborts → `LLAMA_ATTN_ROT_DISABLE=1`. The `--override-tensor` trick isn't wired into the backend (no `override_tensor` config key) — `n_cpu_moe` is the supported offload knob.

**Code changes for this deploy:**
- `Dockerfile` — `pull/N/head` ref support + PR pin (see above).
- `docker-compose.yml` — builder + llamacpp-api build args bumped to `pull/27742/head` / `b10631-qwen4exp`.
- `llamacpp_manager.py` `build_command()` — glob fallback for `{variant}/{name}-{variant}-00001-of-*.gguf` (any shard count; hardcoded list only knew 2/3-shard).
- `download_manager.py` `_parse_name_variant()` — handle `UD-*`/`IQ*` variant suffixes (`Qwen3.8-Flash-Next-UD-Q4_K_XL` → name `Qwen3.8-Flash-Next`, variant `UD-Q4_K_XL`) and strip subdir prefixes.
- `scripts/start.sh` — same glob fallback + 4-shard patterns.
- `scripts/deploy_qwen38_flash.sh` (new) — end-to-end deploy via backend API with offload presets (`GPU_SET`/`CPU_MOE`/`CTX` env overrides).

**Sampling (from model card):** thinking mode `temp 1.0 / top_p 0.95 / top_k 20 / min_p 0 / presence 0 / repeat 1.0`; instruct mode `temp 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5`. Disable thinking via `chat_template_kwargs: {"enable_thinking": false}`.

**Files in `gpt_oss_models` volume:** `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf` + `mmproj-F16.gguf`.

**Working deploy config (verified 2026-08-27, ~18 tok/s gen, 3-GPU + CPU offload):**

| knob | value | why |
|---|---|---|
| GPUs | 0,1,3 | GPU 2 held by moss-tts (16.2 GB) |
| `override_tensor` | `per_layer_token_embd=CPU,blk[.](3|7|...|46)[.].*_exps[.]weight=CPU` | PLE table (~33 GB) to host RAM; every other layer's MoE experts (~30 GB) to CPU — evens out across the 3-way layer split |
| `n_cpu_moe` | 0 | `-ot` handles expert offload instead (per-layer spread) |
| `gpu_layers` | 999 | all layers on GPU (experts via `-ot` are separate) |
| `split_mode` | layer | `tensor` mode aborts on qwen4exp (PR known issue) |
| `parallel_slots` | 1 | multi-slot hits indexer-cache assert (PR known issue) |
| ctx | 65536 | q8_0 KV; 262K native but KV grows with ctx |
| sampling | temp 1.0 / top_p 0.95 / top_k 20 | thinking mode (model card) |

**Balance math (why `-ot`, not `--fit`):** `--fit` refuses to combine with `-ot` ("tensor_buft_overrides already set by user, abort"). Even 3-way layer split puts 16 layers/GPU ≈ 26-27 GB on devices with 22-24 GB free. Offloading every other layer's `*_exps.weight` (~1.26 GB/layer) via regex sidesteps both problems. Expert tensor names: `blk.N.ffn_{up,down,gate}_exps.weight` — note **no dot** between `ffn` and `_exps`. PLE tensor: `per_layer_token_embd.weight` (dims 160×320M, ~33 GB q4).

**Known-issue workarounds applied:** none of `LLAMA_ATTN_ROT_DISABLE` etc. needed on this rig — clean load on build `b10716-e36130336` (PR head rebased). `--image-min-tokens 1024` warning appears for the VL path but text works.

**Deploy/verify:** `./scripts/deploy_qwen38_flash.sh` (env: `GPU_SET`, `CPU_MOE`, `CTX`, `OVERRIDE_TENSOR`, `GPU_LAYERS_CFG`). Frontend Deploy page also works — the backend now wires `override_tensor` (`--override-tensor`) as a config key. Vision path untested — text-only verified.

## 2026-08-30 TPS tuning: MTP depth, poll, tensor-split, ubatch (4×3090 Ti, all-GPU)

**Question:** what else lifts decode TPS now that MTP + cache-reuse are on? Ran a 5-experiment series with a fixed harness (`scripts/bench_tps.sh`: prose/code/JSON × 2 runs, temp 0, thinking off; `scripts/qwen_set.sh` to push config + restart + wait for health).

**Experiment A — MTP draft depth (`--spec-draft-n-max`) / strictness (`p_min`):**

| config | prose | code | JSON | STRUCT avg | acceptance |
|---|---|---|---|---|---|
| n3 / p.75 (start) | 34.9 | 65.4 | 73.0 | 69.2 | 0.98, len 3.8 |
| n5 / p.75 | 32.7 | 69.4 | 88.3 | 78.8 | 0.98, len 5.3 |
| n8 / p.75 | 33.5 | 79.4 | 101.3 | 90.3 | 0.99, len 8.0 |
| **n16 / p.75** | **35.8** | **79.9** | **103.4** | **91.7** | 0.88, len 9.7 |
| n16 / p.90 | 35.3 | 79.8 | 101.8 | 90.8 | 0.90, len 9.2 |

Depth is the dominant knob — JSON nearly +42% from n3→n16. Acceptance stays ≥0.88 even at depth 16; p_min 0.9 buys nothing over 0.75. **Winner: n_max=16, p_min=0.75.**

**Experiment B — busy-poll (`--poll 100 --spec-draft-poll 1`):** +4% across all workloads (prose 37.2, code 82.2, JSON 107.9; STRUCT 95.1). Wired `poll`/`draft_poll` into `mtp` config (`llamacpp_manager.py` `build_command` + `normalize_mtp_config` in `mtp_deploy.py`). Kept.

**Experiment C — `tensor_split 0.85,1.1,1.0,1.05`:** evens VRAM 23.7/20.3→22.0/22.0 GB, code +4%, prose +2%. Kept — also freed ~2.5 GB on GPU0, which unblocked Experiment E.

**Experiment D — power/clock check (`nvidia-smi dmon -s pc` under load):** 50-129 W of a 350 W limit, clocks pinned 1860-1935 MHz, temps ≤50°C. **No thermal/power throttling — workload is launch/latency-bound, not power-bound. Power-limit tinkering is pointless on this rig.**

**Experiment E — drop `mmproj` (vision tower, 0.9 GB) + retry `ubatch_size 1024`:** without the vision tower, ubatch 1024 fits (previously OOM'd on GPU0). Decode flat, **cold prefill 905→996 tok/s (+10%)**. `mmproj: null` + ubatch 1024 kept.

**Final config (applied, verified via public endpoint):** `draft_n_max 16, draft_p_min 0.75, poll 100, draft_poll 1, tensor_split 0.85,1.1,1.0,1.05, mmproj off, ubatch 1024` → prose 38.9 / code 85.4 / JSON 106.3 tok/s (STRUCT 95.9 vs 69.2 at session start, **+39%**; decode ~76.9 avg all-workload vs 57.8).

**New tooling:** `scripts/bench_tps.sh` (fixed 3-workload decode bench), `scripts/qwen_set.sh` (config push + restart + health-wait helper), `scripts/bench_prefill.py` (cold-prefill tok/s — superseded by
`scripts/bench_prefill.sh`, which adds a nonce so the probe cannot be served
from the prompt cache; the `.py` reports cache hits, not prefill).

## 2026-08-30 (round 2) p_min drift fix, parallel slots unlocked, ctx 128k

**Round-2 findings (continuation of the 5-experiment series above):**

- **Drift bug found:** the round-1 "final" actually ran `draft_p_min 0.9` (a leftover from experiment A's last variant), not the declared 0.75 winner. Proper A/B under the full combo: 0.9 → STRUCT 95.9 vs 0.75 → 90.9. **0.9 is the real winner** — the round-1 matrix underestimated it because p_min variants were tested at n16 only after poll/split were applied... reversed: with poll+split+ubatch the stricter p_min pays. Keep 0.9.
- **`--parallel 2` now works** on the patched MTP build — the day-0 indexer-cache assert is gone. Single-stream TPS unchanged (decode serializes) but concurrency improves: 2 streams → **aggregate 65.6-68.7 tok/s vs 59.7 queued** (+10-15%), wall -20%. Note `--parallel N` divides per-slot ctx.
- **ctx 65536 → 131072** fits with mmproj dropped (max 23.8/24.5 GB) — restores full 64k per slot under `--parallel 2`. Decode unchanged.
- **threads 46 vs 40:** wash (PLE table lookups don't parallelize further; 40 kept).
- **cache-reuse window 128 vs 512:** no TTFT difference on agent-style reuse (exact-prefix prompt cache does the work); 128 kept.

**Current live config:** `draft_n_max 16, draft_p_min 0.9, poll 100, draft_poll 1, tensor_split 0.85,1.1,1.0,1.05, parallel 2, ctx 131072 (64k/slot), ubatch 1024, threads 40, cache_reuse 128, mmproj off`. Decode: prose ~37 / code ~77 / JSON ~104-107; 2-stream aggregate ~68; cold prefill ~996 tok/s; warm agent TTFT ~1.25s (5.7x reuse gap).

**New tooling (round 2):** `scripts/bench_concur.sh` (N-stream concurrent throughput + TTFT), `scripts/bench_ttft.sh` (agent-style cold/warm TTFT with reuse-gap ratio).

## 2026-08-30 (round 3) Cursor-specific MTP draft head — capture pipeline foundation

**Goal:** raise MTP draft acceptance on Cursor traffic (~0.83-0.91 today) toward 0.95+ by training the 4B draft head on this server's own outputs on Cursor-shaped requests (self-distillation). The trained artifact is a sidecar GGUF swap (`--model-draft`) — zero serving-code changes, instant rollback.

**Pipeline (all live):**

```
Cursor → https://mlapi.us/cursor/v1 (new key sk-mlapi-cursor-*)
      → nginx (192.168.50.4) /cursor/ location → capture-tee (rig :8601)
      → llama-server (:8600);  tee writes /data/captures/captures-YYYY-MM-DD.jsonl
      → scripts/dataset_build.py → train/val shards → (stage 3) LoRA on 4B head
```

**Pieces built:**
- `backend/capture_tee.py` — stdlib-only streaming reverse proxy. SSE passes through chunk-by-chunk (TTFT untouched; verified parity). After stream end it tees a record: request (messages/tools/sampling), response (content/tool_calls/finish_reason), usage/timings, and **exact server-side prompt token IDs** via `/apply-template` + `/tokenize` (template_ok flag marks unverified). Capture runs in a background thread — never on the response path. Client keys are swapped for the llama `--api-key` upstream (env `TEE_UPSTREAM_API_KEY`); records keep only the client key id.
- `docker-compose.yml` — `capture-tee` service (profile `extra`, port 8601, `datasets` volume). Note: `llamacpp-api` has no DNS alias on the bridge network; upstream is the host gateway `172.18.0.1:8600`.
- nginx (remote) — `cursor.conf` location + `host196_capturetee` upstream; key appended to `/etc/nginx/api_keys`. **Gotcha: `nginx-key-auth` re-reads the file per request, but a stale systemd unit needed `sudo systemctl restart nginx-key-auth` before the new key validated.**
- `scripts/dataset_build.py` — filters (template_ok only, finish in stop/tool_calls, min chars), dedupes (exact sample hash + per tool-schema-prefix cap — Cursor resends identical tool lists every turn), optional PII scrub, 98/2 train/val shards + build_report.json. Run inside the tee container (volume needs root): `docker cp scripts/dataset_build.py llamacpp-capture-tee:/ && docker exec llamacpp-capture-tee python3 /dataset_build.py --scrub`.
- `scripts/acceptance_log.py` — acceptance trending from `docker logs llamacpp-api` (llama-server logs to **stderr**; acceptance can be exactly `1.00000`). `tail` / `trend` / `live --csv`. **Baseline 2026-08-30: 0.832 avg acceptance, mean len 6.28** — this is the number the retrained head must beat.

**Verified:** non-stream + SSE through the full public path, bad-key 401, TTFT parity, records with token-exact prompt ids landing in the volume, builder round-trip (534-char completion → kept → val shard).

**Cursor config:** base URL `https://mlapi.us/cursor/v1`, model `qwen38-flash`, dedicated `sk-mlapi-cursor-*` key (stored only on the rig / nginx, not in git). The `/qwen/` route stays capture-free for non-Cursor use.

**Stage 3 (training window, not yet run):** stop `llamacpp-api` → LoRA r=32 on the BF16 head export (`dzannotti/Qwen3.8-Flash-Next-MTP-BF16.gguf` → safetensors, verify `blk.48.*` 34 tensors) on 2× 3090 Ti → merge → `convert_hf_to_gguf.py --mtp` on the patched tree → `llama-quantize` Q4_K_M (Q4_K output/embedding) → drop into models volume → swap `speculative.model_draft`, restart, gate: acceptance ≥0.93 and STRUCT bench not regressed.




## 2026-06-18 Multi-GPU deploy only attached one GPU (FIXED)

**Symptom:** Last `Qwen3.6-27B-Q6_K` deploy targeting host GPUs 1+3 (both RTX 5060 Ti) crashed with `cudaMalloc failed: out of memory` — llama.cpp log only listed `CUDA0` (single GPU) instead of `CUDA0`+`CUDA1`, so the ~21 GB Q6_K couldn't split across two 16 GB cards.

**Root cause:** `_docker_gpu_attachment` (`backend/modules/managers/llamacpp_manager.py`) attached GPUs via `DeviceRequest(device_ids=[...])` (SDK) / `--gpus '"device=1,3"'` (CLI), but also set `CUDA_VISIBLE_DEVICES=1,3` and `NVIDIA_VISIBLE_DEVICES=1,3` to the *host* indices. The NVIDIA Container Toolkit remaps the requested host GPUs to **internal** indices `0..N-1`, so `CUDA_VISIBLE_DEVICES=1,3` selects internal index `1` (the 2nd attached GPU) plus a non-existent internal index `3` → CUDA ends up with just one device. Same bug duplicated in `embedding_manager.py` (SDK + CLI paths).

**Repro on this rig:** `docker run --rm --entrypoint bash --gpus '"device=1,3"' -e CUDA_VISIBLE_DEVICES=1,3 ...` → llama-server's `device_info` shows only `CUDA0`. Same command *without* `CUDA_VISIBLE_DEVICES` → shows `CUDA0` + `CUDA1`. Confirmed before/after the fix.

**Fix:** drop the host-index `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` env from `_docker_gpu_attachment` and from `embedding_manager` (both paths). Keep `NVIDIA_DRIVER_CAPABILITIES=compute,utility` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`. `STT`/`TTS` managers left alone — they use `--runtime nvidia` + `NVIDIA_VISIBLE_DEVICES` only (no `--gpus`/`DeviceRequest`), so the value is honored as a host filter and these services are single-GPU anyway. Tests in `backend/tests/test_gpu_attachment.py`.

## 2026-06-09 Deploy UI MTP false-negative

**Symptom:** Uncensored heretic deploy blocked with `mtp_capable=false` even though MTP switch looked off.

**Root cause:** Qwen3.6-moe family presets default `mtp.enabled=true`. UI switch used `checked={enabled && canEnable}` so non-MTP GGUFs showed OFF while saved config still had `enabled: true` and command included `--spec-type draft-mtp`. Switch was disabled — user could not turn it off.

**Fix:** `clampMtpForModelCapability` on save/preview/init; backend clamps on merge + `build_command`; `resolveMtpForModel` respects `mtpCapable`.

**Immediate unblock:** `PUT /api/v1/service/config` with body `{"mtp":{"enabled":false}}` (raw config, not wrapped).

## Current Goal

**fast-inference experiment** (`fast-inference/`): MTP speculative decoding with Qwen3.6 on the 4×16GB rig (2×5070 Ti + 2×5060 Ti). Using `llama-nexus-llamacpp-embed:latest` (b9193, has `draft-mtp`) on port **8603**, GPUs 0+1 layer-split. Downloaded `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (22GB MTP) to `/home/alec/models/qwen3.6-35b-mtp/`.

**First A/B (2026-06-08, ctx=65536, NMAX=3, 2 runs each):**
| Profile | baseline | MTP | speedup |
|---|---|---|---|
| code | 83.7 tok/s | 109.1 | **1.30×** |
| structured | 84.7 | 116.6 | **1.38×** |
| chat (temp 0.7) | 83.1 | 90.1 | **1.08×** |

Container: `fast-inference-mtp` (currently baseline / MTP off). Re-enable MTP: restart with `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75`.

**2026-06-08:** Stopped `speaker-moss-sfx-1` (MOSS-SoundEffect on GPU 3, ~15.8GB) to free VRAM for inference experiments. Restart: `cd /home/alec/git/speaker && docker compose --profile moss-sfx start moss-sfx`.

**2026-06-08 27B 3-GPU experiment:** GPUs **0+1+3** via `--gpus '"device=0,1,3"'` (CUDA_VISIBLE_DEVICES alone only bound 2 GPUs!). Avoid GPU 2 (moss-realtime). Port **8603**, container `fast-inference-27b`.

| Config | code tok/s | structured | chat | vs baseline |
|---|---|---|---|---|
| Q6_K, 2 GPU (accidental) | 22.2 | 22.2 | 22.2 | baseline |
| Q6_K, 3 GPU | 20.0 | 20.0 | 20.1 | 0.90× (PCIe sync cost) |
| Q4_K_M MTP, 3 GPU, MTP off | 25.2 | 25.2 | 25.1 | 1.14× |
| **Q4_K_M MTP, 3 GPU, MTP on** | **38.0** | **45.6** | **33.6** | **1.7–2.0×** |

**Winner (baseline):** `Qwen3.6-27B-Q4_K_M.gguf` (MTP) on GPUs 0,1,3 — ctx=180k, parallel=2, `NMAX=4`, q8_0 KV. ~38–46 tok/s interactive. Still slower than 35B-A3B MoE (~90–110 tok/s) but usable for quality/coding workloads.

**2026-06-08 fast-inference 27B NMAX matrix** (`bench/run_matrix.sh`, GPUs 0,1,3, ctx=180k):

| label | code | structured | chat | vs baseline |
|---|---|---|---|---|
| baseline (MTP off) | 25.2 | 25.1 | 25.0 | 1.00× |
| **mtp-n2** | **43.3** | **44.1** | **38.6** | **1.54–1.76×** |
| mtp-n3 | 42.7 | 42.0 | 36.8 | 1.47–1.69× |
| mtp-n4 | 38.9 | 45.1 | 34.1 | 1.36–1.80× |
| mtp-n4-p80 | 41.0 | 44.1 | 33.4 | 1.34–1.76× |

**New 27B baseline:** `NMAX=2` (not 4) — narrower draft window wins on chat/code for this rig. Server on :8603 with mtp-n2 settings.

**2026-06-08 fast-inference 35B NMAX matrix** (GPUs 0,1, ctx=65536, `35b-*.json`):

| label | code | structured | chat | vs baseline |
|---|---|---|---|---|
| 35b-baseline | 83.8 | 85.1 | 84.3 | 1.00× |
| **35b-mtp-n2** | **109.0** | 115.6 | **100.3** | **1.19–1.36×** |
| 35b-mtp-n3 | 91.9 | **126.1** | 97.5 | 1.16–1.48× |
| 35b-mtp-n4 | 109.3 | 122.4 | 83.7 | 0.99–1.44× |
| 35b-mtp-n4-p80 | 106.3 | 114.8 | 70.2 | 0.83–1.35× |

MoE: `NMAX=2` best overall; `NMAX=3` peaks structured; high `PMIN` hurts chat. Re-run: `cd fast-inference/bench && MODEL_SIZE=35b DOCKER=1 GPU_DEVICES=0,1 PORT=8604 RESULT_PREFIX=35b- ./run_matrix.sh`

**2026-06-08 grammar+MTP tool-call experiment** (`bench/run_grammar_mtp.sh`): lazy tool grammar ~28 tok/s; **MTP n8 → 87 tok/s** (3.1×), 100% valid tool calls, 87.5% draft acceptance. On tool path use **NMAX=8** not n2.

**2026-06-08 fast-inference extras** (`bench/run_extras.sh`): 2-GPU beats 3-GPU (~49 vs 44 tok/s MTP); KV q8_0≈f16; ngram-simple useless vs MTP (1.0× vs 1.7×); conc=4 same tok/s but TTFT ~17–19s vs ~0.4s. **2×5070+5060 on 0,1 is faster than 3-GPU** for 27B Q4_K_M.

Primary inference (when not experimenting): **Qwen3.6-27B** (GGUF via llama.cpp / Deploy backend **llama.cpp**), matching `docker-compose.yml` (`MODEL_REPO=unsloth/Qwen3.6-27B-GGUF`, variant **Q6_K**).

Secondary: vLLM (profile `vllm`) where GPU supports it; bundled Nemotron NVFP4 requires **compute capability ≥ 8.9** (not RTX 3090-class Ampere).

### Homelab nginx (mlapi.us modular, stocker repo)

- Added **reader**: `https://mlapi.us/reader/` proxies to `http://192.168.1.77:8012/` via `stocker/config/nginx-modular/apps/reader.conf` and an include in `mlapi.us.conf`. Deploy with `install.sh` or copy the new file and reload nginx (`nginx -t && systemctl reload nginx`).

## What We Did

### Backend
- **VLLMManager** (`backend/modules/managers/vllm_manager.py`): Manages vLLM container lifecycle via docker compose. Start/stop/restart use `docker compose --profile vllm up/stop`. Health checks via httpx to `http://vllm-api:8080/health`.
- **app_state.py**: `vllm_manager` singleton instantiated alongside `manager`.
- **app_lifespan.py**: `app.state.vllm_manager = vllm_manager` wired in lifespan.
- **service routes** (`backend/routes/service.py`):
  - `get_backend_manager(request, backend)` helper dispatches to the right manager.
  - `POST /api/v1/service/action` now accepts `backend` field (`llamacpp` or `vllm`).
  - `GET /api/v1/service/backends` returns status of both backends.
  - `GET /api/v1/service/vllm/status` returns vLLM-specific status.

### Frontend
- **ServiceActionRequest type** updated with `backend?: 'llamacpp' | 'vllm'`.
- **api.ts** added `getBackendsStatus()` and `getVllmStatus()` methods.
- **DeployPage.tsx**:
  - `backend` state restores from `localStorage` (`deployBackend`), default `'llamacpp'`.
  - Backend selector ToggleButtonGroup added to toolbar (llama.cpp | vLLM).
  - `runAction` passes `backend` field to `performServiceAction`.
  - vLLM status fetched during initialization.

## Architecture Notes
- vLLM runs as an external Docker container managed by docker-compose (profile: `vllm`), not spawned by the backend.
- The backend container mounts the repo at `/home/alec/git/llama-nexus` (read-only). `VLLMManager.start()` prefers `docker compose -f …/docker-compose.yml --profile vllm up -d --force-recreate vllm-api` from `PROJECT_DIR` so Deploy settings can drive compose substitutions (e.g. `GPU_MEMORY_UTILIZATION`). Falls back to `docker start vllm-api` only when compose is unavailable or fails.
- vLLM config is merged in-memory on `POST /api/v1/service/action` (start/restart) and via `PUT /api/v1/service/config?backend=vllm`. Launch-time env for `vllm-api` is built by `VLLMManager.compose_launch_environment()` and merged into the `docker compose` process env so YAML `${VAR:-default}` substitutions match Deploy (see `scripts/start-vllm.sh` for the CLI mapping).

## Issues Found and Fixed

### 2026-05-16: Redis compose port conflict on 6381 (FIXED)
- **Problem**: `docker compose up -d --build` failed while starting `llama-nexus-redis` because host port `6381` was already allocated.
- **Fix**: Changed the Redis host port mapping in `docker-compose.yml` from `6381:6379` to `6382:6379`. Internal service URLs remain `redis://redis:6379/0`, so containers still talk to Redis on the Docker network normally.
- **Retry**: Run `docker compose up -d --build` again from the repo root.

### Deploy launch: "Docker image not found: llama-nexus-llamacpp-api" (FIXED)
- **Problem**: Inference image is built from the `llamacpp-api` Compose service; plain `docker compose build` without naming that service skips profile-only services, so the image never existed. SDK path also mounted the wrong volume name (`llamacpp-api_gpt_oss_models`).
- **Fix**: `docker-compose.yml` sets explicit `image:` for `llamacpp-api`. Backend reads `LLAMACPP_DOCKER_IMAGE` and derives `LLAMACPP_MODELS_VOLUME` from `DOCKER_NETWORK` (strip `_default`, append `_gpt_oss_models`). Error text now says to run `docker compose build llamacpp-api`.
- **User action**: From repo root once: `docker compose build llamacpp-api`, then `docker compose build backend-api && docker compose up -d backend-api` (or full stack) so the backend picks up env/code. The image name is `llama-nexus-llamacpp-api:latest`; the backend must use the **same Docker daemon** where that image was built (typical when using the mounted Docker socket).
- **2026-05-06**: Ran `docker compose build llamacpp-api` successfully (~11 min); `docker images` shows `llama-nexus-llamacpp-api:latest`.

### llama.cpp Docker build: `nvcc fatal : Unsupported gpu architecture 'compute_'` (FIXED)
- **Problem**: Dockerfile used `CUDA_ARCH=native` (via compose default). During `docker build` the builder typically has **no GPU**, so CMake’s native SM detection is empty and nvcc gets `compute_`.
- **Fix**: Default `CUDA_ARCH` to `86` (Ampere / RTX 30xx) in `Dockerfile` and `docker-compose.yml`. Override in `.env` for other GPUs (e.g. `89` for RTX 4090, `90` for H100).

### vLLM model selection at top of Deploy (UX)
- **Problem**: GGUF dropdown only applies to llama.cpp; vLLM model fields lived only under the **Model** tab, so it felt like there was no model selector.
- **Change**: With vLLM selected, **Currently Deployed** now includes editable **Model (HF id)**, **Served model name**, **dtype**, and **quantization** (same `updateVllmConfig` paths as the Model tab), plus an **Open full Model tab** link.

### vLLM Deploy tabs not switching / blank panels (FIXED)
- **Cause**: MUI `Tabs` kept internal scroll/selection state when switching llama.cpp ↔ vLLM (different child counts). Partial API payloads could omit nested sections (e.g. `sampling`), so `vllmConfig.sampling[key]` threw and broke updates below the tab strip.
- **Fix**: `key={backend}` on `Tabs`; reset `tab` to `0` when `backend` changes; clamp indices with `DEPLOY_TAB_MAX_*`; wrap tab strip in `Box` with higher `z-index`; merge API config with `mergeVllmApiWithDefaults()` into `VLLM_DEFAULT_VALUES` for complete nested objects.

### DeployPage blank tabs / crash with vLLM (FIXED)
- **Cause**: The GGUF model/template card always accessed `config.model.*`. With **vLLM** selected, `config.model` could be missing or incomplete (for example from persisted settings), which threw during render and prevented everything below—including tab panels—from mounting.
- **Fix**: Render GGUF picker, templates, and "Download more models" only when `backend === 'llamacpp'`. For vLLM, show a short HF model summary card instead. Validate `cfgJson.config` after a successful vLLM config fetch. Reset `vllmReloading` when switching away from vLLM so the loading banner cannot stick. Clamp deploy tab index when switching backends so `Tabs` `value` stays within range (llama.cpp 0–6, vLLM 0–8).

### VLLMManager compose vs `docker start` (FIXED + evolved)
- **Earlier**: `docker compose --profile vllm up -d vllm-api` failed inside backend (no compose file); fallback was `docker start` / `docker stop` only.
- **Current**: `PROJECT_DIR` is mounted into `backend-api`; `VLLMManager.start()` runs compose from the repo first so Deploy-driven env applies; compose failure still falls back to `docker start`.

### 3. Frontend 502 errors after backend restart (FIXED)
- **Problem**: When backend-api container is recreated, it gets a new IP. Nginx in frontend container had stale DNS cache.
- **Fix**: `docker compose restart llamacpp-frontend` refreshes DNS. The nginx resolver has `valid=10s` but the upstream variable caching in nginx can persist longer.

### Chat token speed not measured with vLLM (FIXED)
- **Cause**: Metrics only counted `delta.content`; Nemotron/vLLM streams in `delta.reasoning_content`. llama.cpp `timings.predicted_per_second` is not sent by vLLM.
- **Fix**: Count reasoning deltas; request `stream_options.include_usage`; use `usage.completion_tokens` when present; compute tok/s over generation phase (after first token).

### Chat page wrong port with vLLM (FIXED)
- **Problem**: Chat used saved `baseUrl` `http://localhost:8600` or Vite dev proxied all `/v1/*` to llama.cpp (8600). vLLM listens on 8601, so chat missed the active backend unless users manually picked the vLLM preset.
- **Fix**: `ChatPage` clears direct `:8600`/`:8601` base URLs on load and routes through `/v1/chat/completions` on the current origin (nginx/vite → backend proxy → active backend). Vite dev proxy now sends `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/service` to backend :8700 (fixed wrong `/api/v1/models` rewrite).

### Chat empty response with vLLM backend (FIXED)
- **Problem**: Chat page used wrong model name for vLLM. The `/v1/models/current` endpoint only returned the llama.cpp manager config (`NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD`) instead of the vLLM served model name (`Nemotron-3-Nano-Omni-30B-A3B-Reasoning`). vLLM rejected requests with a 404 "model does not exist" error, but the streaming proxy didn't detect the error, so the frontend got an empty stream.
- **Fix** (four parts):
  1. `backend/routes/models.py` (`/v1/models/current`): Now checks vLLM first and returns the vLLM served model name when vLLM is active.
  2. `backend/routes/service.py` (`_proxy_stream`): Added preflight status check so error responses from the backend are returned as proper HTTP errors, not silently streamed.
  3. `backend/routes/service.py` (`proxy_chat_completions`): Inject vLLM API key from config when client doesn't provide one (fixes 401 Unauthorized).
  4. `frontend/src/pages/ChatPage.tsx`: Always syncs the model name from the API response (instead of only setting it when no model was previously cached), so stale model names from a different backend don't persist.

### 5. Model Manager delete returned 404 for `/v1/models/7` (FIXED)
- **Problem**: Active downloads merged into the models list in `ModelsPage.tsx` used a synthetic numeric `id` (`prev.length + …`). Delete called `DELETE /v1/models/7`; the backend expects a model key (file stem or `name:variant`), so it returned 404 (no files for model `7`).
- **Fix**: Synthetic rows now use `id: download.modelId` (matches backend download / filesystem stem). Delete confirmation maps legacy numeric `id` to `${name}:${variant|quantization}`. Start/stop handlers accept `string | number` for `id`.

### 2026-05-13: Inference choice — Qwen3.6-27B (llama.cpp)
- Use Deploy/backend backend **llama.cpp**, not vLLM default Nemotron, on Ampere (e.g. RTX 3090 Ti): Nemotron NVFP4 needs **sm_89+**.
- Stack: `docker compose up -d` (core), `docker start llamacpp-api` or `docker compose --profile extra up -d llamacpp-api`, ensure GGUF present (`Qwen3.6-27B-Q6_K.gguf` on shared volume).

### 2026-05-11: UI errors / 502 (not backend crash)
- **Symptom**: Deploy page and API calls failed; nginx logs showed `connect() failed (111: Connection refused)` to `http://172.19.0.6:8700/...`.
- **Cause**: `llamacpp-backend` was healthy at **172.19.0.2**, but nginx had a **stale upstream** (old container IP after recreate). Same class of issue as the "Frontend 502 errors after backend restart" note above.
- **Fix**: `docker compose restart llamacpp-frontend`. Verified `GET /v1/service/status` via `:3002` returns 200.

### vLLM Version tab (Docker base image) — DONE
- **What**: With **vLLM** selected, tab **vLLM Version** mirrors LlamaCPP Version: live GitHub data (`vllm-project/vllm` releases + recent commits), validate ref, apply by rewriting **`Dockerfile.vllm`** `FROM vllm/vllm-openai:<tag>`, rebuild **`vllm-api`** via `docker compose --profile vllm up -d --build vllm-api`.
- **Backend** (`routes/llamacpp.py`): `GET /api/v1/vllm/image-versions`, `GET /api/v1/vllm/image-tag/{ref}/validate`, `POST /api/v1/vllm/image-tag/{tag}/apply`, `POST /api/v1/vllm/rebuild`.
- **Compose**: `backend-api` mounts `./Dockerfile.vllm` read-write at `/home/alec/git/llama-nexus/Dockerfile.vllm` (same pattern as main `Dockerfile`).
- **Frontend**: `LlamaCppCommitSelector` accepts `variant="vllm"`; `api.ts` wrappers for the new endpoints. Deploy tabs: Environment (5), **vLLM Version (6)**, Server (7), Command Line (8).

### DeployPage persistence (llama.cpp vs vLLM)
- **Problem**: Only llama.cpp `config` and `selectedApiKey` were written to `llama-nexus-deploy-settings`; `saveDeploySettings` replaced the whole blob, so the inference backend toggle was never stored and vLLM field edits were lost on navigation.
- **Fix**: `readDeployStorage` / `writeDeployStorage` merge patches into the same JSON object. Persist `deployBackend` on toggle and `vllmConfig` on edit / Clear All. Initialize `backend` from `loadDeploySettings().deployBackend`. When loading vLLM config after navigation, use `persisted.vllmConfig ?? server payload`, then `mergeVllmApiWithDefaults` (same idea as llama.cpp `persisted.config || server`).

### Deploy vLLM tabs not switching (FIXED)
- **Cause**: `Tabs` used raw `tab` while llama.cpp exposes indices 0–6 and vLLM 0–8. After switching backends, `tab` could exceed the new backend’s max for a render; MUI `Tabs` with an out-of-range `value` stops handling clicks reliably. `scrollButtons="auto"` could also overlap tabs on some widths.
- **Fix**: `activeDeployTab = clamp(tab, 0, deployTabMax)` drives `Tabs` and every panel; `useLayoutEffect` resets tab when `backend` changes; explicit numeric `value` on each `Tab`; `scrollButtons={false}`; higher tab-strip `zIndex`; `deployLog('deployTabs', …)` on change.

### vLLM `gpu_memory_utilization` showed 0.95 after setting 0.90 (FIXED)
- **Cause**: Deploy saved `performance.gpu_memory_utilization` on the manager and in the command preview, but `vllm-api` was started with `docker start` / compose **without** passing that value; `docker-compose.yml` hardcoded `GPU_MEMORY_UTILIZATION=0.95`. `docker start` cannot change a container’s env.
- **Fix (expanded)**: All Deploy-backed vLLM fields map through `VLLMManager.compose_launch_environment()` → `docker compose` substitutions → `scripts/start-vllm.sh` (`dtype`, quantization, parallelism toggles, MoE, tools, speculative JSON, multimodal `LIMIT_MM_PER_PROMPT`, Deploy Environment/HF token, etc.). `start()` runs compose with `--force-recreate` from `PROJECT_DIR` first; `/api/v1/vllm/rebuild` uses the same env merge. Port mapping: `${VLLM_PUBLISHED_PORT:-8601}:${PORT:-8080}` so internal `PORT` can track Deploy **Server** settings while the host side stays on 8601 unless `VLLM_PUBLISHED_PORT` is set.
- **UX note**: The Deploy **VRAM Utilization** bar under “VRAM Estimation” is a **read-only** estimate from the estimator API, not `gpu_memory_utilization`. The editable field is **Performance → gpu_memory_utilization** and uses a **0.0–1.0** fraction (e.g. `0.9` for 90%), not `90`.

### vLLM model picker matches Models catalog
- **Problem**: vLLM only exposed free-text HF repo id, not the same name/variant dropdowns as llama.cpp from `/v1/models`.
- **Fix**: DeployPage uses shared **Model Name** / **Model Variant** selects (same `models` list). Choosing an entry sets `vllmConfig.model.name` from `repositoryId` when the backend supplies it. **Backend**: GGUF entries pick up `repositoryId` from `.metadata` JSON when present; transformers dirs expose repo id via `.hf_repo_id` written at download time (existing dirs need re-download or manual file). Frontend maps `framework` from API.

### vLLM Deploy control from UI (reliability)
- **Persist**: `VLLMManager` loads/saves `/data/vllm_deploy_config.json` (`VLLM_DEPLOY_CONFIG_PATH`) so backend restart keeps Deploy settings; PUT config and service actions persist after merge.
- **Merge**: `POST /api/v1/service/action` uses `_deep_merge_config` for vLLM (same as PUT), not one-level dict merge.
- **Errors**: Failed start/stop/restart returns HTTP 500 with `detail` including `last_action_error` (compose/docker stderr). Fallback `docker start` after compose failure sets a warning on `last_action_error`.
- **Compose**: `backend-api` sets `PROJECT_DIR=/home/alec/git/llama-nexus` explicitly (alongside `USE_DOCKER=true`).
- **Frontend**: Server vLLM config wins over localStorage when loading Deploy (so persisted server state applies); chips show running/stopped + deploy warning tooltip; polls `/api/v1/service/vllm/status` every 5s on vLLM tab; Start/Stop/Restart surfaces FastAPI `detail` via axios error parsing.
- **Single-active backend**: `POST /api/v1/service/action` now stops the opposite backend before `start`/`restart`, preventing simultaneous llama.cpp + vLLM deployments that compete for GPU memory.

## Testing Checklist
- [x] Backend builds: `docker compose build backend-api`
- [x] Frontend builds: `docker compose build llamacpp-frontend`
- [x] vLLM start/stop/restart via API: all return success
- [x] `/api/v1/service/backends` shows both backends with correct status
- [x] `/api/v1/service/vllm/status` shows vLLM status
- [x] Frontend backend selector UI visible in toolbar
- [x] `/v1/models` via nginx returns 200
- [ ] Deploy with llama.cpp backend from UI (inference image built; retry Deploy start)
- [ ] Deploy with vLLM backend from UI (end-to-end test)

## Git / rebuild (2026-06-07)

- `git pull`: already up to date.
- Full `docker compose … up -d --build` with all profiles failed mid-build (~18 min) during parallel `llamacpp-api` CUDA compile + image export.
- Retry without blocking on `extra` profile succeeded: rebuilt and recreated `backend-api`, `llamacpp-frontend`, `llamacpp-embed`, graphrag NER/REL, training, quantization, lm-eval, terminal-bench. Verified `GET /v1/service/status` (:8700) and frontend (:3002).
- `llama-nexus-llamacpp-api:latest` image present from prior/partial build; inference container not started in this pass.

## UI integration Phase 1 (2026-06-08)

**Implemented:** MTP workload profiles (`chat` | `agent` | `throughput` | `custom`) end-to-end.

- Backend: `mtp_deploy.py` — `apply_mtp_workload_profile()`, `chat_template_kwargs_cli_value()`, agent/chat presets
- Backend: `llamacpp_manager.py` — `--chat-template-kwargs`, profile merge in `build_command` / validation
- Frontend: `MtpConfigSection` workload ToggleButtonGroup, n_max slider 1–8, bench hints
- Frontend: `mtpWorkloadProfiles.ts`, `mtpRecommendedDefaults.json` (qwen3.6 chat n2)
- Tests: `backend/tests/test_mtp_deploy.py` — all passing

**Exit check:** Deploy Agent profile → n8, `enable_thinking: false`, `cache_reuse: 1024`, `parallel_slots: 1`.

## UI integration Phase 2 (2026-06-08)

**Implemented:** Agent & tools server config + Chat tools hints + proxy audit.

- `AgentToolsConfigSection` on Deploy Model tab — JSON editor for `chat_template_kwargs`, reasoning_budget, cache_reuse, lazy-grammar info panel
- `chat_proxy.py` + `test_chat_proxy.py` — passthrough contract for `tools`, `tool_choice`, `response_format`, `grammar`
- `service.py` proxy — parse body once via `parse_chat_proxy_body`; document passthrough
- ChatPage — "Tools mode" chip in header; Deploy Agent profile hint when tools enabled on llama.cpp
- `get_config` editable_fields extended for agent server + MTP workload fields

## UI integration Phase 3 (2026-06-08)

**Implemented:** Structured output from Chat + llguidance build path.

- Chat: `ConstraintEditor` wired — toggle, JSON schema + GBNF on requests (`response_format` + `grammar`)
- `chatOutputConstraints.ts` — builds OpenAI-compatible `json_schema` payload
- Warning when tools + structured output both enabled (single grammar slot)
- `ConstraintEditor` GBNF tab now editable (raw grammar advanced mode)
- Dockerfile: opt-in `ENABLE_LLGUIDANCE=true` → `-DLLAMA_LLGUIDANCE=ON` + capability file
- Status API: `llguidance_enabled`, `grammar_gbnf_supported` on `/v1/service/status`
- Deploy Agent & Tools: GBNF / llguidance capability chips

**Next:** Phase 4 GPU presets; optional `ENABLE_LLGUIDANCE=true` rebuild for jump-forward tier.

## llamacpp-api b10089 (2026-06-09)

- Pinned `Dockerfile` + `docker-compose.yml` to latest release **b10089** (was b9193).
- Default `CUDA_ARCH=120` for RTX 5090 (Blackwell sm_120).
- Rebuilt: `docker compose build --no-cache llamacpp-api` (~18 min).
- Start on single-GPU host: `CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES=0 docker compose --profile extra up -d llamacpp-api`
- Status: `mtp_supported: true`, `build_number: 10089`.

## llamacpp-api b9193 (2026-06-08)

- Root cause: `llama-nexus-llamacpp-api:latest` image missing; backend docker-py client failed → build info null
- Immediate fix: tagged `llama-nexus-llamacpp-embed:latest` → `llama-nexus-llamacpp-api:latest` (b9193)
- Backend fix: `_read_llamacpp_build_from_docker_image` subprocess fallback when docker-py unavailable
- Dockerfile: `CMAKE_BUILD_JOBS=2` default (was `-j8`) to avoid CUDA compile OOM
- Full `llamacpp-api` source rebuild in progress with `CMAKE_BUILD_JOBS=2`

## Chat page layout fix (2026-06-09)

**Bug:** On narrow viewports (sidebar open), message input textarea collapsed to ~13px wide — placeholder stacked vertically character-by-character.

**Cause:** Flex row with `fullWidth` TextField + `flexGrow: 1` + Send `minWidth: 100`; `flex-basis: auto` let the field shrink below usable width.

**Fix:** `flex: 1 1 0`, `minWidth: 0`, `width: 0` on TextField; `flexShrink: 0` on icon/send buttons; icon-only Send on xs; `minWidth: 0` on main content in `App.tsx`.

## Deploy failure (2026-06-09)

**Symptom:** `llamacpp-api` OOM on model load with `ctx-size 48000`, `cuda_devices 0,2`, `main_gpu 2`, no `--tensor-split`.

**Root cause (not context size):**
- `docker-compose.yml` had `CUDA_DEVICES=0,2` and `MAIN_GPU=2`. GPU 2 hosts `speaker-moss-realtime` (~12GB) → CUDA1 in container had ~3.5GB free; llama tried ~16.4GB on CUDA0 alone.
- Lowering ctx to 48k did not help — failure was **weight allocation**, not KV cache.
- 256k ctx works only on **free GPUs 0+1** with `--tensor-split` + q8_0 KV (experiments: 48k trivial, 128k comfortable, ~250k ceiling).

**Fixes applied:**
- Compose defaults: `CUDA_DEVICES=0,1`, `MAIN_GPU=0`, `TENSOR_SPLIT=1,1`.
- `_docker_gpu_attachment`: `device_ids` + quoted CLI `--gpus '"device=0,1"'` (bare `device=0,1` breaks Docker).
- `apply_mtp_workload_profile`: use `setdefault` so chat profile does not overwrite `mtp.enabled: false`.

**2026-06-09 MTP graft (heretic Q6_K):** Safe offline job via `scripts/run_mtp_graft_heretic.sh` — isolated `python:3.12-slim` container, `ionice -c3 nice -n19`, never touches `llamacpp-api`. Donor: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` → `Qwen3.6-35B-A3B-Q8_0.gguf`. Output: `Qwen3.6-35B-A3B-uncensored-heretic-Q6_K-MTP.gguf`. Deploy symlink: `…-heretic-MTP-Q6_K.gguf`.

**2026-06-09 Heretic MTP live benchmark** (`:8600`, grafted file, MTP chat n2 vs off):

| Metric | MTP off | MTP on | Notes |
|--------|---------|--------|-------|
| compare_live decode tok/s | 170.1 | 227.6 | **1.34×** |
| compare_live completion_tokens | 128 | 128 | same prompt |
| bench code tok/s | 184.8 | 210.6 | 1.14× |
| bench structured tok/s | 186.7 | 220.5 | 1.18× |
| bench chat tok/s | 186.8 | 173.9 | 0.93× (acceptance dipped on persona prompt) |
| seed=42 add(a,b) tokens | 18 | 18 | **content match: yes** |
| draft acceptance (15 samples) | n/a | mean **76.9%** | min 33% max 100% |

Results: `results/heretic-graft-mtp-{off,on}.json`, `results/heretic-output-mtp-{off,on}.json`, `fast-inference/bench/results/heretic-mtp-{off,on}.json`. Verdict: graft is **lossless** on fixed-seed output; **worth enabling** for code/structured; chat profile may need n3 or accept lower acceptance on graft.

## MTP graft UI (2026-06-09)

- Backend: `MtpGraftManager` runs graft in isolated `python:3.12-slim` Docker (never in `llamacpp-api`). API: `POST/GET/DELETE /v1/models/mtp-graft*`, preflight + donor suggestions.
- Frontend: `MtpGraftDialog` on Models page (**Add MTP** on non-MTP models); Deploy MTP section links to graft flow.
- Output naming: `{name}-MTP-{variant}.gguf` for deploy `{name}-{variant}.gguf` resolution.
- Rebuild: `docker compose --profile extra up -d --build backend-api llamacpp-frontend`

## mlapi.us ingress (2026-06-09)

- **Edge host:** node-4 `192.168.1.4` (nginx active, TLS on `:8443`; router NAT `:443` → `:8443`).
- **llama.cpp API:** `https://mlapi.us/vllm/` → sonic `192.168.1.78:8600` (upstream `host77_vllm`). Added `proxy_buffering off` for streaming.
- **llama-nexus UI:** `https://mlapi.us/nexus/` → sonic `:3002` (upstream `host78_llama_nexus`). Config: `/etc/nginx/conf.d/apps/llama-nexus.conf` on node-4.
- **Caveat:** UI built for `/` — static assets under `/nexus/` may 404 (`/assets/...` not prefixed). Subdomain or Vite `base: '/nexus/'` rebuild fixes that.

## Git (2026-06-08)

- Pulled latest `origin/master` (`1e8377f` — MTP deploy/chat integration, Dockerfile/compose updates).
- Stashed local WIP (mmproj download feature) before pull; stash: `WIP before pull latest`.
- Rebase conflict in `backend/routes/service.py` resolved: use `_inject_inference_api_key()` for llama.cpp proxy paths (matches vLLM path).
- Local commit rebased on top: `fb8f20d` Fix Chat 401 when llama.cpp deploy API key is unset.
- Rebuilt: `docker compose --profile extra up -d --build redis backend-api llamacpp-frontend llamacpp-api`.

## Git (2026-05-11)

- Merged `origin/master` into local `master`; resolved conflicts in `frontend/src/pages/ChatPage.tsx` and `docs/archive/development-journal-2024-2026.md`; merged `notes.md` sections (deploy/CUDA + vLLM UX fixes).
- Merge commit: `d30135b`. Pushed to `origin` (`master`).
- `origin` was switched from HTTPS to SSH (`git@github.com:alecKarfonta/llama-nexus.git`) because HTTPS push failed with no stored credentials in this environment.

## Chat UI max tokens (2026-06-09)

- Raised Chat settings slider max from 4K → 128K (default remains 8K). Matches workflow OpenAI node limit.


