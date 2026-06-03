# Notes

## Current Goal

Primary inference: **Qwen3.6-27B** (GGUF via llama.cpp / Deploy backend **llama.cpp**), matching `docker-compose.yml` (`MODEL_REPO=unsloth/Qwen3.6-27B-GGUF`, variant **Q6_K**).

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

## Git (2026-05-11)

- Merged `origin/master` into local `master`; resolved conflicts in `frontend/src/pages/ChatPage.tsx` and `docs/archive/development-journal-2024-2026.md`; merged `notes.md` sections (deploy/CUDA + vLLM UX fixes).
- Merge commit: `d30135b`. Pushed to `origin` (`master`).
- `origin` was switched from HTTPS to SSH (`git@github.com:alecKarfonta/llama-nexus.git`) because HTTPS push failed with no stored credentials in this environment.




