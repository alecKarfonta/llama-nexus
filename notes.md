# Notes

## Current Goal: vLLM Backend Support

Add vLLM as an alternative inference backend alongside llama.cpp, with full lifecycle management through the DeployPage UI.

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
- VLLMManager uses `docker start/stop <container_name>` directly (not `docker compose`), since the backend container doesn't have the compose file.
- vLLM config is merged in-memory on `POST /api/v1/service/action` (start/restart) and can be saved via `PUT /api/v1/service/config?backend=vllm`. Docker-compose env vars are still the source of truth for production unless you mirror them into the manager config / compose file.

## Issues Found and Fixed

<<<<<<< HEAD
### vLLM model selection at top of Deploy (UX)
- **Problem**: GGUF dropdown only applies to llama.cpp; vLLM model fields lived only under the **Model** tab, so it felt like there was no model selector.
- **Change**: With vLLM selected, **Currently Deployed** now includes editable **Model (HF id)**, **Served model name**, **dtype**, and **quantization** (same `updateVllmConfig` paths as the Model tab), plus an **Open full Model tab** link.

### vLLM Deploy tabs not switching / blank panels (FIXED)
- **Cause**: MUI `Tabs` kept internal scroll/selection state when switching llama.cpp ↔ vLLM (different child counts). Partial API payloads could omit nested sections (e.g. `sampling`), so `vllmConfig.sampling[key]` threw and broke updates below the tab strip.
- **Fix**: `key={backend}` on `Tabs`; reset `tab` to `0` when `backend` changes; clamp indices with `DEPLOY_TAB_MAX_*`; wrap tab strip in `Box` with higher `z-index`; merge API config with `mergeVllmApiWithDefaults()` into `VLLM_DEFAULT_VALUES` for complete nested objects.

### DeployPage blank tabs / crash with vLLM (FIXED)
- **Cause**: The GGUF model/template card always accessed `config.model.*`. With **vLLM** selected, `config.model` could be missing or incomplete (for example from persisted settings), which threw during render and prevented everything below—including tab panels—from mounting.
- **Fix**: Render GGUF picker, templates, and “Download more models” only when `backend === 'llamacpp'`. For vLLM, show a short HF model summary card instead. Validate `cfgJson.config` after a successful vLLM config fetch. Reset `vllmReloading` when switching away from vLLM so the loading banner cannot stick. Clamp deploy tab index when switching backends so `Tabs` `value` stays within range (llama.cpp 0–6, vLLM 0–8).

### 1. VLLMManager `docker compose` not found (FIXED)
=======
### 0. Deploy launch: "Docker image not found: llama-nexus-llamacpp-api" (FIXED)
- **Problem**: Inference image is built from the `llamacpp-api` Compose service; plain `docker compose build` without naming that service skips profile-only services, so the image never existed. SDK path also mounted the wrong volume name (`llamacpp-api_gpt_oss_models`).
- **Fix**: `docker-compose.yml` sets explicit `image:` for `llamacpp-api`. Backend reads `LLAMACPP_DOCKER_IMAGE` and derives `LLAMACPP_MODELS_VOLUME` from `DOCKER_NETWORK` (strip `_default`, append `_gpt_oss_models`). Error text now says to run `docker compose build llamacpp-api`.
- **User action**: From repo root once: `docker compose build llamacpp-api`, then `docker compose build backend-api && docker compose up -d backend-api` (or full stack) so the backend picks up env/code. The image name is `llama-nexus-llamacpp-api:latest`; the backend must use the **same Docker daemon** where that image was built (typical when using the mounted Docker socket).
- **2026-05-06**: Ran `docker compose build llamacpp-api` successfully (~11 min); `docker images` shows `llama-nexus-llamacpp-api:latest`.

### 1. llama.cpp Docker build: `nvcc fatal : Unsupported gpu architecture 'compute_'` (FIXED)
- **Problem**: Dockerfile used `CUDA_ARCH=native` (via compose default). During `docker build` the builder typically has **no GPU**, so CMake’s native SM detection is empty and nvcc gets `compute_`.
- **Fix**: Default `CUDA_ARCH` to `86` (Ampere / RTX 30xx) in `Dockerfile` and `docker-compose.yml`. Override in `.env` for other GPUs (e.g. `89` for RTX 4090, `90` for H100).

### 2. VLLMManager `docker compose` not found (FIXED)
>>>>>>> 875f5f5 (fix(llamacpp): deploy image and models volume config)
- **Problem**: `docker compose --profile vllm up -d vllm-api` ran inside backend container which has no compose file.
- **Fix**: Changed to `docker start vllm-api` / `docker stop vllm-api` which works via the Docker socket mount.

### 3. Frontend 502 errors after backend restart (FIXED)
- **Problem**: When backend-api container is recreated, it gets a new IP. Nginx in frontend container had stale DNS cache.
- **Fix**: `docker compose restart llamacpp-frontend` refreshes DNS. The nginx resolver has `valid=10s` but the upstream variable caching in nginx can persist longer.

### 4. Chat empty response with vLLM backend (FIXED)
- **Problem**: Chat page used wrong model name for vLLM. The `/v1/models/current` endpoint only returned the llama.cpp manager config (`NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD`) instead of the vLLM served model name (`Nemotron-3-Nano-Omni-30B-A3B-Reasoning`). vLLM rejected requests with a 404 "model does not exist" error, but the streaming proxy didn't detect the error, so the frontend got an empty stream.
- **Fix** (four parts):
  1. `backend/routes/models.py` (`/v1/models/current`): Now checks vLLM first and returns the vLLM served model name when vLLM is active.
  2. `backend/routes/service.py` (`_proxy_stream`): Added preflight status check so error responses from the backend are returned as proper HTTP errors, not silently streamed.
  3. `backend/routes/service.py` (`proxy_chat_completions`): Inject vLLM API key from config when client doesn't provide one (fixes 401 Unauthorized).
  4. `frontend/src/pages/ChatPage.tsx`: Always syncs the model name from the API response (instead of only setting it when no model was previously cached), so stale model names from a different backend don't persist.

### 5. Model Manager delete returned 404 for `/v1/models/7` (FIXED)
- **Problem**: Active downloads merged into the models list in `ModelsPage.tsx` used a synthetic numeric `id` (`prev.length + …`). Delete called `DELETE /v1/models/7`; the backend expects a model key (file stem or `name:variant`), so it returned 404 (no files for model `7`).
- **Fix**: Synthetic rows now use `id: download.modelId` (matches backend download / filesystem stem). Delete confirmation maps legacy numeric `id` to `${name}:${variant|quantization}`. Start/stop handlers accept `string | number` for `id`.

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
