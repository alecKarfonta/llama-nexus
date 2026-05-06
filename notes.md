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
- **Problem**: `docker compose --profile vllm up -d vllm-api` ran inside backend container which has no compose file.
- **Fix**: Changed to `docker start vllm-api` / `docker stop vllm-api` which works via the Docker socket mount.

### 2. Frontend 502 errors after backend restart (FIXED)
- **Problem**: When backend-api container is recreated, it gets a new IP. Nginx in frontend container had stale DNS cache.
- **Fix**: `docker compose restart llamacpp-frontend` refreshes DNS. The nginx resolver has `valid=10s` but the upstream variable caching in nginx can persist longer.

### 3. Chat empty response with vLLM backend (FIXED)
- **Problem**: Chat page used wrong model name for vLLM. The `/v1/models/current` endpoint only returned the llama.cpp manager config (`NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD`) instead of the vLLM served model name (`Nemotron-3-Nano-Omni-30B-A3B-Reasoning`). vLLM rejected requests with a 404 "model does not exist" error, but the streaming proxy didn't detect the error, so the frontend got an empty stream.
- **Fix** (four parts):
  1. `backend/routes/models.py` (`/v1/models/current`): Now checks vLLM first and returns the vLLM served model name when vLLM is active.
  2. `backend/routes/service.py` (`_proxy_stream`): Added preflight status check so error responses from the backend are returned as proper HTTP errors, not silently streamed.
  3. `backend/routes/service.py` (`proxy_chat_completions`): Inject vLLM API key from config when client doesn't provide one (fixes 401 Unauthorized).
  4. `frontend/src/pages/ChatPage.tsx`: Always syncs the model name from the API response (instead of only setting it when no model was previously cached), so stale model names from a different backend don't persist.

### 4. vLLM command preview and config in DeployPage (DONE)
- **What**: When the backend selector is switched to "vLLM", the deploy page now loads the vLLM config and shows the vLLM launch command in the command preview.
- **Changes**:
  - `backend/modules/managers/vllm_manager.py`: Added `build_command()` that generates the `vllm serve ...` command from config. Added `sampling` section to config (temperature, top_p, etc.).
  - `backend/routes/service.py`: `GET /api/v1/service/config` and `POST /api/v1/service/config/preview` now accept `backend` query param and return the correct manager's config and command. Service action applies config for vLLM too (deep merge for nested dicts).
  - `frontend/src/pages/DeployPage.tsx`: `updateCommandPreview` passes `backend` field. `useEffect` reloads config when `backend` changes. Command preview is backend-specific.

### 5. vLLM-specific Deploy tabs and cross-framework mapping (DONE)
- **What**: With **vLLM** selected, Deploy shows dedicated tabs (Model, Sampling, Performance, MoE & Reasoning, Tools & Speculative, Environment, **vLLM Version**, Server, Command Line). llama.cpp-only tabs (Context Extension, LlamaCPP Version, GPU layers, etc.) are hidden. Shared concepts use green **Shared** badges; vLLM-only fields use blue **vLLM** badges. An info alert summarizes rough llama.cpp ⇄ vLLM parameter mapping.
- **Backend**:
  - `vllm_manager.py`: Expanded default config (`dtype`, `quantization`, parallelism, chunked prefill, async scheduling, MoE, speculative JSON, environment mirror). `build_command()` emits matching CLI flags. `get_field_metadata()` documents scopes and equivalents.
  - `llamacpp_manager.py`: `get_field_metadata()` for llama.cpp scopes and vLLM equivalents.
  - `service.py`: `GET /api/v1/service/config/fields?backend=…`, `PUT /api/v1/service/config?backend=vllm` (deep merge into `VLLMManager.config`). Alias `PUT /v1/service/config` accepts `backend` query param.
- **Frontend**: `DeployPage.tsx` loads field metadata; `vllmConfig` + `originalVllmConfig`; Save/Start/Restart/Clear All wired for vLLM; tab index is **clamped** when switching backends (llama.cpp 0–6, vLLM 0–8); command preview is tab **8** for vLLM; `scrollMarginTop` on vLLM cards to reduce overlap when scrolling past fixed header.

### 6. vLLM Version tab (Docker base image) — DONE
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

### vLLM model picker matches Models catalog
- **Problem**: vLLM only exposed free-text HF repo id, not the same name/variant dropdowns as llama.cpp from `/v1/models`.
- **Fix**: DeployPage uses shared **Model Name** / **Model Variant** selects (same `models` list). Choosing an entry sets `vllmConfig.model.name` from `repositoryId` when the backend supplies it. **Backend**: GGUF entries pick up `repositoryId` from `.metadata` JSON when present; transformers dirs expose repo id via `.hf_repo_id` written at download time (existing dirs need re-download or manual file). Frontend maps `framework` from API.

## Testing Checklist
- [x] Backend builds: `docker compose build backend-api`
- [x] Frontend builds: `docker compose build llamacpp-frontend`
- [x] vLLM start/stop/restart via API: all return success
- [x] `/api/v1/service/backends` shows both backends with correct status
- [x] `/api/v1/service/vllm/status` shows vLLM status
- [x] Frontend backend selector UI visible in toolbar
- [x] `/v1/models` via nginx returns 200
- [ ] Deploy with llama.cpp backend from UI (needs llamacpp-api container)
- [ ] Deploy with vLLM backend from UI (end-to-end test)
