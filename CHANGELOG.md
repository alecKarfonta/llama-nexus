# Changelog

All notable changes to Llama Nexus are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
semantic versioning.

## [Unreleased]

### Added
- Real Settings page (server-backed): HuggingFace token management (persisted
  on the backend data volume, masked in transit, re-applied to the runtime
  environment on boot), masked view of the effective inference/embedding API
  keys, storage-path inventory with free-space reporting, and a LAN-exposure
  toggle that rewrites `BIND_HOST` in `.env` and reports when a stack restart
  is required. New endpoints: `GET /api/v1/settings`, `PUT
  /api/v1/settings/hf-token`, `PUT /api/v1/settings/lan`, `GET
  /api/v1/settings/management-token/suggest`.

## [0.1.0] — 2026-10-15

### Added
- One-command installers: `install.sh` (Linux/macOS/WSL2) and `install.ps1`
  (Windows) — hardware detection, tier selection, .env generation, prebuilt
  image pull, stack start.
- Prebuilt image distribution: `.github/workflows/build-images.yml` publishes
  backend, frontend, and llama.cpp CPU/CUDA (86/89/120) variants to GHCR;
  `docker-compose.prebuilt.yml` consumes them with zero compile steps.
- CI: `.github/workflows/ci.yml` runs backend pytest, frontend type-check +
  tests, and compose portability validation on every PR.
- Qdrant vector DB lifecycle management: `QdrantManager` + `/api/v1/qdrant/*`
  routes (start/stop/restart/status/config) — RAG no longer needs manual
  compose profiles.
- Curated model catalog: `backend/data/model_catalog_seed.json` seeds the
  registry with recommended LLM/embedding/STT models on first run;
  `/api/v1/registry/recommendations` returns hardware-matched picks.
- "Chat · Starter (4B)" one-click deploy preset — portable default preset
  that runs on CPU or any 4GB+ GPU.
- First-run onboarding wizard in the web UI (hardware summary, use-case
  picker, guided model download).
- Voice mode toggle in Chat — enables the fully-local STT → LLM → TTS loop.
- Simple/Advanced navigation mode: new users see core pages only; the sidebar
  toggle reveals all expert tools.
- Optional management-API authentication: set `MANAGEMENT_AUTH_TOKEN` to
  require a bearer token on all management routes.
- `scripts/detect-cuda-arch.sh` — maps host GPU compute capability to the
  CUDA_ARCH build value.
- This CHANGELOG and the MIT LICENSE file.

### Changed
- `docker-compose.yml` is now portable: named Docker volumes instead of
  host-specific bind mounts; GPU access moved to the `docker-compose.gpu.yml`
  overlay; rig-specific model/threading defaults moved to
  `docker-compose.rig.yml`; generic single-GPU/CPU-safe defaults in the base.
- `CUDA_ARCH` build default is 86 (Ampere, broadly compatible) instead of 120
  (Blackwell-only).
- Frontend/backend ports bind to 127.0.0.1 by default; set `BIND_HOST=0.0.0.0`
  to expose on the LAN.
- TTS/STT/embedding/streaming-STT configurations persist under the backend
  data directory instead of `/tmp` (previously lost on every restart).
- Backend resolves the compose project root dynamically (`local_paths.project_root()`)
  instead of hardcoding `/home/alec/git/llama-nexus`.
- Frontend nginx proxy injects the real `API_KEY` at container start instead
  of a hardcoded placeholder.
- llama.cpp builds use upstream official tags (b10723+); the MTP fork/patch
  is an opt-in build-arg flavor.

### Fixed
- `docker compose up` no longer fails on machines without the author's
  USB-NVMe mount paths.
- Deploy presets no longer assume 44 threads / GPU index 2 / 4-GPU splits.

## [1.0.0] — internal development builds
- Core platform: llama.cpp orchestration, model downloads, chat, RAG,
  GraphRAG, workflows, fine-tuning, quantization, benchmarks, MTP research.
