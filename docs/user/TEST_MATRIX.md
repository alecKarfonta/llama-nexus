# Platform Test Matrix

[2026-10-16] | INFO | DOCS.TEST_MATRIX | Milestone 3.2 validation record.
How the stack behaves on each support tier, what was tested, and known gaps.

## Tiers

| Tier | Platform | llamacpp image | Status |
|------|----------|----------------|--------|
| 1 | Linux + NVIDIA (CUDA 8.6/8.9/12.0) | `cuda-86` / `cuda-89` / `cuda-120` | ✅ Tested |
| 2 | Windows 10/11 + WSL2 + Docker Desktop | tier by GPU (cpu or cuda-*) | ✅ Tested (installer logic) |
| 3a | macOS Apple Silicon (M1/M2/M3) | `cpu` (native arm64) | ✅ Build-verified |
| 3b | macOS Intel | `cpu` (amd64) | ⚠️ Untested on hardware |
| — | Linux CPU-only | `cpu` | ✅ CI-verified (GH runners) |

## What was tested, where

### Ubuntu + NVIDIA (Tier 1) — primary dev rig
- Fresh `install.sh` run: GPU tier detection (compute cap → cuda-120), `.env`
  generation (random `API_KEY` + `NEO4J_PASSWORD`), prebuilt pull, stack start.
- Wizard flow on the live stack: hardware detect → use-case → download →
  **auto-deploy** → auto-land in Chat with ready banner.
- `scripts/smoke_e2e.sh` core: **10/10 pass** (health, registry seed, models,
  Qdrant lifecycle, service status routes).
- `scripts/smoke_e2e.sh --full`: chat round-trip verified; STT/TTS/embed/RAG
  stages contract-tested against mock backends (25/25) — full live run is
  wired into CI (`smoke-e2e` job) and runs on every PR.
- CUDA image builds for all three arches publish from CI (11/11 jobs green).

### Ubuntu CPU (GitHub Actions runners)
- The `smoke-e2e` CI job boots the published cpu-tier stack from GHCR on a
  clean runner and runs the full pipeline — this is the continuously-tested
  CPU tier.

### Windows WSL2 (Tier 2)
- `install.ps1` path: Docker Desktop detection, GPU capability query via
  `nvidia-smi` inside WSL, tier selection, `.env` generation, compose up —
  logic reviewed and dry-run; no physical Windows box in the loop yet.
- Known gap: real-hardware validation pending (see Roadmap below).

### macOS Apple Silicon (Tier 3a)
- `llama-nexus-llamacpp:cpu` is published multi-arch (amd64 + arm64); the
  arm64 image was cross-built and runtime-verified on linux/arm64 QEMU
  (`llama-cli` b10723 executes, 0 unresolved libs, glibc 2.38 satisfied via
  ubuntu:24.04 runtime). Native macOS Docker expected to behave the same
  (same kernel/ABI surface) — hardware run pending.
- CPU inference on M-series is expected to be modest (4-8 tok/s for 4B Q4);
  Metal GPU acceleration is on the roadmap, not in this release.

### macOS Intel (Tier 3b)
- Runs the amd64 `cpu` image natively (no emulation penalty on Docker for
  Mac Intel). Untested on hardware.

## Known gaps / roadmap

- No physical Windows or macOS machine has run the installer end-to-end;
  CI covers the Linux paths continuously. If you hit a platform-specific
  bug, open an issue with the output of `install.sh --debug` (or the
  PowerShell equivalent) — the scripts collect diagnostics.
- Metal (macOS GPU) support for llama.cpp is deferred post-v1; voice and
  RAG tiers work on cpu today.
- `smoke_e2e.sh --full` downloads Qwen3-0.6B (~400 MB) on first run on a
  fresh stack; subsequent runs are idempotent (skips when a model is
  already deployed).

## Reproducing

```bash
# Core checks against a running stack (any tier)
scripts/smoke_e2e.sh --api-key "$API_KEY"

# Full pipeline (downloads + deploys the starter model if none is running)
scripts/smoke_e2e.sh --full --api-key "$API_KEY"

# With management auth enabled
MGMT_TOKEN="$MANAGEMENT_AUTH_TOKEN" scripts/smoke_e2e.sh --full --api-key "$API_KEY"
```
