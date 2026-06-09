# Multi-Token Prediction (MTP) Support — Development Roadmap

**Project:** llama-nexus
**Epic:** Performance Overhaul — LlamaCPP Deployments
**Status:** Draft
**Target:** llama.cpp `b9193+` (MTP merged upstream via ggml-org/llama.cpp PR #22673, May 2026)

---

## 1. Background & Motivation

MTP (Multi-Token Prediction) is now merged into upstream llama.cpp. Models trained with MTP heads (GLM-4.5/4.6, Qwen3.6 27B / 35B-A3B, Gemma 4 assistant variants, DeepSeek V3/R1 family) can self-draft multiple tokens per decode step and verify them in a single forward pass — speculative decoding **without a separate draft model**. The MTP head loads from the same GGUF and maintains its own KV cache, adding roughly one transformer layer of overhead instead of a second network's worth of VRAM.

Community-reported results on hardware comparable to ours:

| Setup | Baseline | MTP | Gain |
|---|---|---|---|
| Qwen3.6 27B Q8_0, 2× RTX 3090 layer-split | 25.7 tok/s | 55.9 tok/s | 2.17× |
| Qwen3.6 27B Q6_K, RTX 3090 + 3060 | 22.4 tok/s | 42.5 tok/s | 1.90× |
| Qwen3.6 27B Q8_0, Strix Halo | 7.4 tok/s | 18.1 tok/s | 2.44× |

Typical steady-state acceptance is ~75% at 3 draft tokens. For our single-stream coding/agent workloads on the 4× 3090 Ti box, this is the highest-leverage decode optimization available, and it composes with our existing flash-attention + KV-cache-quant configuration.

**Known trade-off:** MTP boosts single-stream latency but can *reduce* aggregate throughput under high concurrency (`-np > 1`). The deploy UI must surface this as a per-deployment decision, not a global default.

---

## 2. Scope

### In scope
- MTP capability detection in the model registry (GGUF metadata inspection)
- Deploy Manager support for MTP flags in generated `llama-server` invocations
- Deploy UI: MTP toggle + draft tunables, gated on model capability and backend build version
- Container image upgrade to an MTP-capable llama.cpp build
- Benchmark harness extension + before/after validation on our standard model set
- Documentation: README Performance Tuning section rewrite (part of the wider overhaul)

### Out of scope (this epic)
- vLLM backend MTP/speculative config (separate ticket; vLLM handles MTP natively for DeepSeek-style models)
- Training or converting MTP heads ourselves
- llama-bench integration (upstream gap — spec flags not yet supported there; tracked as ggml-org/llama.cpp #22947)

---

## 3. Architecture Overview

```
Model Registry                Deploy Manager                llamacpp-api container
┌──────────────────┐         ┌──────────────────┐          ┌──────────────────────┐
│ GGUF metadata     │         │ Config schema +   │          │ llama-server b9193+  │
│ scan on import:   │ ──────▶ │ flag generation:  │ ───────▶ │ --spec-type draft-mtp│
│ nextn_predict_    │         │ MTP block in      │          │ --spec-draft-n-max N │
│ layers > 0 ?      │         │ deployment config │          │ --spec-draft-p-min P │
└──────────────────┘         └──────────────────┘          └──────────────────────┘
        │                            ▲
        ▼                            │
   `mtp_capable: true`        Frontend Deploy page:
   in /api/models/ payload    toggle + tunables (gated)
```

Design principle: **capability flows from metadata, never from user assertion.** The UI never lets a user enable MTP on a model the registry hasn't flagged, and the backend re-validates before generating flags.

---

## 4. Phases

### Phase 0 — Foundation: Build & Image Upgrade
*Estimate: 2–3 days*

- [x] Bump the llama.cpp build in `Dockerfile` to a release ≥ `b9193` (first build with merged MTP support); pin the exact tag in the image label.
- [x] Verify CUDA build flags carry over: `-DGGML_CUDA=ON`, FA enabled across all quant types (`-DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON`). Without all-quant FA, prompt processing regresses with quantized KV cache — which we run by default.
- [ ] Smoke-test existing deployments (non-MTP models) on the new build: no regression in `/v1/chat/completions`, embeddings, VLM/mmproj paths. *(Harness: `./scripts/smoke_mtp.sh` — run after `docker compose --profile extra build llamacpp-api backend-api`)*
- [x] Add build-version reporting to `/v1/service/status` so the backend and frontend can gate MTP features on `llamacpp_build >= 9193`.

**Exit criteria:** New image deployed in CI, all existing integration tests green, build version visible via management API.

---

### Phase 1 — Model Registry: MTP Capability Detection
*Estimate: 2–3 days*

- [x] Extend the GGUF metadata reader in the registry to parse `{arch}.nextn_predict_layers`. `> 0` → model has MTP heads.
- [x] Persist `mtp_capable: bool` and `mtp_nextn_layers: int` in model metadata; expose in `/api/models/` responses.
- [x] Backfill: one-shot migration script to re-scan existing downloaded models (multi-part GGUFs: read header from part 1 only).
- [x] HuggingFace browser: surface MTP capability in search/download results where metadata is available; tag known MTP-converted repos. Important caveat to encode in docs/UI copy: **standard GGUFs of MTP-trained models usually lack the heads** — only MTP-converted GGUFs work. Capability must come from the file, not the model family name.
- [x] Unit tests: MTP GGUF fixture, non-MTP fixture, multi-part fixture, corrupt-header fixture.

**Exit criteria:** Registry correctly flags an MTP-converted Qwen3.6 GGUF and does not flag the standard GGUF of the same model.

---

### Phase 2 — Deploy Manager: Config Schema & Flag Generation
*Estimate: 3–4 days*

- [x] Add an `mtp` block to the deployment config schema (persisted via `/v1/service/config`):

```yaml
mtp:
  enabled: false            # default off — opt-in per deployment
  draft_n_max: 3            # --spec-draft-n-max  (sweet spot 2–4)
  draft_n_min: 0            # --spec-draft-n-min
  draft_p_min: 0.75         # --spec-draft-p-min  (draft confidence threshold)
```

- [x] Flag generation in the Docker orchestration layer: when `mtp.enabled`, emit `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-n-min M --spec-draft-p-min P`.
- [x] Mirror as env vars for compose-driven deployments, consistent with existing convention:
  `MTP_ENABLED`, `MTP_DRAFT_N_MAX`, `MTP_DRAFT_N_MIN`, `MTP_DRAFT_P_MIN`.
- [x] Server-side validation guardrails (reject with actionable error message):
  - `mtp.enabled=true` on a model where `mtp_capable=false`
  - `mtp.enabled=true` when container build `< b9193`
  - warn (non-blocking) when `mtp.enabled=true` and parallel slots `-np > 1` — concurrency throughput trade-off
- [x] VRAM accounting: MTP head allocates its own context/KV cache. Update the memory estimator used by the Deploy page (≈ one extra layer of weights + a parallel KV cache at the configured context size). Matters on 24 GB cards running 24B+ models near the ceiling.
- [x] Log line parsing: capture llama-server's draft acceptance stats from logs and forward via the existing WebSocket log stream with a structured `mtp_stats` event type.

**Exit criteria:** A deployment configured with MTP starts cleanly, the generated command line is correct, and invalid combinations are rejected with clear errors.

---

### Phase 3 — Frontend: Deploy UI & Monitoring
*Estimate: 3–4 days*

- [x] **Deploy page:** "Speculative Decoding (MTP)" section, rendered only when the selected model reports `mtp_capable=true` *and* the backend build supports it. Disabled state shows the reason ("Model lacks MTP heads" / "Backend build too old — upgrade required").
- [x] Controls: enable toggle, `draft_n_max` slider (1–6, default 3), advanced collapsible for `draft_n_min` / `draft_p_min` with inline help text explaining the acceptance-rate trade-off.
- [x] Inline warning when MTP is enabled with parallel slots > 1.
- [x] **Model registry views:** "MTP" capability badge on model cards and the HF browser.
- [x] **Monitoring:** acceptance rate + drafted-vs-accepted token counters on the deployment dashboard, fed by the `mtp_stats` log events from Phase 2. Acceptance rate is the single number that tells you whether your tunables are right (target ≥ 70%).
- [x] Persist last-used MTP settings per model in the per-model sampling config store.

**Exit criteria:** Full deploy → monitor loop usable from the UI without touching compose files.

---

### Phase 4 — Benchmarking & Validation
*Estimate: 3–5 days, overlaps Phase 3*

Since upstream `llama-bench` doesn't support spec flags yet, benchmarking goes through `llama-server` — which fits our existing harness anyway.

- [x] Extend the benchmark scripts in `scripts/` (same harness used for the TurboQuant KV-cache runs) with an MTP sweep mode: `draft_n_max ∈ {2,3,4,5}` × `draft_p_min ∈ {0.5, 0.75, 0.9}` × quant levels.
- [x] Standard test matrix on the 4× 3090 Ti box:
  - Qwen3.6 27B (dense, MTP GGUF) — Q4_K_M, Q6_K, Q8_0
  - Qwen3.6 35B-A3B (MoE, MTP GGUF)
  - GLM-4.6 if a converted GGUF fits in VRAM budget
  - Each at `-np 1` and `-np 4` to quantify the concurrency penalty
- [x] Measure: decode tok/s, prompt-processing tok/s, acceptance rate, TTFT, VRAM delta vs. non-MTP baseline, and interaction with `--cache-type-k/v q8_0` (our default KV quant).
- [x] Output-quality spot check: MTP verification should be lossless (accepted tokens match the main model's distribution), but validate with a fixed-seed diff suite on coding prompts to catch implementation bugs early in this still-fresh upstream feature.
- [x] Publish results to `results/` and derive recommended per-model-family defaults that the Deploy UI pre-fills.

**Exit criteria:** Documented speedup table for our hardware, defaults tuned to ≥ 70% acceptance, no quality regressions on the diff suite.

---

### Phase 5 — Documentation & Rollout
*Estimate: 1–2 days*

- [ ] Rewrite the README **Performance Tuning** section as part of the wider overhaul, organized as a decision tree:
  1. Flash attention (always on)
  2. KV cache quantization (q8_0 default, TurboQuant notes)
  3. **Speculative decoding — MTP** (single-stream, MTP-capable models) vs. classic draft-model speculation (everything else)
  4. Batch/ubatch tuning for throughput workloads
  5. Multi-GPU layer split guidance
- [ ] Add an MTP how-to in `docs/`: which models work, where to get converted GGUFs, what acceptance rate means, troubleshooting table (e.g., "MTP toggle disabled" → causes).
- [ ] Changelog entry + version bump; mark MTP as **beta** for the first release given upstream is weeks old.
- [ ] Rollout: enable on the AI VTuber inference deployment first (single-stream, latency-sensitive — ideal MTP workload), monitor acceptance for a week, then recommend as default for compatible single-slot deployments.

---

## 5. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Upstream flag churn (`--spec-type mtp` vs `draft-mtp` aliasing has already shifted during the PR's life) | Medium | Pin exact llama.cpp build; flag generation behind a version-keyed adapter so a rename is a one-line change |
| Concurrency throughput regression silently enabled | Medium | Hard warning in UI + config validation; document `-np 1` recommendation |
| VRAM overrun from MTP KV cache on near-full cards | Medium | Memory estimator update (Phase 2); fail fast with clear OOM guidance |
| Users enable MTP on non-MTP GGUFs of MTP model families | High | Capability from file metadata only; UI gating + server-side rejection |
| Fresh upstream code → correctness bugs | Low–Med | Fixed-seed diff suite in Phase 4; beta label at launch |

---

## 6. Acceptance Criteria (Epic-level)

1. An MTP-converted GGUF imported through the registry is auto-flagged, deployable with MTP from the UI in under a minute, and shows live acceptance metrics.
2. Measured ≥ 1.7× single-stream decode speedup on Qwen3.6 27B Q6_K on our 3090 Ti hardware vs. the non-MTP baseline on the same build.
3. Zero regressions for non-MTP deployments on the upgraded image.
4. Performance Tuning docs rewritten with MTP integrated into the broader optimization decision tree.

**Total estimate:** ~3 weeks of focused work, Phases 3/4 parallelizable.