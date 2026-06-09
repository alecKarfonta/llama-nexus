# UI Integration Plan — fast-inference → llama-nexus

Integrate experimental results from `fast-inference/` into the llama-nexus Deploy UI, Chat, and backend for llama.cpp deployments.

**Status:** Draft  
**Last updated:** 2026-06-08  
**Related:** `grammar-mtp.md`, `docs/MTP_roadmap.md`, `bench/run_grammar_mtp.sh`

---

## 1. Current state

### Already shipped (MTP epic Phases 0–4)

| Area | What exists |
|------|-------------|
| Registry | `mtp_capable`, `mtp_nextn_layers` from GGUF metadata |
| Backend | `mtp` config block → `--spec-type draft-mtp` flags in `LlamaCPPManager` |
| Deploy UI | `MtpConfigSection` — toggle, n_max slider (1–6), advanced p_min/n_min |
| Monitoring | `useMtpStats` + `MtpStatsPanel` — draft acceptance from server logs |
| Defaults | `mtpRecommendedDefaults.json` + per-model localStorage via `resolveMtpForModel` |
| Validation | Reject MTP on non-MTP GGUF; warn on `parallel_slots > 1` |
| Benchmarks | `scripts/mtp_bench.py`, `fast-inference/bench/run_matrix.sh` |

### Gaps vs experiment learnings

| Experiment finding | Not yet in product |
|--------------------|-------------------|
| MTP **n2** best for free chat (~43 tok/s) | Defaults still n3 for qwen3.6 |
| MTP **n8** best on tool path (~87 tok/s) | No agent/tools preset |
| Grammar-only tools ~28 tok/s; MTP ~3× | Tools path not configured from Deploy |
| 2-GPU faster than 3-GPU for decode | No GPU speed preset |
| `enable_thinking: false` for tool latency | No `chat-template-kwargs` in manager |
| `cache-reuse` + `q8_0` KV part of recipe | Partially exposed, not preset-driven |
| llguidance / jump-forward | Image built without `LLAMA_LLGUIDANCE=ON` |
| `ConstraintEditor` | Built in frontend, not wired to ChatPage |
| Proxy grammar / `response_format` | Not audited end-to-end |

---

## 2. Design inputs (benchmark summary)

### Free chat (Q4_K_M MTP, 2 GPU, ctx 131k)

| Config | code tok/s | Notes |
|--------|------------|-------|
| MTP off | ~25 | baseline |
| MTP n2 | **~49** | best for chat |
| MTP n4+ | ~38–45 | worse than n2 on chat |

### Tool-call path (`tools` API → lazy grammar, same hardware)

| Config | weather tok/s | MTP acceptance |
|--------|---------------|----------------|
| Grammar only (MTP off) | ~28 | — |
| MTP n2 | ~54 | 100% |
| MTP n8 | **~87** | 87.5% |

**Takeaway:** Optimal `draft_n_max` is **workload-dependent** — opposite direction on chat vs tools.

### GPU / context (Q6_K 27B, 2×16GB)

- **48k ctx:** trivial fit
- **128k ctx:** comfortable
- **~250k ctx:** ceiling with q8_0 KV
- **2 GPU > 3 GPU** for decode speed (PCIe sync cost)

---

## 3. Target architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ Deploy UI       │     │ Backend            │     │ llama-server        │
│ workload profile│────▶│ LlamaCPPManager    │────▶│ --spec-type draft-mtp│
│ MTP n2 / n8     │     │ chat_template_kwargs│     │ --chat-template-kwargs│
│ GPU presets     │     │ cache_reuse, KV    │     │ lazy tool grammar   │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
         │                                                  ▲
         │                                                  │
┌─────────────────┐     ┌──────────────────┐                  │
│ Chat UI         │────▶│ service proxy    │─────────────────┘
│ tools toggle    │     │ passthrough      │   tools / response_format
│ ConstraintEditor│     │ grammar/json_schema
└─────────────────┘
```

**Principle:** Capability flows from metadata and build version (existing MTP pattern). Workload profiles encode experiment-backed defaults; Custom unlocks full control.

---

## 4. Phases

### Phase 1 — Workload profiles (~2–3 days)

**Goal:** One Deploy control applies the right MTP + server knobs for how the deployment is used.

#### Config schema (additive)

```yaml
mtp:
  enabled: true
  workload_profile: agent   # chat | agent | throughput | custom
  draft_n_max: 8            # set by profile unless custom
  draft_n_min: 0
  draft_p_min: 0.75
```

#### Profile definitions

| Profile | `draft_n_max` | `parallel_slots` | `chat_template_kwargs` | Use case |
|---------|---------------|------------------|------------------------|----------|
| **chat** | 2 | 1 | `{"preserve_thinking": true}` | Interactive chat, reasoning |
| **agent** | 8 | 1 | `{"enable_thinking": false}` | Tool calls, overlay commands |
| **throughput** | off (or 2) | 4+ | default | Batch; show concurrency warning |
| **custom** | user | user | user | Advanced |

#### Frontend

- Extend `MtpConfigSection` with **Workload** `ToggleButtonGroup`.
- Profile change updates MTP + related server fields; Custom enables sliders.
- Inline bench hints (e.g. “Agent + MTP: ~3× grammar-only on tool path”).

#### Backend

- `normalize_mtp_config()` / `mtp_deploy.py`: map profile → fields.
- Validation warnings for agent + `parallel_slots > 1`.

#### Data

- Update `frontend/src/config/mtpRecommendedDefaults.json`:
  - `qwen3.6` chat: `draft_n_max: 2`
  - Add agent override: `draft_n_max: 8` when `workload_profile: agent`
- Optional: extend `scripts/mtp_report.py --write-ui-defaults` to ingest `tools-on-*.json`.

**Exit criteria:** User selects Agent profile, deploys, gets n8 + thinking disabled without manual flag editing.

---

### Phase 2 — Agent & tools server config (~3–4 days)

**Goal:** Deploy configures the server for the tool-call path.

#### Backend (`llamacpp_manager.py`)

- Add `server.chat_template_kwargs` → `--chat-template-kwargs '{"..."}'`.
- Default `cache_reuse: 1024` for agent profile.
- Expose in field metadata API (`llamacpp_flag` pattern).

#### Deploy UI — new **Agent & Tools** subsection

| Field | Purpose |
|-------|---------|
| `chat_template_kwargs` | JSON editor, validated |
| `reasoning_budget` | Thinking token cap; “disable for tool latency” helper |
| `cache_reuse` | Prefix cache for system-prompt-heavy agents |
| Info panel | Tool grammar is automatic when clients send `tools` (lazy GBNF) |

#### Chat UI

- When `enableTools` on: chip “Tools mode” + tooltip linking to Deploy Agent profile.
- Optional: suggest Deploy profile switch when enabling tools.

#### Proxy audit (`backend/routes/service.py`)

- Verify passthrough: `tools`, `tool_choice`, `response_format`, `grammar` reach llama-server unchanged.

**Exit criteria:** Deploy Agent profile produces correct command line; Chat tool calls hit optimized server config.

---

### Phase 3 — Structured output & grammar (~1 week)

**Goal:** Grammar-constrained decoding from Chat; path to llguidance.

#### Chat

- Wire existing `ConstraintEditor` into `ChatPage`:
  - Toggle → `response_format: { type: "json_schema", json_schema: { schema, strict: true } }`.
  - Advanced: raw GBNF `grammar` field.
- Warn when both tools and custom grammar enabled (single grammar slot in llama.cpp today).

#### Build (`Dockerfile`)

```cmake
-DLLAMA_LLGUIDANCE=ON   # requires Rust in builder stage
```

- Report `llguidance_enabled` in `/v1/service/status`.
- Deploy capability chip: GBNF (current) vs llguidance (after rebuild).

**Exit criteria:** User constrains Chat output to JSON schema; requests use llama-server grammar path. llguidance build optional follow-up for jump-forward tier.

---

### Phase 4 — GPU & context presets (~3–5 days)

**Goal:** Encode 2-GPU speed vs max-context without manual probing.

#### Deploy — Execution tab presets

| Preset | `cuda_devices` | `split_mode` | Notes |
|--------|----------------|--------------|-------|
| **Speed (2 GPU)** | `0,1` | `layer` | Experiment winner for decode |
| **Max context (2 GPU)** | `0,1` | `layer` | ctx up to ~250k Q6_K + q8_0 |
| **3 GPU** | `0,1,3` | `layer` | Warning: ~10% slower decode |

#### Context estimator

- Update VRAM estimator for MTP head overhead.
- Show fits / tight / OOM from Q6_K probe data (48k trivial, 128k safe, 250k edge).

#### Models UX

- Q6_K 27B on 16GB cards: suggest 2-GPU + ctx 48k–128k.

**Exit criteria:** Preset applies cuda_devices + context with clear speed vs context tradeoff copy.

---

### Phase 5 — Monitoring & in-product benchmarks (~1 week)

**Goal:** Close deploy → measure loop without shell scripts.

#### Monitoring

- `MtpStatsPanel`: show active workload profile + acceptance rate.
- Deploy dashboard badge: “Tool-path optimized” when agent profile active.

#### Benchmarks (optional Deploy action)

- `POST /api/v1/benchmark/mtp` — runs dockerized `run_matrix.sh` / `run_grammar_mtp.sh`.
- Results from `fast-inference/bench/results/` displayed in UI table.

**Exit criteria:** User triggers tool-call bench from Deploy; sees n2 vs n8 comparison.

---

### Phase 6 — Models catalog (~2–3 days)

**Goal:** Users download the right GGUF without reading README.

- Models / HF browser: **Standard vs MTP** filter; download CTA for `*-MTP-GGUF`.
- Deploy picker: banner when Qwen3.6 selected but `mtp_capable=false`.
- Recommended quant copy: Q4_K_M MTP for 2×16GB 27B agent path.

**Exit criteria:** New user can find and deploy MTP GGUF from UI alone.

---

## 5. Deploy UI information architecture (llama.cpp)

```
Deploy (llama.cpp)
├── Model           — MTP badge, download MTP variant CTA
├── Performance     — KV q8_0, cache_reuse, parallel_slots, flash-attn
├── Speculative     — Workload profile + MtpConfigSection
├── Agent & Tools   — chat_template_kwargs, thinking, tools path docs  [NEW]
├── GPU             — cuda_devices presets, split_mode, ctx estimator
├── Server          — jinja, metrics (existing)
└── Monitoring      — MTP acceptance + profile-aware hints
```

### Chat (complementary)

- Tools toggle + structured output (`ConstraintEditor`).
- Footer: MTP acceptance when connected to managed deployment.
- Hint when Deploy profile mismatches Chat mode (tools on + Chat profile).

---

## 6. Full config schema sketch

```yaml
mtp:
  enabled: true
  workload_profile: agent   # chat | agent | throughput | custom
  draft_n_max: 8
  draft_n_min: 0
  draft_p_min: 0.75

server:
  chat_template_kwargs: '{"enable_thinking": false}'
  cache_reuse: 1024
  jinja: true
  reasoning_budget: 0

performance:
  cache_type_k: q8_0
  cache_type_v: q8_0
  parallel_slots: 1
  split_mode: layer

execution:
  cuda_devices: "0,1"
```

---

## 7. Implementation order

```
Phase 1 Workload profiles
    ↓
Phase 2 Agent server config ──→ Phase 3 Chat grammar/tools
    ↓
Phase 4 GPU presets
    ↓
Phase 5 llguidance build + benchmarks
    ↓
Phase 6 Models catalog polish
```

| Priority | Phase | Rebuild required? |
|----------|-------|-------------------|
| 1 | Workload profiles | No |
| 2 | Agent server config | No |
| 3 | Chat structured output | No |
| 4 | GPU presets | No |
| 5 | llguidance + benchmarks | Yes (Dockerfile) |
| 6 | Models catalog | No |

---

## 8. Out of scope (v1)

- vLLM grammar/MTP parity (vLLM Deploy has separate tools/speculative sections).
- Per-request dynamic MTP n_max (profile-level only for v1).
- Grammar + tools composition (upstream pre-trigger grammar) until llama.cpp version supports it.
- Automatic Deploy ↔ Chat profile sync (hints only).

---

## 9. Acceptance criteria (epic)

1. User selects **Agent** workload on Deploy for Qwen3.6 MTP GGUF → server starts with n8, thinking off, cache_reuse on.
2. Chat with tools enabled achieves tool-path performance class (~50–90 tok/s on 2×16GB) without manual compose edits.
3. Chat structured output via `ConstraintEditor` produces schema-valid JSON through llama-server.
4. GPU preset “Speed (2 GPU)” matches experiment-winning binding.
5. MTP acceptance visible in Deploy monitoring; ≥70% on agent profile in steady state.
6. Documentation links from Deploy help to `fast-inference/plans/grammar-mtp.md` and updated `mtpRecommendedDefaults.json`.

---

## 10. References

- `fast-inference/README.md` — stack recipe
- `fast-inference/plans/grammar-mtp.md` — tool-call experiment results
- `fast-inference/bench/run_grammar_mtp.sh` — reproducible grammar+MTP matrix
- `fast-inference/bench/run_extras.sh` — KV, GPU, concurrency extras
- `docs/MTP_roadmap.md` — Phases 0–4 completion status
- `frontend/src/components/deploy/MtpConfigSection.tsx`
- `backend/modules/managers/llamacpp_manager.py`
- `backend/modules/mtp_deploy.py`
