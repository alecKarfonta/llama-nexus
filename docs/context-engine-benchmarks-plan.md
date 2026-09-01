# Context-Engine Benchmarking for Llama Nexus — Development Plan

## 0. What we're actually measuring (and why it's the interesting part)

The premise: stop benchmarking the *model* and start benchmarking the *system* = **model + context engine**. This is the same shift that made SWE-bench-with-a-harness (Cursor, Claude Code) more informative than raw model scores — the scaffolding is part of the intelligence.

The headline number is task accuracy of the full system. But the scientifically useful output is not a single leaderboard cell — it's the **paired delta**: for a *fixed model and fixed item set*, how does swapping the context engine change the outcome? That framing drives every design decision below (paired evals, per-item logging, controls).

Three subtleties this plan is explicitly built around:

1. **Parametric leakage.** On MMLU/GPQA-style sets the model often already knows the answer, so retrieval either does nothing or *injects noise and hurts*. A benchmark that doesn't stress external knowledge can't show a context engine's value. We select for retrieval-sensitivity and keep a reasoning-only control to prove no regression.
2. **Oracle leakage.** If the corpus contains the gold passage (or the eval's source docs), you're measuring retrieval-of-the-answer, not capability. We treat corpus composition as a first-class experimental variable: closed-book → realistic corpus → oracle corpus → adversarial (distractor-injected).
3. **Unfair budgets.** A context engine that stuffs 40k tokens will beat one that injects 2k — but that's a cost story, not an intelligence story. We report accuracy *and* accuracy-per-context-token (a Pareto frontier), and support holding the token budget constant across engines.

## 1. Core abstraction: the `ContextEngine`

Everything hangs off one protocol. A context engine takes the task input (and optionally conversation state) and returns a bundle of context to inject, plus provenance for scoring retrieval quality.

```python
# backend/benchmarks/engines/base.py
from typing import Protocol, Sequence
from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    doc_id: str
    text: str
    score: float
    source: str          # which index/store it came from
    meta: dict

@dataclass
class ContextBundle:
    chunks: Sequence[RetrievedChunk]
    injected_prompt: str        # the fully-rendered context block
    token_count: int
    retrieval_ms: float
    trace: dict                 # queries issued, hops, rerank order — for analysis

class ContextEngine(Protocol):
    name: str
    def prepare(self, corpus_id: str | None) -> None: ...      # build/attach indexes
    def retrieve(self, query: str, k: int, budget_tokens: int,
                 history: list[dict] | None = None) -> ContextBundle: ...
```

### The engine ladder (the independent variable)

Implement as adapters so a run config just names an engine + params. Order is roughly increasing sophistication:

| Tier | Engine | Notes / reuses |
|---|---|---|
| 0 | `NoOpEngine` | Closed-book baseline. Every delta is measured against this. |
| 1 | `DenseEngine` | Qdrant top-k over the embedding server. Your existing `:8602` + `:6333`. |
| 2 | `HybridEngine` | Dense + BM25/sparse fusion (RRF). |
| 3 | `RerankEngine` | Tier 1/2 candidates + cross-encoder reranker, then budget-truncate. |
| 4 | `GraphRAGEngine` | Thin wrapper over your existing `/api/graphrag/search` + `code-search`. This is the one that's genuinely differentiated vs. off-the-shelf RAG. |
| 5 | `AgenticEngine` | Model-in-the-loop: query decomposition, multi-hop, "retrieve→read→retrieve". This is the true analog of the Cursor/Claude-Code tier — the system decides *what* and *when* to fetch. |
| 6 | `LongContextStuffEngine` | No retrieval; place whole (small) corpus in context. Upper-bound oracle for "if retrieval were perfect." |
| — | `WebSearchEngine` | Optional: live web as the store, for open-domain factuality sets. |

Keep each adapter dumb — all fairness logic (budget enforcement, prompt templating) lives in the runner, so engines differ *only* in how they select context.

## 2. Where it lives in the repo (reuse map)

You are not building infrastructure; you're composing what's running. New code is a `benchmarks/` package plus a results view.

```
backend/
  benchmarks/
    __init__.py
    engines/            # ContextEngine adapters (§1)
    datasets/           # loaders + normalizers -> canonical Task schema
    graders/            # task graders + retrieval graders + LLM-judge
    corpus/             # ingest/dedup/contamination tooling
    runner.py           # orchestrates model x engine x dataset -> RunResult
    stats.py            # paired tests, bootstrap CIs
    cli.py              # `python -m benchmarks run --config ...`
    api.py              # FastAPI router: launch runs, stream progress, serve results
results/
  runs/<run_id>/...     # per-item JSONL + aggregate.json (already have results/)
frontend/
  src/pages/Benchmarks/ # leaderboard + Pareto + per-item drill-down
```

Reuse specifically:
- **Serving**: hit the OpenAI-compatible endpoint (`:8600/v1` or vLLM `:8601`) exactly like a client. The model-under-test is a URL + model name, so you can benchmark any deployed config from the Deploy page.
- **Embeddings + Qdrant**: `DenseEngine`/`HybridEngine` call `:8602` and `:6333` directly.
- **GraphRAG**: `GraphRAGEngine` proxies your existing endpoints — no new graph code.
- **Workflow Engine**: your "Multi-Model Comparison" and "RAG Pipeline" templates are conceptual prototypes for the runner; the harness is the batch/scored/statistical version of the same composition. Consider emitting a workflow graph per run for the existing visualizer.
- **Judge model**: LLM-as-judge should call a *separate* deployed model (ideally a stronger one) to avoid self-grading bias.

## 3. Benchmark suite (the dependent variable)

Pick sets that actually move under retrieval, plus controls. Start narrow (2–3 real + 2 controls), expand later.

**Retrieval-sensitive (the point of the exercise):**
- **Multi-hop QA** — HotpotQA, 2WikiMultiHopQA, or MuSiQue. These reward hybrid/graph/agentic engines and punish naive top-k. MuSiQue is the hardest and most discriminating.
- **Comprehensive/realistic RAG** — CRAG or FRAMES. FRAMES is especially good: it couples factuality, multi-hop retrieval, *and* reasoning in one, which is exactly your "smartest system" target.
- **Recent/private knowledge** — a small custom set over docs the model provably can't have memorized (your homelab docs, recent papers, or a dated news slice). This is where the closed-book delta is cleanest and least contaminated.

**Controls (prove the engine helps where it should and doesn't hurt where it shouldn't):**
- **Reasoning-only** — GPQA-Diamond or MMLU-Pro subset. Expectation: retrieval ≈ flat or slightly negative. If a "better" engine tanks this, it's injecting noise — a critical failure signal you'd otherwise miss.
- **Long-context stress** — RULER or a needle-in-haystack variant, to separate "good retrieval" from "model just has a big context window."

Each dataset loader normalizes to one canonical schema so graders/runner are dataset-agnostic:

```python
@dataclass
class Task:
    id: str
    question: str
    gold_answers: list[str]
    gold_doc_ids: list[str] | None   # enables retrieval recall@k / nDCG when present
    metadata: dict                   # hop count, domain, date, contamination_tag
```

## 4. Corpus & contamination handling

Corpus composition is an experimental knob, not a fixed asset. Support building the same eval against:
- **Realistic**: a large general corpus (e.g., a Wikipedia dump for the QA sets) where gold passages exist but must be *found* among millions.
- **Oracle**: gold passages + a controlled distractor ratio. Sweeping distractor ratio (0 → 50 → 200 distractors) produces the most informative curves — it directly measures each engine's noise-robustness.
- **Closed-book**: no corpus (pairs with `NoOpEngine`).

Contamination controls (bake these in from day one — they're the difference between a credible result and a meaningless one):
- Dedup/near-dup (MinHash/embedding-sim) between corpus and eval items; log overlap.
- A **closed-book delta** column everywhere: `acc(engine) − acc(NoOp)`. If NoOp already scores high, retrieval has little headroom and you say so rather than claiming a win.
- Canary items with known-unmemorizable facts to confirm the corpus is actually being used.

Reuse your Document nodes / ingestion for building indexes so corpus prep is one pipeline, one `corpus_id`, shared by every dense/hybrid/graph engine.

## 5. Metrics & scoring

**Task graders** (per dataset): exact match + token-F1 for QA; a constrained **LLM-as-judge** for free-form answers (rubric + gold, judge = separate deployed model, temperature 0, log rationale). Always keep a deterministic grader alongside the judge so you can quantify judge noise.

**Retrieval graders** (only when `gold_doc_ids` present): recall@k, MRR, nDCG@k. This decomposes *system* failures into "didn't retrieve it" vs "retrieved it and still got it wrong" — the single most useful diagnostic you'll get.

**Efficiency / fairness:**
- `accuracy_per_1k_context_tokens` and a **Pareto frontier** plot (accuracy vs injected tokens). "Smartest system" without this collapses to "whoever stuffed the most."
- Wall-clock latency and, if you care, a cost proxy (tokens × rate). Agentic engines win accuracy but lose latency/cost — show the tradeoff explicitly.
- A **fixed-budget mode**: cap every engine at the same injected-token budget so comparisons isolate *selection quality* from *quantity*.

**Statistics** (`stats.py`): because runs are paired per item on a fixed model —
- **McNemar's test** for accuracy deltas between two engines (correct paired test for binary outcomes).
- **Bootstrap CIs** over items for every aggregate metric; report CIs, not bare means.
- ≥3 seeds where sampling is non-deterministic (temp > 0, agentic branching); report seed variance.
- Effect size, not just p-value — a significant +0.4% is not a story.

## 6. Results store + leaderboard UI

Persist per-item JSONL (task, engine, model, retrieved chunk ids, injected tokens, verdict, latency, judge rationale) + an `aggregate.json` per run under `results/runs/<run_id>/`. Per-item logs are what let you re-score later without re-running and do error analysis.

Frontend `Benchmarks` page (this is squarely in your wheelhouse and the payoff view):
- **Leaderboard**: rows = (model × engine), columns = datasets, cell = accuracy ± CI, with the closed-book delta surfaced.
- **Pareto scatter**: accuracy vs context tokens, one point per engine — the "is this intelligence or just more tokens?" view.
- **Per-item drill-down**: click a cell → see which items flipped between two engines, with retrieved context and judge rationale. This is where you'll actually learn things.
- Launch/monitor runs over the run API with live progress (reuse your WebSocket log-streaming pattern).

## 7. Phased roadmap

**Phase 0 — Walking skeleton (thin end-to-end slice).** Wire `Task` schema + one dataset loader (HotpotQA subset, ~200 items) + `NoOpEngine` and `DenseEngine` + EM/F1 grader + `runner.py` + JSONL results. No UI, no stats. Goal: one command produces a real closed-book-vs-dense delta. Everything after is depth.

**Phase 1 — Real comparison.** Add `HybridEngine`, `RerankEngine`, and the `GraphRAGEngine` wrapper. Add McNemar + bootstrap CIs. Add the reasoning-only control (GPQA-Diamond) to catch regressions. This is the first genuinely publishable internal result.

**Phase 2 — Fairness & corpus science.** Fixed-budget mode, Pareto metric, oracle/realistic/adversarial corpus builder with distractor sweep, contamination dedup + closed-book delta everywhere. Add retrieval graders (recall@k/nDCG).

**Phase 3 — The agentic tier + UI.** `AgenticEngine` (multi-hop retrieve→read→retrieve) and `LongContextStuffEngine` upper bound. Build the Benchmarks leaderboard + Pareto + drill-down. Add FRAMES/CRAG.

**Phase 4 — Scale & polish.** `WebSearchEngine`, long-context control (RULER), multi-seed automation, run API + live progress, export/share. Optional: register benchmark runs as first-class Workflow templates so any deployed model is one click from evaluation.

## 8. Risks / open questions to resolve early

- **Judge reliability** dominates free-form results. Prefer datasets with checkable answers for headline claims; treat judged sets as secondary until you've measured judge–human agreement on a sample.
- **GraphRAG ingestion cost.** Graph construction is expensive per corpus; cache aggressively per `corpus_id` and treat index-build time as a reported cost of the `GraphRAGEngine`, separate from query-time latency.
- **Agentic engine variance & cost** can swamp signal — pin seeds, cap hops, and always report the latency/token tradeoff beside the accuracy win.
- **Contamination is the credibility killer.** If you publish any of this (and given Stockastic-adjacent visibility, you might), the closed-book delta + dedup logs are what make it defensible.
- **Scope creep in datasets.** Resist adding sets until the harness is solid on three. Breadth of engines > breadth of benchmarks for answering *your* question.

---

### The one-line thesis
Build a `ContextEngine` protocol, run a **paired** eval of `{model} × {engine} × {corpus}` on retrieval-sensitive sets with a reasoning-only control, and report **accuracy, the closed-book delta, and accuracy-per-context-token with CIs** — so "smartest possible system" means *provably better selection*, not just more tokens.
