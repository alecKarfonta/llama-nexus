# MTP Benchmarking (Phase 4)

Upstream `llama-bench` does not expose MTP / speculative flags yet. This repo benchmarks through **llama-server** and the OpenAI-compatible HTTP API — the same path the Deploy page uses in production.

## Prerequisites

- llama.cpp **b9193+** (`llama-server` with `--spec-type draft-mtp`)
- MTP-converted GGUF (`mtp_capable=true` in the model registry)
- Optional: `pip install pyyaml` for matrix runs
- NVIDIA GPU(s) with enough VRAM for the model + MTP head overhead

Set the server binary if it is not on `PATH`:

```bash
export LLAMA_SERVER=/path/to/llama-server
export MODELS_DIR=./models
```

## Quick run (single model)

Baseline + one MTP configuration:

```bash
chmod +x scripts/mtp_bench.sh
./scripts/mtp_bench.sh models/Qwen3.6-27B-Q6_K.gguf
```

Full sweep (`draft_n_max` × `draft_p_min` × `parallel_slots`):

```bash
./scripts/mtp_bench.sh models/Qwen3.6-27B-Q6_K.gguf --sweep --parallel 1,4
```

## Standard matrix (4× 3090 Ti)

Edit `scripts/mtp_benchmark_matrix.yaml` for local GGUF paths, then:

```bash
./scripts/mtp_bench.sh --matrix
# or
python3 scripts/mtp_bench.py --matrix scripts/mtp_benchmark_matrix.yaml --dry-run
```

The matrix includes:

- Qwen3.6 27B dense — Q4_K_M, Q6_K, Q8_0
- Qwen3.6 35B-A3B MoE
- Optional GLM-4.6 (commented if VRAM is tight)
- `parallel_slots` 1 and 4
- MTP grid: `draft_n_max ∈ {2,3,4,5}`, `draft_p_min ∈ {0.5,0.75,0.9}`
- KV cache `q8_0` (production default)

## Metrics captured

| Metric | Source |
|--------|--------|
| Decode tok/s | HTTP completion, generation phase |
| Prompt tok/s | HTTP completion, prefill → first token |
| TTFT | First streamed token timestamp |
| Acceptance rate | llama-server stderr (`draft acceptance rate = …`) |
| VRAM delta | `nvidia-smi` after server start vs before |

Results are written to `results/mtp_bench_<timestamp>.jsonl`.

## Quality diff (fixed seed)

Verifies MTP-off vs MTP-on outputs match at `temperature=0` (lossless verification path):

```bash
python3 scripts/mtp_quality_diff.py --model models/Qwen3.6-27B-MTP-Q6_K.gguf
```

Exit code `0` = all coding prompts matched; `1` = divergence (investigate upstream or tunables).

## Report & UI defaults

```bash
python3 scripts/mtp_report.py --bench 'results/mtp_bench_*.jsonl' --write-ui-defaults
```

Produces:

- `docs/mtp_benchmark_results.md` — speedup table and recommendations
- `results/mtp_recommended_defaults.json`
- `frontend/src/config/mtpRecommendedDefaults.json` — prefill for Deploy UI (≥ 70% acceptance picks)

## Scripts

| Script | Role |
|--------|------|
| `scripts/mtp_bench_lib.py` | Server runner, HTTP metrics, matrix expansion |
| `scripts/mtp_bench.py` | Main sweep driver |
| `scripts/mtp_bench.sh` | Shell wrapper |
| `scripts/mtp_quality_diff.py` | Fixed-seed output diff suite |
| `scripts/mtp_report.py` | Markdown report + recommended defaults |
| `scripts/mtp_benchmark_matrix.yaml` | Standard hardware matrix |
