#!/bin/bash
# ============================================================================
# MTP Speed Benchmark (llama-server HTTP harness)
# ============================================================================
# llama-bench does not support --spec-type draft-mtp yet; this script uses
# scripts/mtp_bench.py which drives llama-server + OpenAI-compatible API.
#
# Usage:
#   ./scripts/mtp_bench.sh <model.gguf>              # baseline + single MTP
#   ./scripts/mtp_bench.sh <model.gguf> --sweep      # full grid
#   ./scripts/mtp_bench.sh --matrix                  # standard 3090 Ti matrix
#
# Environment:
#   LLAMA_SERVER  — path to llama-server (b9193+)
#   MODELS_DIR    — model directory for matrix paths (default: ./models)
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODELS_DIR="${MODELS_DIR:-$ROOT/models}"

if [[ "${1:-}" == "--matrix" ]]; then
  exec python3 scripts/mtp_bench.py --matrix scripts/mtp_benchmark_matrix.yaml "${@:2}"
fi

MODEL="${1:?Usage: $0 <model.gguf> [--sweep] | $0 --matrix}"
shift || true

EXTRA=()
if [[ "${1:-}" == "--sweep" ]]; then
  EXTRA+=(--sweep --baseline)
  shift
elif [[ "${1:-}" == "--mtp-only" ]]; then
  EXTRA+=(--mtp --no-baseline)
  shift
else
  EXTRA+=(--baseline --mtp)
fi

exec python3 scripts/mtp_bench.py --model "$MODEL" "${EXTRA[@]}" "$@"
