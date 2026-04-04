#!/bin/bash
# ============================================================================
# TurboQuant Speed Benchmark (wraps llama-bench)
# ============================================================================
# Thin wrapper around llama-bench that runs the TurboQuant comparison matrix.
# Uses llama-bench directly — no custom classes needed for raw throughput.
#
# Usage:
#   ./scripts/turboquant_bench.sh <model_path> [extra llama-bench args...]
#
# Example:
#   ./scripts/turboquant_bench.sh models/Llama-3.1-8B-Q4_K_M.gguf
#   ./scripts/turboquant_bench.sh models/model.gguf -p 512,4096 -r 3
# ============================================================================

set -euo pipefail

MODEL="${1:?Usage: $0 <model_path> [extra args...]}"
shift || true

# Auto-find llama-bench
BENCH=""
for candidate in \
    "./llama.cpp/build/bin/llama-bench" \
    "./build/bin/llama-bench" \
    "/usr/local/bin/llama-bench"; do
    [[ -x "$candidate" ]] && BENCH="$candidate" && break
done
[[ -z "$BENCH" ]] && { echo "❌ llama-bench not found"; exit 1; }
[[ ! -f "$MODEL" ]] && { echo "❌ Model not found: $MODEL"; exit 1; }

mkdir -p results
TS=$(date +%Y%m%d_%H%M%S)
NAME=$(basename "$MODEL" .gguf)
OUTFILE="results/turboquant_speed_${NAME}_${TS}.csv"
SYSINFO="results/system_info_${TS}.txt"

# Record system info
{
    echo "Date: $(date -Iseconds)"
    echo "Host: $(hostname)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
    [[ -d "llama.cpp/.git" ]] && echo "Commit: $(cd llama.cpp && git rev-parse --short HEAD)" || true
} > "$SYSINFO"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          TurboQuant Speed Benchmark (llama-bench)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Model: $MODEL"
echo "Output: $OUTFILE"
echo ""

# Core benchmark: all KV types × context depths × flash attn modes
# llama-bench natively handles the combinatorial expansion
"$BENCH" \
    -m "$MODEL" \
    -ngl 99 \
    -r 5 \
    -ctk f16,q8_0,q4_0,tbq4_0,tbq3_0 \
    -ctv f16,q8_0,q4_0,tbq4_0,tbq3_0 \
    -fa 0,1 \
    -p 512,2048,4096,8192,16384,32768 \
    -n 128 \
    -o csv \
    "$@" \
    > "$OUTFILE" 2>&1

echo ""
echo "✅ Done! Results: $OUTFILE ($(wc -l < "$OUTFILE") lines)"
echo "   System info: $SYSINFO"
echo ""
echo "Next: python3 scripts/turboquant_report.py --speed $OUTFILE"
