#!/bin/bash
# ============================================================================
# TurboQuant Quality Benchmark (wraps llama-perplexity)
# ============================================================================
# Two-step quality evaluation:
#   1. Generate FP16 baseline logits (one-time)
#   2. KL divergence per KV cache type
#
# Usage:
#   ./scripts/turboquant_quality.sh <model_path> [baseline.kld]
#
# If baseline.kld is provided, step 1 is skipped.
# ============================================================================

set -euo pipefail

MODEL="${1:?Usage: $0 <model_path> [baseline.kld]}"
BASELINE="${2:-}"

# Auto-find llama-perplexity
PPL=""
for candidate in \
    "./llama.cpp/build/bin/llama-perplexity" \
    "./build/bin/llama-perplexity" \
    "/usr/local/bin/llama-perplexity"; do
    [[ -x "$candidate" ]] && PPL="$candidate" && break
done
[[ -z "$PPL" ]] && { echo "❌ llama-perplexity not found"; exit 1; }
[[ ! -f "$MODEL" ]] && { echo "❌ Model not found: $MODEL"; exit 1; }

mkdir -p results
TS=$(date +%Y%m%d_%H%M%S)
NAME=$(basename "$MODEL" .gguf)

# Get WikiText-2
WIKI="results/wikitext-2-raw/wiki.test.raw"
if [[ ! -f "$WIKI" ]]; then
    echo "📥 Downloading WikiText-2..."
    mkdir -p results/wikitext-2-raw
    curl -sL "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" \
        -o results/wikitext-2-raw.zip
    unzip -oq results/wikitext-2-raw.zip -d results/
    rm -f results/wikitext-2-raw.zip
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        TurboQuant Quality Benchmark (KL Divergence)        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Model: $MODEL"
echo ""

# Step 1: FP16 baseline
if [[ -z "$BASELINE" ]]; then
    BASELINE="results/logits_f16_${NAME}.kld"
    if [[ ! -f "$BASELINE" ]]; then
        echo "━━━ Step 1: Generating FP16 baseline (large file!) ━━━"
        echo "   Free disk: $(df -h results | tail -1 | awk '{print $4}')"
        "$PPL" -m "$MODEL" -f "$WIKI" \
            --kl-divergence-base "$BASELINE" \
            -ngl 99 2>&1 | tee "results/baseline_log_${TS}.txt"
    else
        echo "✅ Baseline exists: $BASELINE"
    fi
fi
[[ ! -f "$BASELINE" ]] && { echo "❌ Baseline not found: $BASELINE"; exit 1; }

# Step 2: KL divergence per KV type
KV_TYPES=(q8_0 q4_0 tbq4_0 tbq3_0)
RESULTS="results/turboquant_quality_${NAME}_${TS}.json"
echo "[" > "$RESULTS"
FIRST=true

echo ""
echo "━━━ Step 2: KL Divergence per KV cache type ━━━"
for i in "${!KV_TYPES[@]}"; do
    kv="${KV_TYPES[$i]}"
    echo ""
    echo "── [$((i+1))/${#KV_TYPES[@]}] $kv ──"
    
    LOG="results/kld_${kv}_${NAME}_${TS}.txt"
    START=$(date +%s)
    
    if "$PPL" -m "$MODEL" -f "$WIKI" \
        --kl-divergence-base "$BASELINE" --kl-divergence \
        -ctk "$kv" -ctv "$kv" -fa 1 -ngl 99 \
        2>&1 | tee "$LOG"; then
        
        DUR=$(( $(date +%s) - START ))
        KLD_MEAN=$(grep -oP 'mean\s*=\s*\K[0-9.]+' "$LOG" | tail -1 || echo "null")
        KLD_MAX=$(grep -oP 'max\s*=\s*\K[0-9.]+' "$LOG" | tail -1 || echo "null")
        KLD_P999=$(grep -oP '99\.9%\s*=\s*\K[0-9.]+' "$LOG" | tail -1 || echo "null")
        PPL_VAL=$(grep -oP 'perplexity\s*=\s*\K[0-9.]+' "$LOG" | tail -1 || echo "null")
        
        [[ "$FIRST" == "true" ]] && FIRST=false || echo "," >> "$RESULTS"
        cat >> "$RESULTS" <<EOF
  {"kv_type":"${kv}","model":"${NAME}","kld_mean":${KLD_MEAN},"kld_max":${KLD_MAX},"kld_p999":${KLD_P999},"perplexity":${PPL_VAL},"duration_s":${DUR}}
EOF
    else
        [[ "$FIRST" == "true" ]] && FIRST=false || echo "," >> "$RESULTS"
        echo "  {\"kv_type\":\"${kv}\",\"model\":\"${NAME}\",\"error\":\"failed\"}" >> "$RESULTS"
    fi
done

echo "]" >> "$RESULTS"

echo ""
echo "✅ Done! Results: $RESULTS"
echo "Next: python3 scripts/turboquant_report.py --speed results/turboquant_speed_*.csv --quality $RESULTS"
