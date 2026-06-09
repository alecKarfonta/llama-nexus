#!/usr/bin/env bash
# 10_llamacpp_qwen36_35b_mtp.sh — Qwen3.6-35B-A3B (MoE) with built-in MTP
# speculative decoding on llama.cpp (build b9180+).
#
# Why this config:
#  * 35B-A3B has only ~3B active params -> already fast at decode; MTP adds
#    ~1.4-2.2x on top (best on code/structured output, less on hot-temp chat).
#  * Single-GPU is usually the right call on a PCIe-only TRX40: Q4_K_XL fits
#    one 3090 Ti with ~100K ctx, and you avoid cross-GPU sync entirely.
#    Set GPUS=all to layer-split across the rig for huge context instead.
#
# Tuning knobs that matter (measured by the community on 3090-class cards):
#  * --spec-draft-n-max: 2-3 for MoE (acceptance drops at wider windows;
#    dense 27B tolerates 3-5). If acceptance <50%, lower n-max / raise p-min.
#  * --spec-draft-p-min: 0.75 default; 0.8+ for high-temp chat workloads.
#  * KV cache q8_0 ~halves KV memory with negligible quality loss; keep f16
#    if you have headroom and care about long-context recall.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-$(ls "$MODEL_DIR"/qwen3.6-35b-mtp/*.gguf 2>/dev/null | head -1)}"
[[ -n "$MODEL" ]] || { echo "No GGUF found; run 01_download_models.sh"; exit 1; }

PORT="${PORT:-8080}"
CTX="${CTX:-131072}"          # raise toward 262144 if VRAM allows
GPUS="${GPUS:-0}"             # "0" single-GPU (recommended) | "all" layer-split
NMAX="${NMAX:-3}"             # spec draft tokens
PMIN="${PMIN:-0.75}"
KV="${KV:-q8_0}"              # q8_0 | f16
ALIAS="${ALIAS:-qwen3.6-35b-mtp}"
SPEC="${SPEC:-on}"            # on | off (off = baseline for A/B benchmarking)

SPEC_ARGS=()
if [[ "$SPEC" == "on" ]]; then
  SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "$NMAX" --spec-draft-p-min "$PMIN")
fi

SPLIT_ARGS=()
if [[ "$GPUS" == "all" ]]; then
  SPLIT_ARGS=(--split-mode layer)      # layer split: right call without NVLink
else
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

exec llama-server \
  --model "$MODEL" \
  --alias "$ALIAS" \
  --host 0.0.0.0 --port "$PORT" \
  -ngl 99 "${SPLIT_ARGS[@]}" \
  --ctx-size "$CTX" \
  --flash-attn on \
  -ctk "$KV" -ctv "$KV" \
  --cache-reuse 1024 \
  --no-mmap --mlock \
  --parallel "${NP:-2}" \
  --metrics \
  --jinja --chat-template-kwargs '{"preserve_thinking": true}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  "${SPEC_ARGS[@]}"

# A/B test:  SPEC=off ./10_llamacpp_qwen36_35b_mtp.sh  -> no-MTP baseline,
# then bench both with bench/bench.py.
