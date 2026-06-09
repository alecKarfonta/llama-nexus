#!/usr/bin/env bash
# 11_llamacpp_qwen36_27b_mtp.sh — Qwen3.6-27B (dense, hybrid GatedDeltaNet) + MTP.
# Use this for coding/quality workloads; the dense model benefits MORE from MTP
# than the MoE (decode is expensive, so accepted drafts save more). ~1.5-2x typical.
# Q4_K_M (~17GB) fits one 3090 Ti; two cards give you full 262K ctx + vision.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-$(ls "$MODEL_DIR"/qwen3.6-27b-mtp/*Q4_K_M*.gguf 2>/dev/null | head -1)}"
[[ -n "$MODEL" ]] || { echo "No GGUF found; run 01_download_models.sh"; exit 1; }

PORT="${PORT:-8081}"
CTX="${CTX:-131072}"
GPUS="${GPUS:-1}"             # park it on a different card than the 35B
NMAX="${NMAX:-4}"             # dense tolerates wider draft windows (3-5)
PMIN="${PMIN:-0.75}"
KV="${KV:-q8_0}"
SPEC="${SPEC:-on}"
MMPROJ="${MMPROJ:-}"          # set to mmproj gguf path to enable vision

export CUDA_VISIBLE_DEVICES="$GPUS"

ARGS=(
  --model "$MODEL"
  --alias qwen3.6-27b-mtp
  --host 0.0.0.0 --port "$PORT"
  -ngl 99
  --ctx-size "$CTX"
  --flash-attn on
  -ctk "$KV" -ctv "$KV"
  --cache-reuse 1024
  --no-mmap --mlock
  --metrics
  --jinja --chat-template-kwargs '{"preserve_thinking": true}'
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
)
[[ -n "$MMPROJ" ]] && ARGS+=(--mmproj "$MMPROJ") || ARGS+=(--no-mmproj)
[[ "$SPEC" == "on" ]] && ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$NMAX" --spec-draft-p-min "$PMIN")

exec llama-server "${ARGS[@]}"
