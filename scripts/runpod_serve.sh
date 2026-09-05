#!/usr/bin/env bash
# [TIME] | INFO | RUNPOD_SERVE | launch llama-server on the pod with the same
# knobs as the local 4x3090 Ti deploy, so bench numbers are comparable.
#
# Differences from local, all forced by having one GPU instead of four:
#   --tensor-split   dropped (nothing to split)
#   --threads        sized to pod vCPUs rather than the rig's 40
# Everything else (ctx, q8_0 KV, MTP spec-decode depth, cache-reuse, sampling)
# is copied verbatim from the running llamacpp-api container.
#
# --parallel stays 1: qwen4exp multi-slot trips an indexer-cache assert in
# PR #27742. Not a tuning choice.

set -uo pipefail

WORK=/workspace
MODELS="$WORK/models"
BIN="$WORK/llama.cpp/build/bin/llama-server"
API_KEY="${LLAMA_API_KEY:?LLAMA_API_KEY must be set}"
CTX="${CTX:-196608}"

MAIN=$(ls "$MODELS"/UD-Q4_K_XL/*-00001-of-*.gguf 2>/dev/null | head -1)
DRAFT="$MODELS/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf"

if [ -z "$MAIN" ] || [ ! -f "$DRAFT" ]; then
  echo "ERROR missing weights: main='$MAIN' draft='$DRAFT'"
  ls -la "$MODELS" "$MODELS/UD-Q4_K_XL" 2>/dev/null
  exit 1
fi

THREADS=$(nproc)
echo "[serve] main=$MAIN"
echo "[serve] draft=$DRAFT ctx=$CTX threads=$THREADS"

exec "$BIN" \
  --model "$MAIN" \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key "$API_KEY" \
  --ctx-size "$CTX" \
  --n-gpu-layers 999 \
  --n-cpu-moe 0 \
  --override-tensor per_layer_token_embd=CPU \
  --n-predict 16000 \
  --threads "$THREADS" \
  --batch-size 512 \
  --ubatch-size 256 \
  --keep 256 \
  --parallel 1 \
  --split-mode layer \
  --main-gpu 0 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --repeat-penalty 1.0 \
  --repeat-last-n 64 \
  --frequency-penalty 0.0 \
  --presence-penalty 0.0 \
  --alias qwen38-flash \
  --cache-reuse 128 \
  --reasoning-budget 0 \
  --metrics \
  --jinja \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  --flash-attn on \
  --spec-type draft-mtp \
  --model-draft "$DRAFT" \
  --gpu-layers-draft 999 \
  --spec-draft-n-max 16 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.75 \
  --poll 100 \
  --spec-draft-poll 1
