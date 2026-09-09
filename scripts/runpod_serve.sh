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
# --parallel: multi-slot aborted with an indexer-cache assert in the *day-0* state
# of PR #27742, but that is fixed on this build (notes.md 2026-08-30 "parallel
# slots unlocked": 2 streams gave +10-15% aggregate tok/s and -20% wall). The
# live container still runs 1 only because an old Agent-profile deploy pinned
# parallel_slots=1. Benchmarks want 1 (uncontended timings); a throughput job
# wants more. NOTE: N slots divide ctx N ways - at 196608 that is 98k at N=2 and
# 65k at N=3, against a measured max prompt of ~60k tokens.

set -uo pipefail

# Required on build 10654 / commit bea3b12da: without it the model load aborts on
#   llama-graph.cpp: GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr)
# The running llamacpp-api container sets this too (notes.md claims it is not
# needed on this rig, which is stale - the deployed container does set it).
export LLAMA_ATTN_ROT_DISABLE="${LLAMA_ATTN_ROT_DISABLE:-1}"

WORK=/workspace
MODELS="$WORK/models"
BIN="$WORK/llama.cpp/build/bin/llama-server"
API_KEY="${LLAMA_API_KEY:?LLAMA_API_KEY must be set}"
CTX="${CTX:-196608}"

# One server per GPU rather than one server tensor-split across all of them:
# a split would serialize every token across PCIe for no benefit when each card
# already fits the whole model. CUDA_VISIBLE_DEVICES pins each server to its own
# card, which is why --main-gpu stays 0 below: index 0 is relative to the
# visible set.
GPU_INDEX="${GPU_INDEX:-0}"
PORT="${PORT:-8080}"
PARALLEL="${PARALLEL:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
# Each server mmaps the same 104 GB of weights off one disk, so the page cache
# for the host-resident PLE table is shared rather than duplicated per server.
NSERVERS="${NSERVERS:-1}"

MAIN=$(ls "$MODELS"/UD-Q4_K_XL/*-00001-of-*.gguf 2>/dev/null | head -1)
DRAFT="$MODELS/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf"

if [ -z "$MAIN" ] || [ ! -f "$DRAFT" ]; then
  echo "ERROR missing weights: main='$MAIN' draft='$DRAFT'"
  ls -la "$MODELS" "$MODELS/UD-Q4_K_XL" 2>/dev/null
  exit 1
fi

THREADS=$(( $(nproc) / NSERVERS ))
[ "$THREADS" -lt 4 ] && THREADS=4
echo "[serve] main=$MAIN"
echo "[serve] draft=$DRAFT ctx=$CTX threads=$THREADS gpu=$GPU_INDEX port=$PORT parallel=$PARALLEL ($((CTX/PARALLEL)) tok/slot)"

exec "$BIN" \
  --model "$MAIN" \
  --host 0.0.0.0 \
  --port "$PORT" \
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
  --parallel "$PARALLEL" \
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
