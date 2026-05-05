#!/usr/bin/env bash
set -euo pipefail

echo "=== vLLM Nemotron Omni Startup ==="

MODEL_NAME="${MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Nemotron-3-Nano-Omni-30B-A3B-Reasoning}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-placeholder-api-key}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
REASONING_PARSER="${REASONING_PARSER:-nemotron_v3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
VIDEO_PRUNING_RATE="${VIDEO_PRUNING_RATE:-0.5}"
VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-256}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"

if [[ -n "$HUGGINGFACE_TOKEN" ]]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

echo "Model:         $MODEL_NAME"
echo "Served as:     $SERVED_MODEL_NAME"
echo "Max ctx len:   $MAX_MODEL_LEN"
echo "GPU util:      $GPU_MEMORY_UTILIZATION"
echo "KV cache:      $KV_CACHE_DTYPE"
echo "Reasoning:     $REASONING_PARSER"
echo "Tool parser:   $TOOL_CALL_PARSER"
echo "Max seqs:      $MAX_NUM_SEQS"
echo "Video:         ${VIDEO_FPS}fps, ${VIDEO_NUM_FRAMES} frames, pruning ${VIDEO_PRUNING_RATE}"
echo ""

# Disable cudagraphs (--enforce-eager) to avoid OOM during capture on 32GB GPU.
# The encoder profiling allocates large buffers that exceed available VRAM.
exec vllm serve "$MODEL_NAME" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --reasoning-parser "$REASONING_PARSER" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --video-pruning-rate "$VIDEO_PRUNING_RATE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enforce-eager
