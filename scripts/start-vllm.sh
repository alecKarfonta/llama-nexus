#!/usr/bin/env bash
set -euo pipefail

echo "=== vLLM startup (env-driven; matches Deploy / VLLMManager config) ==="

MODEL_NAME="${MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Nemotron-3-Nano-Omni-30B-A3B-Reasoning}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-placeholder-api-key}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

REASONING_PARSER="${REASONING_PARSER:-nemotron_v3}"
REASONING_PARSER_PLUGIN="${REASONING_PARSER_PLUGIN:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

VIDEO_PRUNING_RATE="${VIDEO_PRUNING_RATE:-0.5}"
VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-256}"

VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"

VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-true}"
VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-true}"

MOE_BACKEND="${MOE_BACKEND:-}"
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-true}"

VLLM_SPECULATIVE_CONFIG="${VLLM_SPECULATIVE_CONFIG:-}"

HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"

VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-marlin}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
VLLM_FLASHINFER_ALLREDUCE_BACKEND="${VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}"
VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-0}"
export VLLM_NVFP4_GEMM_BACKEND VLLM_ALLOW_LONG_MAX_MODEL_LEN VLLM_FLASHINFER_ALLREDUCE_BACKEND VLLM_USE_FLASHINFER_MOE_FP4

export VIDEO_FPS VIDEO_NUM_FRAMES

if [[ -n "$HUGGINGFACE_TOKEN" ]]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

echo "Model:          $MODEL_NAME"
echo "Served as:      $SERVED_MODEL_NAME"
echo "Listen:         $HOST:$PORT"
echo "Max ctx len:    $MAX_MODEL_LEN"
echo "GPU util:       $GPU_MEMORY_UTILIZATION"
echo "Dtype:          $VLLM_DTYPE"
echo "Tensor parallel $TENSOR_PARALLEL_SIZE | PP=$PIPELINE_PARALLEL_SIZE DP=$DATA_PARALLEL_SIZE"
echo ""

CMD=(
    vllm serve "$MODEL_NAME"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --api-key "$API_KEY"
    --dtype "$VLLM_DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --video-pruning-rate "$VIDEO_PRUNING_RATE"
)

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
    CMD+=(--trust-remote-code)
fi

if [[ "$VLLM_ENFORCE_EAGER" == "true" ]]; then
    CMD+=(--enforce-eager)
fi

if [[ "${PIPELINE_PARALLEL_SIZE}" =~ ^[0-9]+$ ]] && [[ "${PIPELINE_PARALLEL_SIZE}" -gt 1 ]]; then
    CMD+=(--pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE")
fi

if [[ "${DATA_PARALLEL_SIZE}" =~ ^[0-9]+$ ]] && [[ "${DATA_PARALLEL_SIZE}" -gt 1 ]]; then
    CMD+=(--data-parallel-size "$DATA_PARALLEL_SIZE")
fi

if [[ "$VLLM_ENABLE_CHUNKED_PREFILL" == "true" ]]; then
    CMD+=(--enable-chunked-prefill)
fi

if [[ "$VLLM_ASYNC_SCHEDULING" == "true" ]]; then
    CMD+=(--async-scheduling)
fi

if [[ -n "$MOE_BACKEND" ]]; then
    CMD+=(--moe-backend "$MOE_BACKEND")
fi

if [[ -n "$MAMBA_SSM_CACHE_DTYPE" ]]; then
    CMD+=(--mamba_ssm_cache_dtype "$MAMBA_SSM_CACHE_DTYPE")
fi

ql="$(printf '%s' "$VLLM_QUANTIZATION" | tr '[:upper:]' '[:lower:]')"
if [[ -n "$VLLM_QUANTIZATION" && "$ql" != "none" ]]; then
    CMD+=(--quantization "$VLLM_QUANTIZATION")
fi

rpl="$(printf '%s' "$REASONING_PARSER" | tr '[:upper:]' '[:lower:]')"
if [[ -n "$REASONING_PARSER" && "$rpl" != "none" ]]; then
    CMD+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ -n "$REASONING_PARSER_PLUGIN" ]]; then
    CMD+=(--reasoning-parser-plugin "$REASONING_PARSER_PLUGIN")
fi

if [[ "$VLLM_ENABLE_AUTO_TOOL_CHOICE" == "true" ]]; then
    CMD+=(--enable-auto-tool-choice)
fi

if [[ -n "$TOOL_CALL_PARSER" ]]; then
    CMD+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ -n "$VLLM_SPECULATIVE_CONFIG" ]]; then
    CMD+=(--speculative_config "$VLLM_SPECULATIVE_CONFIG")
fi

exec "${CMD[@]}"
