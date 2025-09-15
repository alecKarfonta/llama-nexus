#!/bin/bash

echo "🚀 Starting vLLM Server with Native MXFP4 GPT-OSS-20B..."
echo "📊 Model: openai/gpt-oss-20b (Native MXFP4 Quantization)"
echo "🔌 OpenAI compatible endpoints will be available on port 8080"

# Set defaults from environment variables
MODEL_NAME=${MODEL_NAME:-"openai/gpt-oss-20b"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8080"}
API_KEY=${API_KEY:-"placeholder-api-key"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
DTYPE=${DTYPE:-"auto"}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-"true"}

# vLLM specific settings for GPT-OSS
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-"true"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

echo "📋 vLLM Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Host: $HOST:$PORT"
echo "   Max Model Length: $MAX_MODEL_LEN tokens"
echo "   GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "   Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "   Data Type: $DTYPE"
echo "   Trust Remote Code: $TRUST_REMOTE_CODE"
echo "   Chunked Prefill: $ENABLE_CHUNKED_PREFILL"
echo "   Max Batched Tokens: $MAX_NUM_BATCHED_TOKENS"
echo "   Max Sequences: $MAX_NUM_SEQS"

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits

# RTX 5090 specific environment variables (enable FlashAttention for GPT-OSS)
echo "🔧 Setting RTX 5090 compatibility mode - enabling FlashAttention for GPT-OSS..."
export VLLM_USE_FLASH_ATTN=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export CUDA_VISIBLE_DEVICES=0

# Build vLLM command with RTX 5090 optimizations (using A100 config)
VLLM_CMD="vllm serve $MODEL_NAME"
VLLM_CMD="$VLLM_CMD --host $HOST"
VLLM_CMD="$VLLM_CMD --port $PORT"
VLLM_CMD="$VLLM_CMD --api-key $API_KEY"
VLLM_CMD="$VLLM_CMD --max-model-len $MAX_MODEL_LEN"
VLLM_CMD="$VLLM_CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
VLLM_CMD="$VLLM_CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
VLLM_CMD="$VLLM_CMD --dtype $DTYPE"
VLLM_CMD="$VLLM_CMD --trust-remote-code"
VLLM_CMD="$VLLM_CMD --enable-chunked-prefill"
VLLM_CMD="$VLLM_CMD --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"
VLLM_CMD="$VLLM_CMD --max-num-seqs $MAX_NUM_SEQS"
# RTX 5090 compatibility: Disable advanced features that may not work
VLLM_CMD="$VLLM_CMD --disable-custom-all-reduce"
VLLM_CMD="$VLLM_CMD --enforce-eager"

# Add optional parameters if set
if [ ! -z "$SERVED_MODEL_NAME" ]; then
    VLLM_CMD="$VLLM_CMD --served-model-name $SERVED_MODEL_NAME"
fi

if [ ! -z "$CHAT_TEMPLATE" ]; then
    VLLM_CMD="$VLLM_CMD --chat-template $CHAT_TEMPLATE"
fi

echo ""
echo "🚀 Starting vLLM with command:"
echo "$VLLM_CMD"
echo ""

# Start vLLM server
exec $VLLM_CMD
