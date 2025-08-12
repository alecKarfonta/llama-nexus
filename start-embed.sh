#!/bin/bash

echo "🚀 Starting Llama.cpp Embedding Server..."
echo "📊 Optimized for fast embedding generation"
echo "🔌 OpenAI compatible embeddings endpoint on port 8080"

# Set defaults from environment variables
MODEL_NAME=${MODEL_NAME:-"nomic-embed-text-v1.5"}
MODEL_VARIANT=${MODEL_VARIANT:-"Q8_0"}
CONTEXT_SIZE=${CONTEXT_SIZE:-8192}
GPU_LAYERS=${GPU_LAYERS:-999}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
API_KEY=${API_KEY:-"llamacpp-embed"}
THREADS=${THREADS:--1}
BATCH_SIZE=${BATCH_SIZE:-512}
UBATCH_SIZE=${UBATCH_SIZE:-512}
POOLING_TYPE=${POOLING_TYPE:-"mean"}

# Model repository mapping
case "$MODEL_NAME" in
    "nomic-embed-text-v1.5")
        MODEL_REPO="nomic-ai/nomic-embed-text-v1.5-GGUF"
        MODEL_FILE="nomic-embed-text-v1.5.${MODEL_VARIANT}.gguf"
        ;;
    "e5-mistral-7b")
        MODEL_REPO="intfloat/e5-mistral-7b-instruct-GGUF"
        MODEL_FILE="e5-mistral-7b-instruct.${MODEL_VARIANT}.gguf"
        ;;
    "bge-m3")
        MODEL_REPO="BAAI/bge-m3-GGUF"
        MODEL_FILE="bge-m3.${MODEL_VARIANT}.gguf"
        ;;
    "gte-Qwen2-1.5B")
        MODEL_REPO="Alibaba-NLP/gte-Qwen2-1.5B-instruct-GGUF"
        MODEL_FILE="gte-Qwen2-1.5B-instruct.${MODEL_VARIANT}.gguf"
        ;;
    *)
        echo "❌ Unknown embedding model: $MODEL_NAME"
        echo "   Supported models: nomic-embed-text-v1.5, e5-mistral-7b, bge-m3, gte-Qwen2-1.5B"
        exit 1
        ;;
esac

MODEL_PATH="/home/llamacpp/models/${MODEL_FILE}"

# Download model if not exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Downloading embedding model..."
    echo "   Model: $MODEL_NAME"
    echo "   Repository: $MODEL_REPO"
    echo "   File: $MODEL_FILE"
    
    cd /home/llamacpp/models
    python3 -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import hf_hub_download

try:
    file_path = hf_hub_download(
        repo_id='$MODEL_REPO',
        filename='$MODEL_FILE',
        local_dir='/home/llamacpp/models',
        local_dir_use_symlinks=False
    )
    print(f'✅ Model downloaded successfully to {file_path}')
except Exception as e:
    print(f'❌ Download failed: {e}')
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download model"
        exit 1
    fi
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    exit 1
fi

echo "🔧 Starting llama.cpp embedding server..."
echo "   Model: $MODEL_PATH"
echo "   Context Size: $CONTEXT_SIZE"
echo "   GPU Layers: $GPU_LAYERS"
echo "   Pooling: $POOLING_TYPE"
echo "   Batch Size: $BATCH_SIZE"
echo ""

# Start llama.cpp server in embedding mode
exec llama-server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --ctx-size "$CONTEXT_SIZE" \
    --n-gpu-layers "$GPU_LAYERS" \
    --threads "$THREADS" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --embeddings \
    --pooling "$POOLING_TYPE" \
    --verbose \
    --metrics \
    --flash-attn \
    --cont-batching