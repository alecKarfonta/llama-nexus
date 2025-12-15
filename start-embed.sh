#!/bin/bash

echo "Starting Llama.cpp Embedding Server..."
echo "Model: ${MODEL_NAME:-nomic-embed-text-v1.5}"
echo "OpenAI compatible embedding endpoint will be available on port ${PORT:-8080}"

# Set required defaults for container startup
MODEL_NAME=${MODEL_NAME:-"nomic-embed-text-v1.5"}
MODEL_VARIANT=${MODEL_VARIANT:-"Q8_0"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8080"}
API_KEY=${API_KEY:-"llamacpp-embed"}
CONTEXT_SIZE=${CONTEXT_SIZE:-2048}
GPU_LAYERS=${GPU_LAYERS:-0}
THREADS=${THREADS:-8}
BATCH_SIZE=${BATCH_SIZE:-512}
UBATCH_SIZE=${UBATCH_SIZE:-512}
POOLING_TYPE=${POOLING_TYPE:-mean}

# Determine model repository for nomic-embed
# Use official nomic-ai repository for the model
if [ -z "$MODEL_REPO" ]; then
    if [[ "$MODEL_NAME" == "nomic-embed-text-v1.5" ]]; then
        MODEL_REPO="nomic-ai/nomic-embed-text-v1.5-GGUF"
        # The actual filename in the repo
        MODEL_FILE="${MODEL_NAME}.${MODEL_VARIANT}.gguf"
    else
        echo "Unknown embedding model: ${MODEL_NAME}"
        echo "Please specify MODEL_REPO environment variable"
        exit 1
    fi
fi

echo "Model Repository: ${MODEL_REPO}"

# Model path
MODEL_PATH="/home/llamacpp/models/${MODEL_FILE}"

# Download model if not exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading ${MODEL_NAME} embedding model (this may take a while on first run)..."
    echo "   Repository: $MODEL_REPO"
    echo "   Variant: $MODEL_VARIANT"
    echo "   File: $MODEL_FILE"
    
    cd /home/llamacpp/models
    python3 -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import hf_hub_download

print('Downloading embedding model...')
hf_hub_download(
    repo_id='${MODEL_REPO}',
    filename='${MODEL_FILE}',
    local_dir='.',
    local_dir_use_symlinks=False
)
print('Download completed!')
"
    
    if [ $? -ne 0 ]; then
        echo "Failed to download embedding model"
        exit 1
    fi
else
    echo "Embedding model already exists: $MODEL_PATH"
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file not found: $MODEL_PATH"
    exit 1
fi

echo "Starting llama.cpp embedding server..."
echo "   Model: $MODEL_PATH"
echo "   GPU Layers: $GPU_LAYERS"
echo "   Context Size: $CONTEXT_SIZE"
echo "   Pooling Type: $POOLING_TYPE"

# Build command for embedding server
CMD_ARGS=(
    "llama-server"
    "--model" "$MODEL_PATH"
    "--host" "$HOST"
    "--port" "$PORT"
    "-c" "$CONTEXT_SIZE"
    "-ngl" "$GPU_LAYERS"
    "--api-key" "$API_KEY"
    "--threads" "$THREADS"
    "-b" "$BATCH_SIZE"
    "-ub" "$UBATCH_SIZE"
    "--pooling" "$POOLING_TYPE"
    "--embeddings"
    "--verbose"
    "--metrics"
    "--log-disable"
)

echo "Starting llama-server with command:"
printf '%s ' "${CMD_ARGS[@]}"
echo ""
echo ""

# Execute the command
exec "${CMD_ARGS[@]}"
