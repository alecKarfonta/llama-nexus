#!/bin/bash

echo "🚀 Starting Llama.cpp API Server for RTX 5090..."
echo "📊 Model: ${MODEL_NAME} with optimized configuration"
echo "🔌 OpenAI compatible endpoints will be available on port 8080"

# Set required defaults for container startup
MODEL_NAME=${MODEL_NAME:-"Qwen3-Coder-30B-A3B-Instruct"}
MODEL_VARIANT=${MODEL_VARIANT:-"Q4_K_M"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8080"}
API_KEY=${API_KEY:-"placeholder-api-key"}

# Model paths - determine repository based on model name
if [[ "$MODEL_NAME" == "gpt-oss-120b" ]]; then
    MODEL_REPO="unsloth/gpt-oss-120b-GGUF"
elif [[ "$MODEL_NAME" == "gpt-oss-20b" ]]; then
    MODEL_REPO="unsloth/gpt-oss-20b-GGUF"
else
    MODEL_REPO="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
fi

# Try different filename patterns to find the actual model file
# Check multiple patterns for existing files, including multi-part files
patterns=(
    "${MODEL_NAME}-${MODEL_VARIANT}.gguf"
    "${MODEL_NAME}.${MODEL_VARIANT}.gguf"
    "${MODEL_NAME}${MODEL_VARIANT}.gguf"
    "${MODEL_NAME}_${MODEL_VARIANT}.gguf"
    "${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00002.gguf"  # Multi-part file pattern
    "${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00003.gguf"  # 3-part pattern
    # Check in subdirectories (HuggingFace download style)
    "${MODEL_VARIANT}/${MODEL_NAME}-${MODEL_VARIANT}.gguf"  # In variant subdirectory
    "${MODEL_VARIANT}/${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00002.gguf"  # Multi-part in subdirectory
    "${MODEL_VARIANT}/${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00003.gguf"  # 3-part in subdirectory
)

for pattern in "${patterns[@]}"; do
    test_path="/home/llamacpp/models/${pattern}"
    if [ -f "$test_path" ]; then
        MODEL_FILE="$pattern"
        MODEL_PATH="$test_path"
        echo "Found existing model file: $MODEL_PATH"
        break
    fi
done

# If no existing file found, determine the correct download filename
if [ -z "$MODEL_PATH" ]; then
    # For gpt-oss models, use the multi-part filename
    if [[ "$MODEL_NAME" == "gpt-oss-120b" ]]; then
        MODEL_FILE="${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00002.gguf"
    elif [[ "$MODEL_NAME" == "gpt-oss-20b" ]]; then
        MODEL_FILE="${MODEL_NAME}-${MODEL_VARIANT}.gguf"  # Single file for 20b
    else
        MODEL_FILE="${MODEL_NAME}-${MODEL_VARIANT}.gguf"
    fi
    MODEL_PATH="/home/llamacpp/models/${MODEL_FILE}"
    echo "Model file not found, will use: $MODEL_PATH"
fi

# Template selection now supports a directory of templates mounted at /home/llamacpp/templates
# Use CHAT_TEMPLATE env var to pick a file inside that directory
TEMPLATE_DIR="${TEMPLATE_DIR:-/home/llamacpp/templates}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-chat-template-oss.jinja}"
TEMPLATE_PATH="${TEMPLATE_DIR}/${CHAT_TEMPLATE}"

# Download model if not exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Downloading ${MODEL_NAME} model (this may take a while on first run)..."
    echo "   Repository: $MODEL_REPO"
    echo "   Variant: $MODEL_VARIANT (~18.6 GB for Q4_K_M)"
    echo "   File: $MODEL_FILE"
    
    cd /home/llamacpp/models
    python3 -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import hf_hub_download

print('Downloading model...')
hf_hub_download(
    repo_id='${MODEL_REPO}',
    filename='${MODEL_FILE}',
    local_dir='.',
    local_dir_use_symlinks=False
)
print('Download completed!')
"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download model"
        exit 1
    fi
else
    echo "✅ Model already exists: $MODEL_PATH"
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    exit 1
fi

echo "🔧 Starting llama.cpp server..."
echo "   Model: $MODEL_PATH"

# Check if we're being called with arguments (API override) or without (default startup)
if [ $# -eq 0 ]; then
    # Default startup - no arguments passed
    echo "📋 Using default container startup configuration"
    
    # Simple default command - the backend API will override this with proper parameters when needed
    CMD_ARGS=(
        "llama-server"
        "--model" "$MODEL_PATH"
        "--host" "$HOST"
        "--port" "$PORT"
        "--api-key" "$API_KEY"
        "--verbose"
        "--metrics" 
        "--embeddings"
        "--flash-attn"
        "--cont-batching"
    )

    # Add template if it exists
    if [[ -f "$TEMPLATE_PATH" ]]; then
        CMD_ARGS+=("--jinja" "--chat-template-file" "$TEMPLATE_PATH")
    fi

    echo "🚀 Starting llama-server with command:"
    printf '%s ' "${CMD_ARGS[@]}"
    echo ""
    echo ""
    echo "ℹ️  Note: When deployed via API, this command will be overridden with your configured parameters."
    echo ""

    # Execute the command
    exec "${CMD_ARGS[@]}"
else
    # API override - arguments were passed
    echo "🎯 Using API-configured parameters (Deploy page restart)"
    echo "🚀 Starting llama-server with command:"
    printf '%s ' "$@"
    echo ""
    echo ""
    
    # Execute the command with the provided arguments
    exec "$@"
fi