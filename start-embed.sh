#!/bin/bash

echo "🚀 Starting Llama.cpp Embedding Server..."
echo "📊 Model: ${MODEL_NAME} with optimized configuration"
echo "🔌 OpenAI compatible endpoints will be available on port 8080"

# Set required defaults for container startup
MODEL_NAME=${MODEL_NAME:-"nomic-embed-text-v1.5"}
MODEL_VARIANT=${MODEL_VARIANT:-"Q8_0"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8080"}
API_KEY=${API_KEY:-"llamacpp-embed"}

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

# Model paths - determine repository based on model name or use provided MODEL_REPO
if [ -z "$MODEL_REPO" ]; then
    echo "⚠️  No repository metadata found for ${MODEL_NAME}-${MODEL_VARIANT}"
    echo "   This model was likely added manually or downloaded outside the system."
    echo "   Checking if model file exists locally..."
    
    # Check if model file exists locally first
    MODEL_EXISTS=false
    for pattern in "${patterns[@]}"; do
        test_path="/home/llamacpp/models/${pattern}"
        if [ -f "$test_path" ]; then
            MODEL_EXISTS=true
            MODEL_FILE="$pattern"
            MODEL_PATH="$test_path"
            echo "✅ Found local model file: $MODEL_PATH"
            break
        fi
    done
    
    if [ "$MODEL_EXISTS" = true ]; then
        echo "🎉 Using existing local model file"
        # Skip download since we have the file
        MODEL_REPO=""
    else
        echo "❌ Model file not found locally and no repository metadata available"
        echo "   Cannot download or use this model."
        echo "   Please either:"
        echo "   1. Download the model through the web interface (recommended)"
        echo "   2. Manually place the model file in /home/llamacpp/models/"
        echo "   3. Set MODEL_REPO environment variable if you know the repository"
        exit 1
    fi
fi

echo "📦 Model Repository: ${MODEL_REPO:-'(using local file)'}"

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
    MODEL_FILE="${MODEL_NAME}.${MODEL_VARIANT}.gguf"
    MODEL_PATH="/home/llamacpp/models/${MODEL_FILE}"
    echo "Model file not found, will use: $MODEL_PATH"
fi

# Download model if not exists and we have a repository
if [ ! -f "$MODEL_PATH" ] && [ -n "$MODEL_REPO" ]; then
    echo "📥 Downloading ${MODEL_NAME} model (this may take a while on first run)..."
    echo "   Repository: $MODEL_REPO"
    echo "   Variant: $MODEL_VARIANT"
    
    # Use huggingface-cli to download the model
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        echo "🔑 Using HuggingFace token for authentication"
        huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential
    fi
    
    # Download the specific model file
    huggingface-cli download "$MODEL_REPO" "$MODEL_FILE" --local-dir /home/llamacpp/models --local-dir-use-symlinks False
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "❌ Failed to download model file: $MODEL_PATH"
        exit 1
    fi
    
    echo "✅ Model downloaded successfully"
else
    echo "✅ Using existing model file: $MODEL_PATH"
fi

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    exit 1
fi

echo "🔧 Model file size: $(du -h "$MODEL_PATH" | cut -f1)"

# Build llama-server command for embedding
LLAMA_CMD="llama-server"
LLAMA_CMD="$LLAMA_CMD --model $MODEL_PATH"
LLAMA_CMD="$LLAMA_CMD --host $HOST"
LLAMA_CMD="$LLAMA_CMD --port $PORT"
LLAMA_CMD="$LLAMA_CMD --api-key $API_KEY"

# Embedding-specific parameters
LLAMA_CMD="$LLAMA_CMD --embeddings"
LLAMA_CMD="$LLAMA_CMD --pooling ${POOLING_TYPE:-mean}"
LLAMA_CMD="$LLAMA_CMD --ctx-size ${CONTEXT_SIZE:-8192}"
LLAMA_CMD="$LLAMA_CMD --batch-size ${BATCH_SIZE:-512}"
LLAMA_CMD="$LLAMA_CMD --ubatch-size ${UBATCH_SIZE:-512}"

# GPU configuration
if [ "${GPU_LAYERS:-999}" != "0" ]; then
    LLAMA_CMD="$LLAMA_CMD --n-gpu-layers ${GPU_LAYERS:-999}"
else
    LLAMA_CMD="$LLAMA_CMD --n-gpu-layers 0"
fi

# Thread configuration
if [ "${THREADS:--1}" != "-1" ]; then
    LLAMA_CMD="$LLAMA_CMD --threads ${THREADS}"
fi

# Additional optimization flags
LLAMA_CMD="$LLAMA_CMD --metrics"
LLAMA_CMD="$LLAMA_CMD --verbose"
LLAMA_CMD="$LLAMA_CMD --flash-attn auto"
LLAMA_CMD="$LLAMA_CMD --cont-batching"

echo "🚀 Starting llama-server with command:"
echo "   $LLAMA_CMD"
echo ""

# Start the server
exec $LLAMA_CMD
