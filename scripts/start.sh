#!/bin/bash

echo "🚀 Starting Llama.cpp API Server for RTX 5090..."
echo "📊 Model: ${MODEL_NAME} with optimized configuration"
echo "🔌 OpenAI compatible endpoints will be available on port 8080"

# Set required defaults for container startup
MODEL_NAME=${MODEL_NAME:-"Qwen3.8-Flash-Next"}
MODEL_VARIANT=${MODEL_VARIANT:-"UD-IQ1_S"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8080"}
API_KEY=${API_KEY:-"placeholder-api-key"}

# Try different filename patterns to find the actual model file
# Check multiple patterns for existing files, including multi-part files
patterns=(
    "${MODEL_NAME}-${MODEL_VARIANT}.gguf"
    "${MODEL_NAME}.${MODEL_VARIANT}.gguf"
    "${MODEL_NAME}${MODEL_VARIANT}.gguf"
    "${MODEL_NAME}_${MODEL_VARIANT}.gguf"
    "${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00002.gguf"  # Multi-part file pattern
    "${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00003.gguf"  # 3-part pattern
    "${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00004.gguf"  # 4-part pattern (e.g. UD-Q4_K_XL Qwen3.8-Flash-Next)
    # Check in subdirectories (HuggingFace download style)
    "${MODEL_VARIANT}/${MODEL_NAME}-${MODEL_VARIANT}.gguf"  # In variant subdirectory
    "${MODEL_VARIANT}/${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00002.gguf"  # Multi-part in subdirectory
    "${MODEL_VARIANT}/${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00003.gguf"  # 3-part in subdirectory
    "${MODEL_VARIANT}/${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00004.gguf"  # 4-part in subdirectory
    "${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00004.gguf"  # 4-part flat
    "gpt-oss-20b-GGUF/${MODEL_NAME}-${MODEL_VARIANT}.gguf" # Specific pattern for gpt-oss-20b
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

# [TIME] | INFO | START.SH.RESOLVE_MODEL | Glob fallback for split GGUFs of any shard count
# (e.g. Qwen3.8-Flash-Next UD-Q4_K_XL ships as 4 shards inside UD-Q4_K_XL/)
if [ -z "$MODEL_PATH" ]; then
    for split_dir in "/home/llamacpp/models" "/home/llamacpp/models/${MODEL_VARIANT}"; do
        split_match=$(ls "${split_dir}/${MODEL_NAME}-${MODEL_VARIANT}-00001-of-"*.gguf 2>/dev/null | head -1)
        if [ -n "$split_match" ]; then
            MODEL_PATH="$split_match"
            MODEL_FILE="${split_match#/home/llamacpp/models/}"
            echo "Found split model via glob: $MODEL_PATH"
            break
        fi
    done
fi

# If no existing file found, determine the correct download filename
if [ -z "$MODEL_PATH" ]; then
    # Allow MODEL_FILE to be passed from environment
    if [ -z "$MODEL_FILE" ]; then
        # For gpt-oss models, use the multi-part filename
        if [[ "$MODEL_NAME" == "gpt-oss-120b" ]]; then
            MODEL_FILE="${MODEL_NAME}-${MODEL_VARIANT}-00001-of-00002.gguf"
        elif [[ "$MODEL_NAME" == "gpt-oss-20b" ]]; then
            MODEL_FILE="gpt-oss-20b-GGUF/${MODEL_NAME}-${MODEL_VARIANT}.gguf"  # Subdirectory in models repo
        else
            MODEL_FILE="${MODEL_NAME}-${MODEL_VARIANT}.gguf"
        fi
    fi
    MODEL_PATH="/home/llamacpp/models/${MODEL_FILE}"
    echo "Model file not found, will use: $MODEL_PATH"
fi

# Template selection now supports a directory of templates mounted at /home/llamacpp/templates
# Use CHAT_TEMPLATE env var to pick a file inside that directory (empty = use GGUF embedded template)
TEMPLATE_DIR="${TEMPLATE_DIR:-/home/llamacpp/templates}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
TEMPLATE_PATH="${TEMPLATE_DIR}/${CHAT_TEMPLATE}"

# Download model if not exists and we have a repository
if [ ! -f "$MODEL_PATH" ] && [ -n "$MODEL_REPO" ]; then
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

# Download multimodal projection file if specified (for vision-language models)
if [ -n "$MMPROJ_FILE" ]; then
    MMPROJ_PATH="/home/llamacpp/models/${MMPROJ_FILE}"
    if [ ! -f "$MMPROJ_PATH" ]; then
        echo "📥 Downloading multimodal projection file for vision support..."
        echo "   File: $MMPROJ_FILE"
        
        cd /home/llamacpp/models
        python3 -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import hf_hub_download

print('Downloading mmproj file...')
hf_hub_download(
    repo_id='${MODEL_REPO}',
    filename='${MMPROJ_FILE}',
    local_dir='.',
    local_dir_use_symlinks=False
)
print('Download completed!')
"
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to download mmproj file"
            exit 1
        fi
    else
        echo "✅ Multimodal projection file already exists: $MMPROJ_PATH"
    fi
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
        "-c" "${CONTEXT_SIZE:-4096}"
        "--api-key" "$API_KEY"
        "--metrics" 
        "--flash-attn" "auto"
        "--cont-batching"
    )
    
    # Add embeddings flag only for non-multimodal models (multimodal models don't support it)
    if [ -z "$MMPROJ_FILE" ]; then
        CMD_ARGS+=("--embeddings")
    fi

    # Add special parameters for Qwen3-VL-4B-Thinking model (reasoning model)
    if [[ "$MODEL_NAME" == "Qwen_Qwen3-VL-4B-Thinking" ]]; then
        CMD_ARGS+=("--reasoning-format" "deepseek")
        # Skip warmup to avoid assertion error with multimodal reasoning models
        CMD_ARGS+=("--no-warmup")
        echo "✨ Enabled reasoning mode for Qwen3-VL-4B-Thinking"
    fi

    # Add multimodal projection file if specified (for vision-language models)
    if [ -n "$MMPROJ_FILE" ] && [ -f "/home/llamacpp/models/${MMPROJ_FILE}" ]; then
        CMD_ARGS+=("--mmproj" "/home/llamacpp/models/${MMPROJ_FILE}")
        echo "🖼️  Enabled vision support with multimodal projection"
    fi

    # [TIME] | INFO | START.SH | qwen4exp n-gram / PLE: keep table on host, lazy-read from SSD
    if [[ -n "${OVERRIDE_TENSOR:-}" ]]; then
        IFS=',' read -ra _ov_specs <<< "$OVERRIDE_TENSOR"
        for spec in "${_ov_specs[@]}"; do
            spec="${spec#"${spec%%[![:space:]]*}"}"
            spec="${spec%"${spec##*[![:space:]]}"}"
            if [[ -n "$spec" ]]; then
                CMD_ARGS+=("--override-tensor" "$spec")
            fi
        done
    fi
    if [[ -n "${TENSOR_READ_LAZY:-}" ]]; then
        CMD_ARGS+=("--tensor-read-lazy" "$TENSOR_READ_LAZY")
    fi

    # Template modes:
    #   empty CHAT_TEMPLATE          = GGUF embedded, no --jinja (legacy default)
    #   CHAT_TEMPLATE=__embedded_jinja__ = GGUF embedded + --jinja only (multi-turn chat)
    #   CHAT_TEMPLATE=file.jinja     = custom file + --jinja + --chat-template-file
    if [[ "$CHAT_TEMPLATE" == "__embedded_jinja__" ]]; then
        CMD_ARGS+=("--jinja")
    elif [[ -n "$CHAT_TEMPLATE" && -f "$TEMPLATE_PATH" ]]; then
        CMD_ARGS+=("--jinja" "--chat-template-file" "$TEMPLATE_PATH")
    fi

    # Multi-Token Prediction (MTP) — requires MTP-converted GGUF + llama.cpp b9193+
    if [[ "${MTP_ENABLED:-false}" == "true" || "${MTP_ENABLED}" == "1" ]]; then
        CMD_ARGS+=(
            "--spec-type" "draft-mtp"
            "--spec-draft-n-max" "${MTP_DRAFT_N_MAX:-3}"
            "--spec-draft-n-min" "${MTP_DRAFT_N_MIN:-0}"
            "--spec-draft-p-min" "${MTP_DRAFT_P_MIN:-0.75}"
        )
        echo "⚡ MTP speculative decoding enabled (draft-n-max=${MTP_DRAFT_N_MAX:-3})"
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