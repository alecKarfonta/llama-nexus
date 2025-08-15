#!/bin/bash

echo "🚀 Starting Llama.cpp Qwen3-Coder API Server for RTX 5090..."
echo "📊 Model: Qwen3-Coder-30B with CLINE/RooCode optimization"
echo "🔌 OpenAI compatible endpoints will be available on port 8080"

# Set defaults from environment variables
MODEL_NAME=${MODEL_NAME:-"Qwen3-Coder-30B-A3B-Instruct"}
MODEL_VARIANT=${MODEL_VARIANT:-"Q4_K_M"}
CONTEXT_SIZE=${CONTEXT_SIZE:-65536}
GPU_LAYERS=${GPU_LAYERS:-999}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.8}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.03}
REPEAT_PENALTY=${REPEAT_PENALTY:-1.05}
NUM_KEEP=${NUM_KEEP:-1024}
NUM_PREDICT=${NUM_PREDICT:-32768}
# Aggressive DRY sampling parameters to prevent repetition
DRY_MULTIPLIER=${DRY_MULTIPLIER:-0.6}
DRY_BASE=${DRY_BASE:-2.0}
DRY_ALLOWED_LENGTH=${DRY_ALLOWED_LENGTH:-1}
DRY_PENALTY_LAST_N=${DRY_PENALTY_LAST_N:-1024}
# Additional anti-repetition controls
REPEAT_LAST_N=${REPEAT_LAST_N:-256}
FREQUENCY_PENALTY=${FREQUENCY_PENALTY:-0.3}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.2}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
API_KEY=${API_KEY:-"placeholder-api-key"}
THREADS=${THREADS:--1}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
N_CPU_MOE=${N_CPU_MOE:-0}

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
    echo "📥 Downloading Qwen3-Coder-30B model (this may take a while on first run)..."
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

echo "🔧 Starting llama.cpp server with AGGRESSIVE anti-repetition settings..."
echo "   Model: $MODEL_PATH"
echo "   Context Size: $CONTEXT_SIZE"
echo "   GPU Layers: $GPU_LAYERS"
echo "   Temperature: $TEMPERATURE"
echo "   Top-P: $TOP_P"
echo "   Top-K: $TOP_K"
echo "   Min-P: $MIN_P"
echo "   Repeat Penalty: $REPEAT_PENALTY (aggressive)"
echo "   Repeat Last N: $REPEAT_LAST_N"
echo "   Frequency Penalty: $FREQUENCY_PENALTY"
echo "   Presence Penalty: $PRESENCE_PENALTY"
echo "   DRY Multiplier: $DRY_MULTIPLIER (aggressive)"
echo "   DRY Base: $DRY_BASE (aggressive)"
echo "   DRY Allowed Length: $DRY_ALLOWED_LENGTH (strict)"
echo "   DRY Penalty Last N: $DRY_PENALTY_LAST_N"
echo "   Max Predict Tokens: $NUM_PREDICT"
echo "   Threads: $THREADS"
echo "   Batch Size: $BATCH_SIZE"
echo ""

# Start llama.cpp server with aggressive anti-repetition for tool calling
exec llama-server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --ctx-size "$CONTEXT_SIZE" \
    --n-gpu-layers "$GPU_LAYERS" \
    --n-predict "$NUM_PREDICT" \
    --threads "$THREADS" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --n-cpu-moe "$N_CPU_MOE" \
    --temp "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    --repeat-penalty "$REPEAT_PENALTY" \
    --repeat-last-n "$REPEAT_LAST_N" \
    --frequency-penalty "$FREQUENCY_PENALTY" \
    --presence-penalty "$PRESENCE_PENALTY" \
    --dry-multiplier "$DRY_MULTIPLIER" \
    --dry-base "$DRY_BASE" \
    --dry-allowed-length "$DRY_ALLOWED_LENGTH" \
    --dry-penalty-last-n "$DRY_PENALTY_LAST_N" \
    --jinja \
    --chat-template-file "$TEMPLATE_PATH" \
    --verbose \
    --metrics \
    --embeddings \
    --flash-attn \
    --cont-batching