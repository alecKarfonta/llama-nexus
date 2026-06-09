#!/usr/bin/env bash
# 01_download_models.sh — pull the MTP-enabled GGUFs (llama.cpp) and AWQ (vLLM).
# CRITICAL: standard Qwen3.6 GGUFs do NOT contain the MTP draft head.
# Only the *-MTP-GGUF repos work with --spec-type draft-mtp.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
mkdir -p "$MODEL_DIR"
export HF_XET_HIGH_PERFORMANCE=1   # fast Xet downloads (replaces deprecated HF_HUB_ENABLE_HF_TRANSFER)

# Pick ONE quant per model to start; uncomment others as needed.

echo "== Qwen3.6-35B-A3B (MoE, 3B active) — primary VTuber-chat / agent model =="
# Q4_K_XL ~ fits a single 24GB card with ~100K ctx; UD = Unsloth Dynamic quants
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
  --include '*UD-Q4_K_XL*' \
  --local-dir "$MODEL_DIR/qwen3.6-35b-mtp"

# Higher quality if you span 2 GPUs or want more headroom:
# hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF --include "*UD-Q6_K_XL*" \
#   --local-dir "$MODEL_DIR/qwen3.6-35b-mtp"

echo "== Qwen3.6-27B (dense, vision) — coding / quality-leaning workloads =="
hf download unsloth/Qwen3.6-27B-MTP-GGUF \
  --include '*Q4_K_M*' \
  --local-dir "$MODEL_DIR/qwen3.6-27b-mtp"

# Vision projector (only if you want image input on the 27B):
# hf download unsloth/Qwen3.6-27B-MTP-GGUF --include "mmproj*" \
#   --local-dir "$MODEL_DIR/qwen3.6-27b-mtp"

echo "== vLLM path: AWQ/GPTQ-Int4 checkpoints (Marlin kernels, runs on sm_86) =="
# NOTE: verify the chosen quant repo retains the MTP head (mtp.* tensors) if you
# plan to use --speculative-config. Many community INT4 exports drop it, which
# makes the Qwen3_5MTP loader fail on a missing 'mtp.fc.weight'.
# Search current options:  hf search-repos "Qwen3.6-35B-A3B AWQ"
# Example (replace with the repo you validate):
# hf download <org>/Qwen3.6-35B-A3B-AWQ --local-dir "$MODEL_DIR/qwen3.6-35b-awq"

echo "Done. Models in $MODEL_DIR"
