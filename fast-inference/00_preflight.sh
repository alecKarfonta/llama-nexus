#!/usr/bin/env bash
# 00_preflight.sh — verify the box is ready for the MTP/AWQ fast-inference stack.
# Target: 4x RTX 3090 Ti (sm_86, PCIe-only TRX40), llama.cpp + vLLM.
set -uo pipefail

PASS="\033[32m[PASS]\033[0m"; FAIL="\033[31m[FAIL]\033[0m"; WARN="\033[33m[WARN]\033[0m"
ok=0; bad=0

check() { # check <label> <cmd...>
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then echo -e "$PASS $label"; ((ok++)); else echo -e "$FAIL $label"; ((bad++)); fi
}

echo "== GPUs =="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
  NGPU=$(nvidia-smi --list-gpus | wc -l)
  echo -e "$PASS $NGPU GPU(s) visible"
else
  echo -e "$FAIL nvidia-smi not found"; ((bad++))
fi

echo
echo "== llama.cpp (need build b9180+ for Qwen3.6 MTP, merged 2026-05-16) =="
LLAMA_BIN="${LLAMA_BIN:-llama-server}"
if command -v "$LLAMA_BIN" >/dev/null; then
  VER=$("$LLAMA_BIN" --version 2>&1 | grep -oE 'b[0-9]+' | head -1 | tr -d 'b')
  if [[ -n "${VER:-}" && "$VER" -ge 9180 ]]; then
    echo -e "$PASS llama-server build b$VER (>= b9180, MTP supported)"
  else
    echo -e "$FAIL llama-server build b${VER:-unknown} — rebuild from master for --spec-type draft-mtp"
    echo "       git pull && cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON && cmake --build build -j"
    ((bad++))
  fi
else
  echo -e "$WARN llama-server not on PATH (set LLAMA_BIN=/path/to/llama-server)"
fi

echo
echo "== vLLM (need >= 0.19.0 for Qwen3.6 + qwen3_next_mtp) =="
if python3 -c 'import vllm' 2>/dev/null; then
  VV=$(python3 -c 'import vllm; print(vllm.__version__)')
  python3 - "$VV" <<'EOF' && echo -e "\033[32m[PASS]\033[0m vLLM $VV" || { echo -e "\033[31m[FAIL]\033[0m vLLM $VV < 0.19.0 — pip install -U vllm"; }
import sys
from packaging.version import Version
sys.exit(0 if Version(sys.argv[1]) >= Version("0.19.0") else 1)
EOF
else
  echo -e "$WARN vLLM not installed in this python env (only needed for the vLLM path)"
fi

echo
echo "== Model files =="
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
echo "MODEL_DIR=$MODEL_DIR"
for f in Qwen3.6-35B-A3B-MTP Qwen3.6-27B-MTP; do
  if ls "$MODEL_DIR"/**/*"$f"* "$MODEL_DIR"/*"$f"* 2>/dev/null | head -1 | grep -q .; then
    echo -e "$PASS found $f GGUF"
  else
    echo -e "$WARN $f GGUF not found — run 01_download_models.sh"
  fi
done

echo
echo "== Notes for this rig =="
echo " * 3090 Ti = sm_86 (Ampere): no FP8 tensor cores. Use AWQ/GPTQ W4A16 in vLLM"
echo "   (Marlin kernels, sm_80+) and Q4_K_M/Q5_K_M GGUF in llama.cpp."
echo " * FP8 KV cache STORAGE (fp8_e5m2) works on Ampere in vLLM — compute stays fp16."
echo " * No NVLink on TRX40: prefer --split-mode layer (llama.cpp) and consider"
echo "   pipeline-parallel over TP=4 in vLLM if PCIe sync dominates."
echo " * IMPORTANT: standard Qwen3.6 GGUFs/INT4 quants often DROP the MTP head."
echo "   You must use *-MTP-GGUF repos for llama.cpp, and check the vLLM quant"
echo "   actually ships mtp.* weights before enabling speculative-config."

echo
echo "Summary: $ok pass, $bad fail"
exit $(( bad > 0 ? 1 : 0 ))
