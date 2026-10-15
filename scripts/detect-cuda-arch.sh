#!/usr/bin/env bash
# [2026-10-15] | INFO | SCRIPTS.DETECT_CUDA_ARCH | Map host GPU compute capability
# to the CUDA_ARCH compose/build value (SM integer) used by the llama.cpp Dockerfile.
# Usage:  CUDA_ARCH=$(scripts/detect-cuda-arch.sh) docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
# Prints "none" (exit 1) when no NVIDIA GPU is present.

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "none" >&2
  exit 1
fi

CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)"

if [ -z "$CAPS" ]; then
  echo "none" >&2
  exit 1
fi

arch_for_cap() {
  case "$1" in
    7.5) echo 75 ;;   # Turing (T4, RTX 20xx)
    8.0) echo 80 ;;   # A100
    8.6) echo 86 ;;   # Ampere (RTX 30xx, A40/A6000)
    8.9) echo 89 ;;   # Ada (RTX 40xx, L40S)
    9.0) echo 90 ;;   # Hopper (H100)
    12.0) echo 120 ;; # Blackwell (RTX 50xx)
    *) echo "$1" | tr -d '.' ;; # best-effort future caps
  esac
}

# Pick the highest capability present; multi-GPU mixed rigs build for the max
# (llama.cpp CUDA kernels for lower caps are covered by PTX fallback at runtime
# only when built with the right arch list — for mixed rigs set CUDA_ARCH="a;b" manually).
HIGHEST="$(echo "$CAPS" | sort -V | tail -1)"
ARCH="$(arch_for_cap "$HIGHEST")"

echo "$ARCH"
