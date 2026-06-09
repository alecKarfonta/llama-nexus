#!/usr/bin/env bash
# run_matrix.sh — automated A/B: baseline vs MTP on llama.cpp, then compare.
# Tunes NMAX while it's at it.
#
# Host (needs llama-server on PATH):
#   cd bench && ./run_matrix.sh
#
# Docker (this rig: 27B baseline on GPUs 0,1,3):
#   cd bench && MODEL_SIZE=27b DOCKER=1 GPU_DEVICES=0,1,3 PORT=8603 ./run_matrix.sh
#
# 35B MoE via Docker (single GPU 0):
#   cd bench && MODEL_SIZE=35b DOCKER=1 GPU_DEVICES=0 PORT=8604 ./run_matrix.sh
set -euo pipefail
cd "$(dirname "$0")"

MODEL_SIZE="${MODEL_SIZE:-35b}"
DOCKER="${DOCKER:-0}"
PORT="${PORT:-8080}"
GPU_DEVICES="${GPU_DEVICES:-0}"
CONTAINER="${CONTAINER:-fast-inference-matrix}"
RESULT_PREFIX="${RESULT_PREFIX:-}"
IMAGE="${IMAGE:-llama-nexus-llamacpp-embed:latest}"
CTX="${CTX:-}"
NP="${NP:-2}"
URL="http://localhost:$PORT"
BENCH_DOCKER="${BENCH_DOCKER:-auto}"   # auto | 0 | 1

if [[ "$MODEL_SIZE" == "27b" ]]; then
  SERVE=../11_llamacpp_qwen36_27b_mtp.sh
  MODEL_DIR="${MODEL_DIR:-$HOME/models}"
  MODEL="${MODEL:-$(ls "$MODEL_DIR"/qwen3.6-27b-mtp/*Q4_K_M*.gguf 2>/dev/null | head -1)}"
  ALIAS=qwen3.6-27b-mtp
  CTX="${CTX:-180000}"
  DEFAULT_NMAX=4
else
  SERVE=../10_llamacpp_qwen36_35b_mtp.sh
  MODEL_DIR="${MODEL_DIR:-$HOME/models}"
  MODEL="${MODEL:-$(ls "$MODEL_DIR"/qwen3.6-35b-mtp/*.gguf 2>/dev/null | head -1)}"
  ALIAS=qwen3.6-35b-mtp
  CTX="${CTX:-131072}"
  DEFAULT_NMAX=3
fi

[[ -n "${MODEL:-}" ]] || { echo "No GGUF for MODEL_SIZE=$MODEL_SIZE in $MODEL_DIR"; exit 1; }

bench_py() {
  local use_docker=0
  if [[ "$BENCH_DOCKER" == "1" ]]; then
    use_docker=1
  elif [[ "$BENCH_DOCKER" == "auto" ]]; then
    python3 -c 'import aiohttp' 2>/dev/null || use_docker=1
  fi
  if [[ "$use_docker" == 1 ]]; then
    docker run --rm --network host -v "$PWD:/bench" -w /bench python:3.12-slim bash -c \
      'pip install -q aiohttp && python3 bench.py "$@"' _ "$@"
  else
    ./bench.py "$@"
  fi
}

wait_ready() {
  for _ in $(seq 1 120); do
    curl -sf "$URL/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "server never became ready at $URL"; return 1
}

stop_docker() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

start_docker() { # start_docker <spec> <nmax> <pmin>
  local spec="$1" nmax="$2" pmin="$3"
  local model_rel="${MODEL#$MODEL_DIR/}"
  model_rel="${model_rel#/}"

  local spec_args=()
  if [[ "$spec" == "on" ]]; then
    spec_args=(--spec-type draft-mtp --spec-draft-n-max "$nmax" --spec-draft-p-min "$pmin")
  fi

  stop_docker
  docker run -d --name "$CONTAINER" \
    --gpus "\"device=${GPU_DEVICES}\"" \
    -v "$MODEL_DIR:/models:ro" \
    -p "$PORT:8080" \
    --shm-size=16gb \
    --entrypoint llama-server \
    "$IMAGE" \
    --model "/models/$model_rel" \
    --alias "$ALIAS" \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 --split-mode layer --main-gpu 0 \
    --ctx-size "$CTX" \
    --flash-attn on \
    -ctk q8_0 -ctv q8_0 \
    --cache-reuse 1024 \
    --parallel "$NP" \
    --metrics \
    --jinja --chat-template-kwargs '{"preserve_thinking": true}' \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
    "${spec_args[@]}"
}

run_case() { # run_case <label> <env...>
  local label="$1"; shift
  echo "=== $label (MODEL_SIZE=$MODEL_SIZE DOCKER=$DOCKER GPUs=$GPU_DEVICES) ==="

  if [[ "$DOCKER" == "1" ]]; then
    local spec=off nmax="${DEFAULT_NMAX}" pmin=0.75
    eval "$(printf '%s\n' "$@" | sed 's/^/export /')"
    spec="${SPEC:-off}"
    nmax="${NMAX:-$DEFAULT_NMAX}"
    pmin="${PMIN:-0.75}"
    start_docker "$spec" "$nmax" "$pmin"
    trap 'stop_docker' EXIT
    wait_ready
    bench_py --url "$URL" --label "${RESULT_PREFIX}${label}" --runs "${RUNS:-3}"
    stop_docker
    trap - EXIT
    sleep 5
  else
    env "$@" PORT="$PORT" "$SERVE" & local pid=$!
    trap "kill $pid 2>/dev/null || true" EXIT
    wait_ready
    bench_py --url "$URL" --label "${RESULT_PREFIX}${label}" --runs "${RUNS:-3}"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    trap - EXIT
    sleep 5
  fi
}

# Stop any prior experiment container on our port
if [[ "$DOCKER" == "1" ]]; then
  docker rm -f fast-inference-27b fast-inference-mtp "$CONTAINER" 2>/dev/null || true
fi

run_case baseline      SPEC=off
run_case mtp-n2        SPEC=on NMAX=2
run_case mtp-n3        SPEC=on NMAX=3
run_case mtp-n4        SPEC=on NMAX=4
run_case mtp-n4-p80    SPEC=on NMAX=4 PMIN=0.80

echo
bench_py --compare \
  "results/${RESULT_PREFIX}baseline.json" \
  "results/${RESULT_PREFIX}mtp-n2.json" \
  "results/${RESULT_PREFIX}mtp-n3.json" \
  "results/${RESULT_PREFIX}mtp-n4.json" \
  "results/${RESULT_PREFIX}mtp-n4-p80.json"
