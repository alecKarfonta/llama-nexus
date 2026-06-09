#!/usr/bin/env bash
# run_extras.sh — remaining fast-inference experiments (beyond NMAX sweep).
#
#   cd bench && ./run_extras.sh                    # all extras
#   cd bench && EXPERIMENT=gpu2-gpu3 ./run_extras.sh
#
# Experiments: gpu2-gpu3, kv-cache, spec-ngram, concurrency
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8603}"
GPU2="${GPU2:-0,1}"
GPU3="${GPU3:-0,1,3}"
CONTAINER="${CONTAINER:-fast-inference-extras}"
IMAGE="${IMAGE:-llama-nexus-llamacpp-embed:latest}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-$MODEL_DIR/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf}"
CTX="${CTX:-131072}"
URL="http://localhost:$PORT"
EXPERIMENT="${EXPERIMENT:-all}"

bench_py() {
  if python3 -c 'import aiohttp' 2>/dev/null; then
    ./bench.py "$@"
  else
    docker run --rm --network host -v "$PWD:/bench" -w /bench python:3.12-slim bash -c \
      'pip install -q aiohttp && python3 bench.py "$@"' _ "$@"
  fi
}

stop_srv() { docker rm -f "$CONTAINER" 2>/dev/null || true; }

start_srv() { # start_srv <label> <gpus> <extra llama-server args...>
  local label="$1" gpus="$2"; shift 2
  local model_rel="${MODEL#$MODEL_DIR/}"; model_rel="${model_rel#/}"
  stop_srv
  echo ">>> start $label gpus=$gpus"
  docker run -d --name "$CONTAINER" \
    --gpus "\"device=${gpus}\"" \
    -v "$MODEL_DIR:/models:ro" \
    -p "$PORT:8080" \
    --shm-size=16gb \
    --entrypoint llama-server \
    "$IMAGE" \
    --model "/models/$model_rel" \
    --alias qwen3.6-27b-mtp \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 --split-mode layer --main-gpu 0 \
    --ctx-size "$CTX" \
    --flash-attn on \
    --cache-reuse 1024 \
    --parallel "${PARALLEL:-1}" \
    --metrics \
    --jinja --chat-template-kwargs '{"preserve_thinking": true}' \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
    "$@"
  for _ in $(seq 1 90); do
    curl -sf "$URL/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$CONTAINER" | grep -q . || { echo "container died"; docker logs "$CONTAINER" 2>&1 | tail -5; return 1; }
    sleep 2
  done
  echo "timeout waiting for $URL"; return 1
}

run_bench() { # run_bench <label> <gpus> <args...>
  local label="$1" gpus="$2"; shift 2
  start_srv "$label" "$gpus" "$@"
  bench_py --url "$URL" --label "$label" --runs "${RUNS:-3}" --concurrency "${CONCURRENCY:-1}"
  stop_srv
  sleep 3
}

MTP_ARGS=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75)
NGRAM_ARGS=(--spec-type ngram-simple --spec-ngram-simple-size-n 12 --spec-ngram-simple-size-m 4)

run_gpu_compare() {
  echo "=== 2-GPU vs 3-GPU (MTP n2, q8_0 KV) ==="
  run_bench extras-gpu2-mtp "$GPU2" -ctk q8_0 -ctv q8_0 "${MTP_ARGS[@]}"
  run_bench extras-gpu3-mtp "$GPU3" -ctk q8_0 -ctv q8_0 "${MTP_ARGS[@]}"
}

run_kv_compare() {
  echo "=== KV cache q8_0 vs f16 (2-GPU, MTP n2) ==="
  run_bench extras-kv-q8 "$GPU2" -ctk q8_0 -ctv q8_0 "${MTP_ARGS[@]}"
  run_bench extras-kv-f16 "$GPU2" -ctk f16 -ctv f16 "${MTP_ARGS[@]}"
}

run_spec_compare() {
  echo "=== MTP vs ngram-simple vs off (2-GPU, q8_0) ==="
  run_bench extras-spec-mtp "$GPU2" -ctk q8_0 -ctv q8_0 "${MTP_ARGS[@]}"
  run_bench extras-spec-ngram "$GPU2" -ctk q8_0 -ctv q8_0 "${NGRAM_ARGS[@]}"
  run_bench extras-spec-off "$GPU2" -ctk q8_0 -ctv q8_0
}

run_concurrency() {
  echo "=== MTP concurrency 1 vs 4 (2-GPU) ==="
  CONCURRENCY=1 run_bench extras-conc1 "$GPU2" -ctk q8_0 -ctv q8_0 "${MTP_ARGS[@]}"
  CONCURRENCY=4 run_bench extras-conc4 "$GPU2" -ctk q8_0 -ctv q8_0 "${MTP_ARGS[@]}"
}

docker rm -f fast-inference-27b "$CONTAINER" 2>/dev/null || true

case "$EXPERIMENT" in
  all)
    run_gpu_compare
    run_kv_compare
    run_spec_compare
    run_concurrency
    bench_py --compare results/extras-gpu2-mtp.json results/extras-gpu3-mtp.json
    echo; bench_py --compare results/extras-kv-q8.json results/extras-kv-f16.json
    echo; bench_py --compare results/extras-spec-off.json results/extras-spec-ngram.json results/extras-spec-mtp.json
    echo; bench_py --compare results/extras-conc1.json results/extras-conc4.json
    ;;
  gpu2-gpu3) run_gpu_compare; bench_py --compare results/extras-gpu2-mtp.json results/extras-gpu3-mtp.json ;;
  kv-cache)  run_kv_compare; bench_py --compare results/extras-kv-q8.json results/extras-kv-f16.json ;;
  spec-ngram) run_spec_compare; bench_py --compare results/extras-spec-off.json results/extras-spec-ngram.json results/extras-spec-mtp.json ;;
  concurrency) run_concurrency; bench_py --compare results/extras-conc1.json results/extras-conc4.json ;;
  *) echo "Unknown EXPERIMENT=$EXPERIMENT"; exit 1 ;;
esac

echo "Done. Results in results/extras-*.json"
