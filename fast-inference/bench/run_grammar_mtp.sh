#!/usr/bin/env bash
# run_grammar_mtp.sh — grammar (tools API) + MTP acceptance experiment.
#
#   cd bench && ./run_grammar_mtp.sh
#   EXPERIMENT=mtp-only ./run_grammar_mtp.sh   # skip no-tools baseline
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8603}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
CONTAINER="${CONTAINER:-grammar-mtp-exp}"
IMAGE="${IMAGE:-llama-nexus-llamacpp-embed:latest}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-$MODEL_DIR/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf}"
CTX="${CTX:-131072}"
URL="http://localhost:$PORT"
EXPERIMENT="${EXPERIMENT:-all}"
LOG_DIR="${LOG_DIR:-/tmp/grammar-mtp-logs}"
mkdir -p "$LOG_DIR"

bench_tools() {
  if python3 -c 'import aiohttp' 2>/dev/null; then
    python3 bench_tools.py "$@"
  else
    docker run --rm --network host -v "$PWD:/bench" -w /bench python:3.12-slim bash -c \
      'pip install -q aiohttp && python3 bench_tools.py "$@"' _ "$@"
  fi
}

stop_srv() { docker rm -f "$CONTAINER" grammar-mtp-test 2>/dev/null || true; }

start_srv() { # start_srv <spec> <nmax>
  local spec="$1" nmax="$2"
  local model_rel="${MODEL#$MODEL_DIR/}"; model_rel="${model_rel#/}"
  local spec_args=()
  if [[ "$spec" == "on" ]]; then
    spec_args=(--spec-type draft-mtp --spec-draft-n-max "$nmax" --spec-draft-p-min 0.75)
  fi
  stop_srv
  docker run -d --name "$CONTAINER" \
    --gpus "\"device=${GPU_DEVICES}\"" \
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
    -ctk q8_0 -ctv q8_0 \
    --cache-reuse 1024 \
    --parallel 1 \
    --metrics \
    --jinja --chat-template-kwargs '{"enable_thinking": false}' \
    --temp 0.2 --top-p 0.95 --top-k 20 --min-p 0.0 \
    "${spec_args[@]}"
  for _ in $(seq 1 90); do
    curl -sf "$URL/health" >/dev/null 2>&1 && return 0
    docker ps -q -f name="$CONTAINER" | grep -q . || {
      docker logs "$CONTAINER" 2>&1 | tail -8
      return 1
    }
    sleep 2
  done
  return 1
}

run_case() { # run_case <label> <spec> <nmax> <use_tools>
  local label="$1" spec="$2" nmax="$3" use_tools="$4"
  echo "=== $label (spec=$spec nmax=$nmax tools=$use_tools) ==="
  start_srv "$spec" "$nmax"
  # JIT / slot warmup after load
  curl -sf "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6-27b-mtp","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":false}' \
    >/dev/null 2>&1 || true
  sleep 3
  local logfile="$LOG_DIR/${label}.log"
  local args=(--url "$URL" --label "$label" --runs "${RUNS:-3}")
  if [[ "$use_tools" == "0" ]]; then
    args+=(--no-tools)
  fi
  bench_tools "${args[@]}"
  docker logs "$CONTAINER" >"$logfile" 2>&1
  python3 - "$logfile" "$label" <<'PY' 2>/dev/null || true
import json, re, sys
from pathlib import Path
log, label = sys.argv[1], sys.argv[2]
text = Path(log).read_text()
m = list(re.finditer(r"draft\s+acceptance\s+rate\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\s*\)", text, re.I))
path = Path("results") / f"{label}.json"
if path.exists() and m:
    data = json.loads(path.read_text())
    last = m[-1]
    data["mtp_stats"] = {"acceptance_rate": float(last.group(1)), "tokens_accepted": int(last.group(2)), "tokens_generated": int(last.group(3)), "samples": len(m)}
    path.write_text(json.dumps(data, indent=2))
    print(f"MTP {label}: {data['mtp_stats']['acceptance_rate']:.1%}")
PY
  stop_srv
  sleep 3
}

stop_srv

if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "baseline" ]]; then
  run_case tools-on-mtp-off  off 0 1
fi
if [[ "$EXPERIMENT" == "all" ]]; then
  : # skip tools-off free-chat baseline (not tool-call path)
fi
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "mtp" ]]; then
  run_case tools-on-mtp-n2  on 2 1
  run_case tools-on-mtp-n4  on 4 1
  run_case tools-on-mtp-n6  on 6 1
  run_case tools-on-mtp-n8  on 8 1
fi

echo
bench_tools --compare \
  results/tools-off-mtp-off.json \
  results/tools-on-mtp-off.json \
  results/tools-on-mtp-n2.json \
  results/tools-on-mtp-n4.json \
  results/tools-on-mtp-n6.json \
  results/tools-on-mtp-n8.json 2>/dev/null || \
  bench_tools --compare results/tools-on-*.json

echo "Logs: $LOG_DIR/  Results: results/tools-*.json"
