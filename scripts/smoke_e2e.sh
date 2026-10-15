#!/usr/bin/env bash
# [2026-10-15] | INFO | SCRIPTS.SMOKE_E2E | End-to-end health validation of a
# running Llama Nexus stack. Exercises every service plane: backend health,
# registry catalog, model list, Qdrant lifecycle, embedding/TTS/STT status,
# and (when a model is deployed) a chat completion round-trip.
#
# Usage: scripts/smoke_e2e.sh [--api-key KEY] [--backend URL] [--full]
#   --full   also POST a chat completion against :8600 (requires deployed model)
# Exit codes: 0 pass, 1 failure (first failing stage).

set -euo pipefail

BACKEND="${BACKEND:-http://localhost:8700}"
LLM_API="${LLM_API:-http://localhost:8600}"
API_KEY="${API_KEY:-placeholder-api-key}"
FULL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-key) API_KEY="$2"; shift 2;;
    --backend) BACKEND="$2"; shift 2;;
    --full) FULL=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

PASS=0; FAIL=0
check() { # name, expected, actual_cmd
  local name="$1" expected="$2"; shift 2
  if eval "$@" >/dev/null 2>&1; then
    echo "  [ok] $name"; PASS=$((PASS+1))
  else
    echo "  [FAIL] $name (expected: $expected)"; FAIL=$((FAIL+1))
  fi
}

echo "== Llama Nexus e2e smoke =="
echo "   backend: $BACKEND  llm: $LLM_API"

echo "-- Core"
check "backend /health responds" "200" "curl -sf $BACKEND/health"
check "frontend reachable (3002)" "200" "curl -sf http://localhost:3002/health"

echo "-- Registry catalog (curated seed)"
check "registry recommendations non-empty" "json list" \
  "test \"\$(curl -sf \"$BACKEND/api/v1/registry/recommendations?vram_gb=8\" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get(\"recommendations\",[])))')\" -gt 0"

echo "-- Models"
check "model list endpoint" "200" "curl -sf $BACKEND/api/v1/models/"

echo "-- Qdrant (vector DB)"
check "qdrant status route" "200" "curl -sf $BACKEND/api/v1/qdrant/status"
check "qdrant container running" "running=true" \
  "curl -sf $BACKEND/api/v1/qdrant/status | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get(\"running\") else 1)'"

echo "-- Service status routes"
check "embedding status route" "200" "curl -sf $BACKEND/api/v1/embedding/status"
check "stt status route" "200" "curl -sf $BACKEND/api/v1/stt/status"
check "tts status route" "200" "curl -sf $BACKEND/api/v1/tts/status"
check "service status route" "200" "curl -sf $BACKEND/api/v1/service/status"

if [[ "$FULL" == "1" ]]; then
  echo "-- LLM round-trip (--full)"
  check "chat completion returns content" "choices[0]" "
    curl -sf $LLM_API/v1/chat/completions \
      -H 'Authorization: Bearer $API_KEY' -H 'Content-Type: application/json' \
      -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":8}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d[\"choices\"][0][\"message\"][\"content\"] else 1)'"
fi

echo "== Result: $PASS passed, $FAIL failed =="
[[ "$FAIL" -eq 0 ]]
