#!/usr/bin/env bash
# [TIME] | INFO | SCRIPTS.QWEN_SET | push a partial JSON config to the backend and
# restart llamacpp, then wait for /health. Helper for TPS experiments.
# Usage: ./scripts/qwen_set.sh '<json-partial>' [timeout_s]
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8700}"
API_KEY="${API_KEY:-placeholder-api-key}"
TIMEOUT="${2:-240}"

curl -s -X PUT "${BACKEND_URL}/api/v1/service/config" \
    -H "Authorization: Bearer ${API_KEY}" -H 'Content-Type: application/json' \
    -d "$1" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("config:", d.get("success")) if d.get("success") is not None else print("ERROR:", d)'
curl -s -X POST "${BACKEND_URL}/api/v1/service/restart" \
    -H "Authorization: Bearer ${API_KEY}" -o /dev/null -w 'restart: %{http_code}\n'

# resolve the live llama api key for health checks
LKEY=$(curl -s "${BACKEND_URL}/api/v1/service/config" -H "Authorization: Bearer ${API_KEY}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["config"]["server"]["api_key"])')

for i in $(seq 1 $((TIMEOUT / 10))); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 -H "Authorization: Bearer ${LKEY}" http://127.0.0.1:8600/health 2>/dev/null || true)
    if [ "$code" = "200" ]; then
        echo "healthy after ~$((i*10))s"
        exit 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q '^llamacpp-api$'; then
        echo "CONTAINER DIED"; docker logs llamacpp-api 2>&1 | grep -iE 'error|assert|oom' | tail -5; exit 1
    fi
    sleep 10
done
echo "TIMEOUT waiting for health"; exit 1
