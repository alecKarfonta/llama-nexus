#!/usr/bin/env bash
# [TIME] | INFO | SCRIPTS.QWEN_MTP | flip MTP speculative decoding on/off for the
# running Qwen3.8-Flash-Next deployment (~90s model reload per flip).
#
# Measured on this rig (4x 3090 Ti, UD-Q4_K_XL, draft=Q4_K_M sidecar):
#   MTP ON : structured/code/JSON +49% (58-70 tok/s), prose -27% (34 tok/s)
#   MTP OFF: everything ~47 tok/s
# => ON for Cursor/agent/coding traffic, OFF for creative writing/chat.
#
# Usage: ./scripts/qwen_mtp.sh on|off|status
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8700}"
API_KEY="${API_KEY:-placeholder-api-key}"
DRAFT="${MTP_DRAFT_MODEL:-Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf}"

mode="${1:-status}"

get_state() {
    curl -s "${BACKEND_URL}/api/v1/service/config" \
        -H "Authorization: Bearer ${API_KEY}" | python3 -c '
import json, sys
c = json.load(sys.stdin).get("config", {})
mtp = c.get("mtp", {})
spec = c.get("speculative", {})
print("enabled" if mtp.get("enabled") else "disabled", "|", spec.get("model_draft") or "no-draft")'
}

case "${mode}" in
    status)
        echo "MTP: $(get_state)"
        ;;
    on)
        echo "-- enabling MTP (draft: ${DRAFT})"
        curl -s -X PUT "${BACKEND_URL}/api/v1/service/config" \
            -H "Authorization: Bearer ${API_KEY}" -H 'Content-Type: application/json' \
            -d "{
                \"mtp\": {\"enabled\": true, \"draft_n_max\": 3, \"draft_n_min\": 0, \"draft_p_min\": 0.75},
                \"speculative\": {\"model_draft\": \"${DRAFT}\", \"gpu_layers_draft\": 999}
            }" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("config:", d.get("success"))'
        curl -s -X POST "${BACKEND_URL}/api/v1/service/restart" \
            -H "Authorization: Bearer ${API_KEY}" -o /dev/null -w 'restart: %{http_code}\n'
        echo "-- reloading (~90s); watch: docker logs -f llamacpp-api"
        ;;
    off)
        echo "-- disabling MTP"
        curl -s -X PUT "${BACKEND_URL}/api/v1/service/config" \
            -H "Authorization: Bearer ${API_KEY}" -H 'Content-Type: application/json' \
            -d "{
                \"mtp\": {\"enabled\": false},
                \"speculative\": {\"model_draft\": null}
            }" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("config:", d.get("success"))'
        curl -s -X POST "${BACKEND_URL}/api/v1/service/restart" \
            -H "Authorization: Bearer ${API_KEY}" -o /dev/null -w 'restart: %{http_code}\n'
        echo "-- reloading (~90s); watch: docker logs -f llamacpp-api"
        ;;
    *)
        echo "usage: $0 on|off|status" >&2
        exit 1
        ;;
esac
