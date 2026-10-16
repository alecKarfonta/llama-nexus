#!/usr/bin/env bash
# [2026-10-15] | INFO | SCRIPTS.SMOKE_E2E | End-to-end health validation of a
# running Llama Nexus stack. Exercises every service plane: backend health,
# registry catalog, model list, Qdrant lifecycle, embedding/TTS/STT status,
# and (when a model is deployed) a chat completion round-trip.
# [2026-10-16] | INFO | SCRIPTS.SMOKE_E2E | Milestone 3.2 — wizard-flow
# expansion under --full: optional small-model download + LLM deploy (skipped
# when a model is already deployed/healthy), chat completion, STT transcribe,
# TTS synthesize, embedding round-trip, and Qdrant-backed RAG search. New
# --mgmt-token flag attaches X-Management-Token to every management call for
# stacks running with MANAGEMENT_AUTH_TOKEN set.
#
# Usage: scripts/smoke_e2e.sh [--api-key KEY] [--backend URL] [--mgmt-token TOKEN] [--full]
#   --full   full wizard flow: (download+deploy if no model running) ->
#            chat -> STT -> TTS -> embed -> RAG
# Env overrides (in addition to BACKEND / LLM_API / API_KEY / MGMT_TOKEN):
#   SMALL_MODEL_REPO (Qwen/Qwen3-0.6B-GGUF), SMALL_MODEL_FILE
#   (Qwen3-0.6B-Q4_K_M.gguf, ~400MB), DOWNLOAD_TIMEOUT (600s),
#   DEPLOY_TIMEOUT (180s), SVC_WAIT_TIMEOUT (180s), POLL_INTERVAL (5s),
#   TOTAL_TIMEOUT (1800s)
# Exit codes: 0 pass, 1 failure (first failing stage).

set -euo pipefail

BACKEND="${BACKEND:-http://localhost:8700}"
LLM_API="${LLM_API:-http://localhost:8600}"
API_KEY="${API_KEY:-placeholder-api-key}"
FULL=0
MGMT_TOKEN="${MGMT_TOKEN:-}"

# [2026-10-16] | INFO | SCRIPTS.SMOKE_E2E | tiny default download target
SMALL_MODEL_REPO="${SMALL_MODEL_REPO:-Qwen/Qwen3-0.6B-GGUF}"
# [2026-10-16] | FIX | SMOKE_E2E.SMALL_MODEL | upstream repo only publishes Q8_0
# for Qwen3-0.6B (Q4_K_M 404s); ~640MB is still the smallest curated chat model
SMALL_MODEL_FILE="${SMALL_MODEL_FILE:-Qwen3-0.6B-Q8_0.gguf}"
DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-600}"
DEPLOY_TIMEOUT="${DEPLOY_TIMEOUT:-180}"
SVC_WAIT_TIMEOUT="${SVC_WAIT_TIMEOUT:-180}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
TOTAL_TIMEOUT="${TOTAL_TIMEOUT:-1800}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-key) API_KEY="$2"; shift 2;;
    --backend) BACKEND="$2"; shift 2;;
    --mgmt-token) MGMT_TOKEN="$2"; shift 2;;
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

# [2026-10-16] | INFO | SCRIPTS.SMOKE_E2E | management curl wrapper: curl -sf
# plus X-Management-Token header when --mgmt-token (or MGMT_TOKEN) is set
mcurl() {
  if [[ -n "$MGMT_TOKEN" ]]; then
    curl -sf -m 120 -H "X-Management-Token: $MGMT_TOKEN" "$@"
  else
    curl -sf -m 120 "$@"
  fi
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
STT_WAV="$TMP/smoke_stt.wav"
TTS_OUT="$TMP/smoke_tts.bin"

# [2026-10-16] | FIX | SMOKE_E2E.API_KEY_DISCOVERY | must run AFTER mcurl() is
# defined (bash functions are call-time resolved); the deployed llamacpp
# container uses whatever api_key the manager config holds (persisted across
# restarts) — not necessarily compose .env. Unless the caller forced --api-key
# or API_KEY env, discover the live key so the chat round-trip works on any
# stack (CI fresh stacks and long-lived dev rigs alike).
if [[ "${API_KEY}" == "placeholder-api-key" ]]; then
  DISCOVERED=$(mcurl "$BACKEND/api/v1/service/config" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); k=(d.get("config",d) or {}).get("server",{}).get("api_key",""); print(k if k else "")' 2>/dev/null || true)
  if [[ -n "$DISCOVERED" ]]; then
    API_KEY="$DISCOVERED"
    echo "   (api key auto-discovered from service config)"
  fi
fi

# ---- python3 stdlib JSON probes (same pattern as the core checks) ----
RUN_PY='import json,sys; sys.exit(0 if json.load(sys.stdin).get("running") else 1)'
HEALTH_OK_PY='import json,sys
d=json.load(sys.stdin)
h=d.get("llamacpp_health") or {}
sys.exit(0 if (d.get("running") and h.get("healthy")) else 1)'
SVC_HEALTH_PY='import json,sys
d=json.load(sys.stdin)
h=d.get("llamacpp_health") or {}
m=d.get("model") or {}
print("running={} healthy={} model={}/{}".format(d.get("running"), h.get("healthy"), m.get("name"), m.get("variant")))
sys.exit(0 if (d.get("running") and h.get("healthy")) else 1)'
FILE_PY='import json,sys
d=json.load(sys.stdin)
files=(d.get("data") or {}).get("files") or []
sys.exit(0 if any(f.get("name")==sys.argv[1] for f in files) else 1)'
DL_STATE_PY='import json,sys
d=json.load(sys.stdin)
recs=(d.get("data") if isinstance(d,dict) else d) or []
r=next((x for x in recs if x.get("filename")==sys.argv[1]),None)
print(r.get("status") if r else "none")'
DL_POLL_PY='import json,sys
d=json.load(sys.stdin)
recs=(d.get("data") if isinstance(d,dict) else d) or []
r=next((x for x in recs if x.get("filename")==sys.argv[1]),None)
if not r:
    print("waiting for download record"); sys.exit(1)
print("{} {:0.1f}%".format(r.get("status"), float(r.get("progress") or 0)))
sys.exit(0 if r.get("status")=="completed" else 1)'

svc_running() { mcurl "$BACKEND/api/v1/$1/status" | python3 -c "$RUN_PY" >/dev/null 2>&1; }
svc_model_healthy() { mcurl "$BACKEND/api/v1/service/status" | python3 -c "$HEALTH_OK_PY" >/dev/null 2>&1; }
model_file_present() { mcurl "$BACKEND/v1/models/local-files" | python3 -c "$FILE_PY" "$SMALL_MODEL_FILE" >/dev/null 2>&1; }
dl_record_state() { mcurl "$BACKEND/v1/models/downloads" | python3 -c "$DL_STATE_PY" "$SMALL_MODEL_FILE" 2>/dev/null || true; }

wait_svc_running() { # svc, wait_secs — poll /status until running==true
  # NOTE: split local declarations — $((...)) in the same 'local' line as the
  # variable it reads expands before assignment (set -u kills it otherwise)
  local svc="$1" wait_secs="$2"
  local deadline=$((SECONDS + wait_secs))
  while :; do
    svc_running "$svc" && return 0
    if (( SECONDS >= deadline )); then return 1; fi
    sleep "$POLL_INTERVAL"
  done
}

wait_download() { # poll /v1/models/downloads until our file status==completed
  local out
  local deadline=$((SECONDS + DOWNLOAD_TIMEOUT))
  while :; do
    out=$(mcurl "$BACKEND/v1/models/downloads" | python3 -c "$DL_POLL_PY" "$SMALL_MODEL_FILE" 2>/dev/null || true)
    echo "       download: ${out:-waiting for record}"
    case "$out" in
      completed*) return 0;;
      failed*|cancelled*) return 1;;
    esac
    if (( SECONDS >= deadline )); then echo "       download: timed out after ${DOWNLOAD_TIMEOUT}s"; return 1; fi
    sleep "$POLL_INTERVAL"
  done
}

wait_service_healthy() { # poll /api/v1/service/status until running+healthy
  local out
  local deadline=$((SECONDS + DEPLOY_TIMEOUT))
  while :; do
    out=$(mcurl "$BACKEND/api/v1/service/status" | python3 -c "$SVC_HEALTH_PY" 2>/dev/null || true)
    echo "       service: ${out:-status unavailable}"
    [[ "$out" == "running=True healthy=True"* ]] && return 0
    if (( SECONDS >= deadline )); then echo "       service: timed out after ${DEPLOY_TIMEOUT}s"; return 1; fi
    sleep "$POLL_INTERVAL"
  done
}

ensure_service() { # svc — start container if idle, then wait for running==true
  local svc="$1"
  if svc_running "$svc"; then
    echo "  [ok] $svc service already running"; PASS=$((PASS+1)); return 0
  fi
  echo "       $svc not running — starting (wait <=${SVC_WAIT_TIMEOUT}s)"
  check "$svc start accepted" "success=true" "mcurl -X POST $BACKEND/api/v1/$svc/start"
  check "$svc service running (<=${SVC_WAIT_TIMEOUT}s)" "running=true" "wait_svc_running $svc $SVC_WAIT_TIMEOUT"
}

make_wav() { # [2026-10-16] | INFO | SCRIPTS.SMOKE_E2E | 0.5s 16kHz mono soft tone fixture
  python3 - "$STT_WAV" <<'PY'
import math, struct, sys, wave
rate, dur, freq = 16000, 0.5, 440
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    frames = bytearray()
    for i in range(int(rate * dur)):
        t = i / rate
        env = math.sin(math.pi * t / dur)  # soft attack/release envelope
        val = int(6000 * env * math.sin(2 * math.pi * freq * t))
        frames += struct.pack("<h", max(-32768, min(32767, val)))
    w.writeframes(bytes(frames))
PY
}

stt_transcribe() { # pass = HTTP 200; whisper may return empty text on a tone
  if mcurl -X POST "$BACKEND/api/v1/stt/transcribe" \
       -F "file=@$STT_WAV" -F "model=base" -F "language=auto" > "$TMP/stt.json"; then
    local text
    text=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("text") or "").strip())' < "$TMP/stt.json" 2>/dev/null || true)
    echo "       transcribed: ${text:-<empty — ok for tone fixture>}"
    return 0
  fi
  echo "       stt transcribe failed (non-200 or connection error)"
  return 1
}

tts_curl() { # [2026-10-16] | INFO | SCRIPTS.SMOKE_E2E | curl without -f so we can inspect status + content-type
  if [[ -n "$MGMT_TOKEN" ]]; then
    curl -s -m 150 -H "X-Management-Token: $MGMT_TOKEN" "$@"
  else
    curl -s -m 150 "$@"
  fi
}

tts_synthesize() { # pass = HTTP 200 + audio/* content-type + non-empty body
  local meta bytes
  meta=$(tts_curl -o "$TTS_OUT" -w '%{http_code} %{content_type}' \
    -X POST "$BACKEND/api/v1/tts/synthesize" \
    -H 'Content-Type: application/json' \
    -d '{"input":"Hello from the llama-nexus smoke test"}')
  bytes=$(wc -c < "$TTS_OUT" 2>/dev/null || echo 0)
  echo "       tts response: ${meta} (${bytes} bytes saved)"
  [[ "$meta" == "200 audio/"* ]] && [[ "$bytes" -gt 0 ]]
}

guard() { # stage label — TOTAL_TIMEOUT enforcement across --full stages
  if (( SECONDS >= TOTAL_TIMEOUT )); then
    echo "  [FAIL] total timeout (${TOTAL_TIMEOUT}s) exceeded before stage: $1"
    FAIL=$((FAIL+1)); return 1
  fi
  return 0
}

echo "== Llama Nexus e2e smoke =="
echo "   backend: $BACKEND  llm: $LLM_API"

echo "-- Core"
check "backend /health responds" "200" "mcurl $BACKEND/health"
check "frontend reachable (3002)" "200" "curl -sf http://localhost:3002/health"

echo "-- Registry catalog (curated seed)"
check "registry recommendations non-empty" "json list" \
  "test \"\$(mcurl \"$BACKEND/api/v1/registry/recommendations?vram_gb=8\" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get(\"recommendations\",[])))')\" -gt 0"

echo "-- Models"
# NOTE: models router is OpenAI-style /v1/models (no /api prefix)
check "model list endpoint" "200" "mcurl $BACKEND/v1/models"

echo "-- Qdrant (vector DB)"
check "qdrant status route" "200" "mcurl $BACKEND/api/v1/qdrant/status"
check "qdrant container running" "running=true" \
  "mcurl $BACKEND/api/v1/qdrant/status | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get(\"running\") else 1)'"

echo "-- Service status routes"
check "embedding status route" "200" "mcurl $BACKEND/api/v1/embedding/status"
check "stt status route" "200" "mcurl $BACKEND/api/v1/stt/status"
check "tts status route" "200" "mcurl $BACKEND/api/v1/tts/status"
check "service status route" "200" "mcurl $BACKEND/api/v1/service/status"

if [[ "$FULL" == "1" ]]; then
  echo "-- Model deploy (wizard flow, --full)"
  MODEL_BASE="${SMALL_MODEL_FILE%.gguf}"
  DEPLOY_NAME="${MODEL_BASE%-*}"     # 'Qwen3-0.6B-Q4_K_M' -> 'Qwen3-0.6B' (split on last '-')
  DEPLOY_VARIANT="${MODEL_BASE##*-}" # 'Qwen3-0.6B-Q4_K_M' -> 'Q4_K_M'
  echo "   target: $SMALL_MODEL_REPO/$SMALL_MODEL_FILE (name='$DEPLOY_NAME' variant='$DEPLOY_VARIANT')"
  if guard "model download/deploy"; then
    if svc_model_healthy; then
      echo "  [ok] model already deployed and healthy — skipping download/deploy"; PASS=$((PASS+1))
    else
      local_model_ready() { model_file_present || [[ "$(dl_record_state)" == "completed" ]]; }
      if local_model_ready; then
        echo "  [ok] small model already on disk — skipping download"; PASS=$((PASS+1))
      else
        check "download request accepted (or already in progress)" "success=true" \
          "mcurl -X POST $BACKEND/v1/models/download -H 'Content-Type: application/json' -d '{\"repositoryId\":\"$SMALL_MODEL_REPO\",\"filename\":\"$SMALL_MODEL_FILE\"}' || { [[ \"\$(dl_record_state)\" == queued ]] || [[ \"\$(dl_record_state)\" == downloading ]]; }"
        check "small model download completes (<=${DOWNLOAD_TIMEOUT}s)" "status=completed" "wait_download"
      fi
      # [2026-10-16] | INFO | SCRIPTS.SMOKE_E2E | only attempt deploy when the
      # gguf actually landed — avoids a doomed DEPLOY_TIMEOUT health-wait
      if local_model_ready; then
        check "deploy accepted (service/action start)" "success=true" \
          "mcurl -X POST $BACKEND/v1/service/action -H 'Content-Type: application/json' -d '{\"action\":\"start\",\"backend\":\"llamacpp\",\"config\":{\"model\":{\"name\":\"$DEPLOY_NAME\",\"variant\":\"$DEPLOY_VARIANT\"}}}}'"
        check "service healthy with model loaded (<=${DEPLOY_TIMEOUT}s)" "running=true healthy=true" "wait_service_healthy"
      else
        echo "  [FAIL] deploy skipped — '$SMALL_MODEL_FILE' not available after download stage"
        FAIL=$((FAIL+1))
      fi
    fi
  fi

  echo "-- LLM round-trip (--full)"
  if guard "chat completion"; then
    # [2026-10-16] | FIX | SMOKE_E2E.CHAT_HEADER | single-quoted header freezes
    # $API_KEY literal under eval; must be double quotes so the discovered key
    # actually reaches the server
    check "chat completion returns content" "choices[0]" "
      curl -sf -m 120 $LLM_API/v1/chat/completions \
        -H \"Authorization: Bearer $API_KEY\" -H 'Content-Type: application/json' \
        -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":8}' \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d[\"choices\"][0][\"message\"][\"content\"] else 1)'"
  fi

  echo "-- Speech-to-text (--full)"
  if guard "STT transcribe"; then
    check "wav fixture generated (0.5s 16kHz mono tone)" "non-empty file" "make_wav && test -s '$STT_WAV'"
    ensure_service stt
    check "stt transcribe returns 200 (text optional)" "HTTP 200" "stt_transcribe"
  fi

  echo "-- Text-to-speech (--full)"
  if guard "TTS synthesize"; then
    ensure_service tts
    check "tts synthesize returns audio (200 + audio/*)" "audio bytes" "tts_synthesize"
  fi

  echo "-- Embeddings (--full)"
  if guard "embedding round-trip"; then
    ensure_service embedding
    check "embedding round-trip returns non-empty vector" "vectorLength>0" \
      "mcurl -X POST $BACKEND/api/v1/embedding/test -H 'Content-Type: application/json' -d '{\"text\":\"hello from the smoke test\"}' | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get(\"vectorLength\",0)>0 else 1)'"
  fi

  echo "-- RAG over Qdrant (--full)"
  if guard "RAG search"; then
    check "rag search responds (hits optional)" "HTTP 200" \
      "mcurl -X POST $BACKEND/api/v1/rag/search -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"top_k\":3}'"
  fi
fi

echo "== Result: $PASS passed, $FAIL failed =="
[[ "$FAIL" -eq 0 ]]
