#!/usr/bin/env bash
set -euo pipefail

echo "=== NInfer startup (env-driven; matches Deploy / NinferManager config) ==="

MODELS_DIR="${NINFER_MODELS_DIR:-/models}"
NINFER_ARTIFACT="${NINFER_ARTIFACT:-qwen3_8_27b_nvfp4.ninfer}"
NINFER_SERVED_MODEL_NAME="${NINFER_SERVED_MODEL_NAME:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-placeholder-api-key}"

NINFER_MAX_CONTEXT="${NINFER_MAX_CONTEXT:-32768}"
NINFER_KV_CAPACITY="${NINFER_KV_CAPACITY:-auto}"
# [2026-08-30 | INFO | start-ninfer.sh.MAIN | int8 group-64 wins the Phase-1 A/B
# (geomean 182.1 vs 179.9 bf16; +4.6% on code). Matches published ninfer benches.
NINFER_KV_DTYPE="${NINFER_KV_DTYPE:-int8}"
NINFER_MAX_CONCURRENCY="${NINFER_MAX_CONCURRENCY:-2}"

NINFER_SPEC_ENABLED="${NINFER_SPEC_ENABLED:-true}"
NINFER_SPEC_METHOD="${NINFER_SPEC_METHOD:-mtp}"
# [2026-08-28 | INFO | start-ninfer.sh.MAIN | 4 is the empirically-measured sweet
# spot for Qwen3.8-27B MTP on this hardware. Depth-5 costs more verify than the
# extra accepted token is worth (accept prob decays: 75% -> 64% -> 53% -> 45% -> 37%).
NINFER_DRAFT_TOKENS="${NINFER_DRAFT_TOKENS:-4}"
NINFER_LM_HEAD_DRAFT="${NINFER_LM_HEAD_DRAFT:-true}"

NINFER_VISION="${NINFER_VISION:-true}"

# [2026-08-30 | INFO | start-ninfer.sh.MAIN | Process-level sampling defaults from the
# ninfer published campaign. Per-request fields still override these.
NINFER_TEMPERATURE="${NINFER_TEMPERATURE:-0.6}"
NINFER_TOP_P="${NINFER_TOP_P:-0.95}"
NINFER_TOP_K="${NINFER_TOP_K:-20}"
NINFER_PRESENCE_PENALTY="${NINFER_PRESENCE_PENALTY:-1.0}"
NINFER_GREEDY="${NINFER_GREEDY:-false}"
# [2026-08-31 | INFO | start-ninfer.sh.MAIN | Thinking-off is the single-user chat
# profile (bench + UI). --no-thinking makes that the process default.
NINFER_NO_THINKING="${NINFER_NO_THINKING:-true}"

ARTIFACT_PATH="$MODELS_DIR/$NINFER_ARTIFACT"

echo "Artifact:       $ARTIFACT_PATH"
echo "Served as:      ${NINFER_SERVED_MODEL_NAME:-<artifact identity>}"
echo "Listen:         $HOST:$PORT"
echo "Max context:    $NINFER_MAX_CONTEXT"
echo "KV capacity:    $NINFER_KV_CAPACITY"
echo "KV dtype:       $NINFER_KV_DTYPE"
echo "Concurrency:    $NINFER_MAX_CONCURRENCY"
echo "Spec decoding:  $NINFER_SPEC_ENABLED ($NINFER_SPEC_METHOD, draft=$NINFER_DRAFT_TOKENS)"
echo "Vision:         $NINFER_VISION"
echo "Sampling:       temp=$NINFER_TEMPERATURE top_p=$NINFER_TOP_P top_k=$NINFER_TOP_K presence=$NINFER_PRESENCE_PENALTY greedy=$NINFER_GREEDY"
echo "Thinking:       $([ "$NINFER_NO_THINKING" = "true" ] && echo off || echo on)"
# [2026-08-31 | INFO | start-ninfer.sh.MAIN | ninfer-serve only honors --api-key; placeholder means open access
if [[ -n "$API_KEY" && "$API_KEY" != "placeholder-api-key" ]]; then
    echo "Auth:           bearer"
else
    echo "Auth:           disabled"
fi
echo ""

if [[ ! -f "$ARTIFACT_PATH" ]]; then
    echo "ERROR: NInfer artifact not found at $ARTIFACT_PATH" >&2
    echo "Download a registered artifact first, e.g.:" >&2
    echo "  hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models" >&2
    exit 1
fi

CMD=(
    ninfer-serve "$ARTIFACT_PATH"
    --host "$HOST"
    --port "$PORT"
    --max-context "$NINFER_MAX_CONTEXT"
    --kv-capacity "$NINFER_KV_CAPACITY"
    --kv-dtype "$NINFER_KV_DTYPE"
    --max-concurrency "$NINFER_MAX_CONCURRENCY"
)

if [[ -n "$NINFER_SERVED_MODEL_NAME" ]]; then
    CMD+=(--model-id "$NINFER_SERVED_MODEL_NAME")
fi

if [[ "$NINFER_VISION" == "true" ]]; then
    CMD+=(--vision)
fi

if [[ "$NINFER_SPEC_ENABLED" == "true" ]]; then
    CMD+=(--spec "$NINFER_SPEC_METHOD")
    CMD+=(--draft-tokens "$NINFER_DRAFT_TOKENS")
    if [[ "$NINFER_LM_HEAD_DRAFT" == "true" ]]; then
        CMD+=(--lm-head-draft)
    fi
fi

CMD+=(--temperature "$NINFER_TEMPERATURE")
CMD+=(--top-p "$NINFER_TOP_P")
CMD+=(--top-k "$NINFER_TOP_K")
CMD+=(--presence-penalty "$NINFER_PRESENCE_PENALTY")
if [[ "$NINFER_GREEDY" == "true" ]]; then
    CMD+=(--greedy)
fi
if [[ "$NINFER_NO_THINKING" == "true" ]]; then
    CMD+=(--no-thinking)
fi

# [2026-08-31 | INFO | start-ninfer.sh.MAIN | Pass --api-key so OpenAI Bearer / x-api-key are required
if [[ -n "$API_KEY" && "$API_KEY" != "placeholder-api-key" ]]; then
    CMD+=(--api-key "$API_KEY")
fi

exec "${CMD[@]}"
