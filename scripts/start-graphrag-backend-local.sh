#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the Llama Nexus backend for local GraphRAG document-chat testing.

Usage:
  scripts/start-graphrag-backend-local.sh [--check] [--print-env] [--force]

Options:
  --check       Validate the local runtime and paths without starting uvicorn.
  --print-env   Print the exported environment for debugging.
  --force       Start even if the target port already responds.
  --help        Show this help text.

Useful environment overrides:
  BACKEND_HOST=127.0.0.1
  BACKEND_PORT=8799
  LLAMA_NEXUS_BACKEND_DATA_DIR=/path/to/data
  LLAMA_NEXUS_BACKEND_VENV=/path/to/venv
  LLAMA_NEXUS_INSTALL_REQUIREMENTS=1
  LLAMA_NEXUS_LOAD_DOTENV=0
  LLAMA_NEXUS_SKIP_PIP_INSTALL=1
EOF
}

check_only=false
print_env=false
force_start=false

for arg in "$@"; do
  case "$arg" in
    --check)
      check_only=true
      ;;
    --print-env)
      print_env=true
      ;;
    --force)
      force_start=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [ "${LLAMA_NEXUS_LOAD_DOTENV:-1}" = "1" ] && [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

BACKEND_DIR="$ROOT_DIR/backend"
DATA_DIR="${LLAMA_NEXUS_BACKEND_DATA_DIR:-$BACKEND_DIR/data}"
VENV_DIR="${LLAMA_NEXUS_BACKEND_VENV:-${XDG_CACHE_HOME:-$HOME/.cache}/llama-nexus/backend-venv}"
PYTHON_BIN="${PYTHON:-python3}"
HOST="${BACKEND_HOST:-127.0.0.1}"
PORT="${BACKEND_PORT:-8799}"
HEALTH_URL="http://${HOST}:${PORT}/api/v1/graphrag/health"

mkdir -p \
  "$DATA_DIR" \
  "$DATA_DIR/conversations" \
  "$DATA_DIR/models" \
  "$DATA_DIR/prompts" \
  "$DATA_DIR/batch" \
  "$DATA_DIR/benchmarks" \
  "$DATA_DIR/quantization" \
  "$DATA_DIR/quantization_output" \
  "$DATA_DIR/rag" \
  "$DATA_DIR/runtime" \
  "$DATA_DIR/finetune_datasets" \
  "$DATA_DIR/finetune_jobs" \
  "$DATA_DIR/adapter_exports" \
  "$DATA_DIR/adapters"

export GRAPHRAG_ENABLED="${GRAPHRAG_ENABLED:-true}"
export LLAMA_NEXUS_ROOT="${LLAMA_NEXUS_ROOT:-$ROOT_DIR}"
export PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR}"
export PYTHONPATH="$BACKEND_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LLAMA_NEXUS_DB_PATH="${LLAMA_NEXUS_DB_PATH:-$DATA_DIR/llama_nexus.db}"
export TOKEN_DB_PATH="${TOKEN_DB_PATH:-$DATA_DIR/token_usage.db}"
export CONVERSATION_STORAGE_DIR="${CONVERSATION_STORAGE_DIR:-$DATA_DIR/conversations}"
export MODEL_REGISTRY_DB_PATH="${MODEL_REGISTRY_DB_PATH:-$DATA_DIR/model_registry.db}"
export PROMPT_LIBRARY_DB_PATH="${PROMPT_LIBRARY_DB_PATH:-$DATA_DIR/prompt_library.db}"
export BATCH_DB_PATH="${BATCH_DB_PATH:-$DATA_DIR/batch_jobs.db}"
export BENCHMARK_DB_PATH="${BENCHMARK_DB_PATH:-$DATA_DIR/benchmarks.db}"
export EMBEDDING_BENCHMARK_DB_PATH="${EMBEDDING_BENCHMARK_DB_PATH:-$DATA_DIR/embedding_benchmarks.db}"
export QUANTIZATION_DB_PATH="${QUANTIZATION_DB_PATH:-$DATA_DIR/quantization.db}"
export QUANTIZATION_OUTPUT_DIR="${QUANTIZATION_OUTPUT_DIR:-$DATA_DIR/quantization_output}"
export FEEDBACK_DB_PATH="${FEEDBACK_DB_PATH:-$DATA_DIR/retrieval_feedback.db}"
export RAG_DB_PATH="${RAG_DB_PATH:-$DATA_DIR/rag}"
export WORKFLOW_DB_PATH="${WORKFLOW_DB_PATH:-$DATA_DIR/workflows.db}"
export PROFILE_DB_PATH="${PROFILE_DB_PATH:-$DATA_DIR/deploy_profiles.db}"
export MODELS_DIR="${MODELS_DIR:-$DATA_DIR/models}"
export ADAPTERS_DIR="${ADAPTERS_DIR:-$DATA_DIR/adapters}"
export ADAPTER_EXPORTS_DIR="${ADAPTER_EXPORTS_DIR:-$DATA_DIR/adapter_exports}"
export FINETUNE_DATASET_DIR="${FINETUNE_DATASET_DIR:-$DATA_DIR/finetune_datasets}"
export FINETUNE_JOB_DIR="${FINETUNE_JOB_DIR:-$DATA_DIR/finetune_jobs}"
export DISTILLATION_OUTPUT_DIR="${DISTILLATION_OUTPUT_DIR:-$DATA_DIR/distillation}"
export VLLM_DEPLOY_CONFIG_PATH="${VLLM_DEPLOY_CONFIG_PATH:-$DATA_DIR/vllm_deploy_config.json}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

ensure_venv() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating backend venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
  fi

  if [ "${LLAMA_NEXUS_INSTALL_REQUIREMENTS:-0}" = "1" ]; then
    "$VENV_DIR/bin/python" -m pip install -r "$BACKEND_DIR/requirements.txt"
    return
  fi

  if "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import aiosqlite
PY
  then
    return
  fi

  if [ "${LLAMA_NEXUS_SKIP_PIP_INSTALL:-0}" = "1" ]; then
    echo "aiosqlite is missing. Re-run without LLAMA_NEXUS_SKIP_PIP_INSTALL=1 or install backend requirements." >&2
    exit 1
  fi

  "$VENV_DIR/bin/python" -m pip install "aiosqlite==0.19.0"
}

port_has_healthy_backend() {
  command -v curl >/dev/null 2>&1 && curl -fsS "$HEALTH_URL" >/dev/null 2>&1
}

port_is_open() {
  "$VENV_DIR/bin/python" - "$HOST" "$PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
}

ensure_venv

if [ "$print_env" = true ]; then
  env | grep -E '^(GRAPHRAG_ENABLED|LLAMA_NEXUS|PROJECT_DIR|PYTHONPATH|PYTHONFAULTHANDLER|PYTHONUNBUFFERED|TOKEN_DB_PATH|CONVERSATION_STORAGE_DIR|MODEL_REGISTRY_DB_PATH|PROMPT_LIBRARY_DB_PATH|BATCH_DB_PATH|BENCHMARK_DB_PATH|EMBEDDING_BENCHMARK_DB_PATH|QUANTIZATION_DB_PATH|QUANTIZATION_OUTPUT_DIR|FEEDBACK_DB_PATH|RAG_DB_PATH|WORKFLOW_DB_PATH|PROFILE_DB_PATH|MODELS_DIR|ADAPTERS_DIR|ADAPTER_EXPORTS_DIR|FINETUNE_|DISTILLATION_OUTPUT_DIR|VLLM_DEPLOY_CONFIG_PATH)=' | sort
fi

if [ "$check_only" = true ]; then
  "$VENV_DIR/bin/python" - <<'PY'
import aiosqlite
import fastapi
import uvicorn
print("Local GraphRAG backend runtime is ready.")
PY
  exit 0
fi

if [ "$force_start" = false ]; then
  if port_has_healthy_backend; then
    echo "GraphRAG backend is already healthy at $HEALTH_URL"
    exit 0
  fi
  if port_is_open; then
    echo "Port $HOST:$PORT is already in use, but $HEALTH_URL did not respond." >&2
    echo "Stop the existing process, choose BACKEND_PORT, or pass --force." >&2
    exit 1
  fi
fi

echo "Starting Llama Nexus backend for local GraphRAG testing"
echo "  backend: $BACKEND_DIR"
echo "  data:    $DATA_DIR"
echo "  url:     http://$HOST:$PORT"

cd "$BACKEND_DIR"
exec "$VENV_DIR/bin/python" -m uvicorn main:app --host "$HOST" --port "$PORT"
