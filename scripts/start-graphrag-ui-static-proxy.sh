#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the built frontend through a lightweight static/proxy server.

This is useful for local GraphRAG UI testing when the Vite dev server is not
available. It serves a built frontend directory and proxies /api/* and /v1/*
to the backend.

Usage:
  scripts/start-graphrag-ui-static-proxy.sh [--check] [--force]

Options:
  --check       Validate build artifacts, backend health, and the proxy script.
  --force       Start a fresh foreground server if the UI health check is inconclusive.
                The target port must still be free.
  --help        Show this help text.

Useful environment overrides:
  GRAPHRAG_UI_HOST=127.0.0.1
  GRAPHRAG_UI_PORT=55846
  GRAPHRAG_UI_DIST_DIR=/path/to/frontend/dist-graphrag
  GRAPHRAG_BACKEND_URL=http://127.0.0.1:8799
  PYTHON=python3
EOF
}

check_only=false
force_start=false

for arg in "$@"; do
  case "$arg" in
    --check)
      check_only=true
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

PYTHON_BIN="${PYTHON:-python3}"
HOST="${GRAPHRAG_UI_HOST:-127.0.0.1}"
PORT="${GRAPHRAG_UI_PORT:-55846}"
DIST_DIR="${GRAPHRAG_UI_DIST_DIR:-$ROOT_DIR/frontend/dist-graphrag}"
BACKEND_URL="${GRAPHRAG_BACKEND_URL:-http://127.0.0.1:8799}"
PROXY_SCRIPT="$ROOT_DIR/scripts/serve_graphrag_ui_static_proxy.py"
UI_URL="http://${HOST}:${PORT}/graphrag"
BACKEND_HEALTH_URL="${BACKEND_URL%/}/api/v1/graphrag/health"

port_is_open() {
  "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
}

ui_responds() {
  command -v curl >/dev/null 2>&1 && curl -fsSI "$UI_URL" >/dev/null 2>&1
}

backend_responds() {
  command -v curl >/dev/null 2>&1 && curl -fsS "$BACKEND_HEALTH_URL" >/dev/null 2>&1
}

if [ ! -f "$DIST_DIR/index.html" ]; then
  echo "Frontend build not found at $DIST_DIR/index.html" >&2
  echo "Run: node scripts/build-graphrag-ui-esbuild.mjs" >&2
  exit 1
fi

"$PYTHON_BIN" -m py_compile "$PROXY_SCRIPT"

if backend_responds; then
  echo "GraphRAG backend is healthy at $BACKEND_HEALTH_URL"
else
  echo "Warning: GraphRAG backend health did not respond at $BACKEND_HEALTH_URL" >&2
fi

if [ "$check_only" = true ]; then
  echo "Static GraphRAG UI proxy runtime is ready."
  echo "Frontend dist: $DIST_DIR"
  echo "UI URL: $UI_URL"
  exit 0
fi

if ui_responds; then
  if [ "$force_start" = false ]; then
    echo "Static GraphRAG UI is already responding at $UI_URL"
    exit 0
  fi
  echo "Port $HOST:$PORT is already serving $UI_URL; stop it before forcing a new server." >&2
  exit 1
fi

if port_is_open; then
  echo "Port $HOST:$PORT is already in use, but $UI_URL did not respond." >&2
  echo "Clear the existing process before starting the static GraphRAG UI proxy." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$PROXY_SCRIPT" \
  --host "$HOST" \
  --port "$PORT" \
  --dist-dir "$DIST_DIR" \
  --backend-url "$BACKEND_URL"
