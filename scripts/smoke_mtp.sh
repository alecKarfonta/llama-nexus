#!/bin/bash
# Phase 0 + MTP smoke tests (unit + live stack)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/smoke_mtp.py "$@"
