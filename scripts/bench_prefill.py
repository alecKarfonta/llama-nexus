#!/usr/bin/env bash
# [TIME] | INFO | SCRIPTS.BENCH_PREFILL | cold-ish prefill benchmark: flush prompt cache
# (cache_reuse=0 makes every request re-process), send a long prompt, report prompt
# processing tok/s from server timings.
# Usage: ./scripts/bench_prefill.sh [label] [prompt_tokens]
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8700}"
API_KEY="${API_KEY:-placeholder-api-key}"
LABEL="${1:-run}"
PTOK="${2:-8000}"

python3 - "$BACKEND_URL" "$API_KEY" "$PTOK" <<'PY'
import json, sys, urllib.request

base, key, ptok = sys.argv[1], sys.argv[2], int(sys.argv[3])

cfg = json.load(urllib.request.urlopen(urllib.request.Request(
    f"{base}/api/v1/service/config", headers={"Authorization": f"Bearer {key}"})))
lkey = cfg["config"]["server"]["api_key"]

body = json.dumps({
    "model": "qwen38-flash",
    "messages": [{"role": "user", "content": "word " * (ptok // 1) + "\nSummarize in one sentence."}],
    "max_tokens": 40, "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
req = urllib.request.Request("http://127.0.0.1:8600/v1/chat/completions", data=body,
    headers={"Authorization": f"Bearer {lkey}", "Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
t = r.get("timings", {})
nps = t.get("prompt_per_second", 0)
print(f"prefill: {nps:.0f} tok/s ({t.get('prompt_n', '?')} prompt tokens)") 
PY

# [TIME] | INFO | SCRIPTS.BENCH_PREFILL | acceptance line for reference
docker logs llamacpp-api 2>&1 | grep -o 'draft acceptance = [0-9.]* ( *[0-9]* accepted / *[0-9]* generated)' | tail -1 | sed 's/^/  mtp: /' || true
