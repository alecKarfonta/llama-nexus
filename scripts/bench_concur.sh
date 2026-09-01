#!/usr/bin/env bash
# [TIME] | INFO | SCRIPTS.BENCH_CONCUR | N concurrent chat completions; reports
# per-stream decode tok/s, aggregate tok/s, and TTFT spread. Tests whether extra
# parallel slots convert to real concurrent throughput on this deployment.
# Usage: ./scripts/bench_concur.sh [n_streams] [label]
set -euo pipefail

N="${1:-2}"
LABEL="${2:-concur}"

python3 - "$N" "$LABEL" <<'PY'
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

n, label = int(sys.argv[1]), sys.argv[2]

cfg = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8700/api/v1/service/config",
    headers={"Authorization": "Bearer placeholder-api-key"})))
lkey = cfg["config"]["server"]["api_key"]

def call(i):
    body = json.dumps({
        "model": "qwen38-flash",
        "messages": [{"role": "user", "content":
            f"Write a Python class hierarchy for sensor {i} with three classes, "
            "each with __init__, read, calibrate methods."}],
        "max_tokens": 350, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8600/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {lkey}", "Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    wall = time.time() - t0
    t = r.get("timings", {})
    return {
        "tps": t.get("predicted_per_second", 0),
        "ttft": t.get("prompt_n", 0) / max(t.get("prompt_per_second", 1), 1),
        "tokens": t.get("predicted_n", 0),
        "wall": wall,
    }

# warm
call(0)
t0 = time.time()
with ThreadPoolExecutor(max_workers=n) as ex:
    rs = list(ex.map(call, range(n)))
total = time.time() - t0

tps = [r["tps"] for r in rs]
agg = sum(r["tokens"] for r in rs) / total
print(f"[{label}] streams={n}")
print(f"  per-stream tps: {', '.join(f'{v:.1f}' for v in tps)} (min {min(tps):.1f})")
print(f"  aggregate tps  : {agg:.1f}  (wall {total:.1f}s)")
PY
