#!/usr/bin/env bash
# [TIME] | INFO | SCRIPTS.BENCH_TPS | fixed 3-workload decode benchmark for the
# running Qwen3.8-Flash-Next deployment. Prose / code / JSON, temp 0, 2 runs each,
# reports predicted tok/s per workload plus the latest MTP draft acceptance from logs.
#
# Decode only, and decode is just ~32% of real wall time on this box. For the
# other 68% use scripts/bench_prefill.sh; see
# fast-inference/plans/throughput-tuning.md for the split.
#
# Usage: ./scripts/bench_tps.sh [label]
#   label - free-form experiment tag printed with results (default: "run")
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8700}"
API_KEY="${API_KEY:-placeholder-api-key}"
LLAMA_KEY="${LLAMA_KEY:-$(curl -s ${BACKEND_URL}/api/v1/service/config -H "Authorization: Bearer ${API_KEY}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["config"]["server"]["api_key"])')}"
LLAMA_URL="${LLAMA_URL:-http://127.0.0.1:8600}"
LABEL="${1:-run}"
RUNS="${BENCH_RUNS:-2}"

python3 - "$LLAMA_URL" "$LLAMA_KEY" "$LABEL" "$RUNS" <<'PY'
import json
import sys
import time
import urllib.request

url, key, label, runs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

WORKLOADS = {
    "prose": "Write a detailed 300-word story about a lighthouse keeper.",
    "code": "Write a Python module with three classes Sensor, Actuator, Device. Each has __init__, read, calibrate, __repr__. Consistent style.",
    "json": "Output a JSON array of 30 sensor objects, each with fields id, name, unit, min, max, sampling_hz. No prose, JSON only.",
}

def call(prompt):
    body = json.dumps({
        "model": "qwen38-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        # Explicit UA: reverse proxies in front of remote llama-servers (e.g. the
        # RunPod pod proxy) reject urllib's default agent with a 403.
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "llama-nexus-bench/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.load(resp)
    return r.get("timings", {}).get("predicted_per_second", 0)

print(f"[{label}]")
results = {}
for name, prompt in WORKLOADS.items():
    call(prompt)  # warm
    vals = [call(prompt) for _ in range(runs)]
    results[name] = sum(vals) / len(vals)
    print(f"  {name:5s}: {results[name]:6.1f} tok/s   (runs: {', '.join(f'{v:.1f}' for v in vals)})")

g = lambda a, b: (results[a] + results[b]) / 2
print(f"  STRUCT (code+json avg): {g('code','json'):.1f} | ALL avg: {sum(results.values())/3:.1f}")
PY

# [TIME] | INFO | SCRIPTS.BENCH_TPS | latest draft acceptance line from server logs
docker logs llamacpp-api 2>&1 | grep -o 'draft acceptance = [0-9.]* ( *[0-9]* accepted / *[0-9]* generated), mean len = *[0-9.]*' | tail -1 | sed 's/^/  mtp: /' || true
