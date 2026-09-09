#!/usr/bin/env bash
# [TIME] | INFO | SCRIPTS.BENCH_PREFILL | prompt-processing (prefill) benchmark
# for the Qwen3.8-Flash-Next deployment, companion to bench_tps.sh (which only
# measures decode).
#
# Why a separate harness: on agent workloads this model spends most of its wall
# time in prefill, not decode, and `bench_tps.sh` reports only
# timings.predicted_per_second. This script reports timings.prompt_per_second at
# several prompt sizes.
#
# Every prompt is prefixed with a random nonce so no request can be served from
# a cached prefix - these are cold-prefill numbers. Run against an otherwise
# idle server: llama-server is started with --parallel 1 for this arch, so a
# concurrent workload will queue behind (or ahead of) the probe.
#
# Usage:
#   LLAMA_URL=http://127.0.0.1:8600 LLAMA_KEY=... ./scripts/bench_prefill.sh [label]
#   SIZES="8000 28000 48000" ./scripts/bench_prefill.sh pod
set -uo pipefail

LLAMA_URL="${LLAMA_URL:-http://127.0.0.1:8600}"
LLAMA_KEY="${LLAMA_KEY:-}"
LABEL="${1:-run}"
SIZES="${SIZES:-8000 16000 28000 48000}"
RUNS="${PREFILL_RUNS:-2}"

# -u matters: piped/redirected stdout is block-buffered otherwise, so a long
# probe prints nothing until it exits and is indistinguishable from a hang.
python3 -u - "$LLAMA_URL" "$LLAMA_KEY" "$LABEL" "$RUNS" "$SIZES" <<'PY'
import json
import random
import string
import sys
import urllib.error
import urllib.request

url, key, label, runs, sizes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split()


# Measured against server-reported prompt_n on this model's tokenizer: each
# generated line (14-char nonce + index + 6 words) costs ~18 tokens. The first
# version of this script assumed ~1.15 tokens/word and overshot every target by
# ~1.94x, which silently invalidated cross-run comparisons. Treat the
# server's prompt_n as authoritative and `target` as a request, not a promise.
TOKENS_PER_LINE = 18.0


def filler(target_tokens):
    nonce = "".join(random.choices(string.ascii_lowercase, k=14))
    lines = max(1, int(target_tokens / TOKENS_PER_LINE))
    return "\n".join(
        f"{nonce}{i} sensor actuator calibration threshold sample window"
        for i in range(lines)
    )


def probe(target):
    body = json.dumps({
        "model": "qwen38-flash",
        "messages": [{"role": "user",
                      "content": "Reply with the single word OK.\n" + filler(target)}],
        # keep decode negligible so the measurement is prefill-dominated
        "max_tokens": 8,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "llama-nexus-bench/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            t = (json.load(resp).get("timings") or {})
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  {target:>6}: FAILED ({type(e).__name__})")
        return None
    return {
        "prompt_n": t.get("prompt_n") or 0,
        "pps": t.get("prompt_per_second") or 0.0,
        "cache_n": t.get("cache_n") or 0,
    }


print(f"[{label}] cold prefill (nonce-prefixed, cache-defeating)")
print(f"  {'target':>6} {'prompt_n':>9} {'cache_n':>8} {'tok/s':>9} {'seconds':>8}")
for s in sizes:
    vals = []
    for _ in range(runs):
        r = probe(int(s))
        if r and r["pps"] > 0:
            vals.append(r)
    if not vals:
        continue
    best = max(vals, key=lambda v: v["pps"])   # least-contended sample
    pn, pps = best["prompt_n"], best["pps"]
    spread = ""
    if len(vals) > 1:
        lo, hi = min(v["pps"] for v in vals), max(v["pps"] for v in vals)
        spread = f"   (runs {lo:.0f}-{hi:.0f})"
    print(f"  {int(s):>6} {pn:>9,} {best['cache_n']:>8,} {pps:>9.1f} {pn/pps:>8.1f}{spread}")
PY
