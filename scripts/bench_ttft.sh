#!/usr/bin/env bash
# [TIME] | INFO | SCRIPTS.BENCH_TTFT | agent-style cache-reuse bench: a long
# "conversation" prompt, then a follow-up that reuses the prefix with a small
# edit at the tail (the Cursor/agent pattern). Reports TTFT for cold vs warm
# follow-up and the reuse ratio. High warm/cold gap == cache-reuse working.
# Usage: ./scripts/bench_ttft.sh [label] [prefix_words]
set -euo pipefail

LABEL="${1:-ttft}"
WORDS="${2:-4000}"

python3 - "$LABEL" "$WORDS" <<'PY'
import json, random, sys, time, urllib.request

label, words = sys.argv[1], int(sys.argv[2])
random.seed(42)

cfg = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8700/api/v1/service/config",
    headers={"Authorization": "Bearer placeholder-api-key"})))
lkey = cfg["config"]["server"]["api_key"]

# plausible agent context: file listing + code excerpts, stable prefix
vocab = [f"def handler_{i}(request): return {i}" for i in range(200)]
prefix = "Context files:\n" + "\n".join(random.choice(vocab) for _ in range(words // 8))
prefix += "\n\nThe user asks: summarize the codebase structure. Assistant: it is modular."

def ttft(messages):
    body = json.dumps({
        "model": "qwen38-flash",
        "messages": messages,
        "max_tokens": 30, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8600/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {lkey}", "Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    wall = time.time() - t0
    t = r.get("timings", {})
    # server-side prompt timing is the honest TTFT component
    pt = t.get("prompt_n", 0) / max(t.get("prompt_per_second", 1), 1)
    return pt, wall, t.get("prompt_n", 0), t.get("prompt_per_second", 0)

cold_pt, _, n1, pps1 = ttft([{"role": "user", "content": prefix}])
warm_pt, _, n2, pps2 = ttft([{"role": "user", "content": prefix + "\n\nFollow-up: name the top-level modules only."}])

print(f"[{label}] prefix ~{n1} tok")
print(f"  cold  TTFT: {cold_pt:6.2f}s  ({pps1:6.0f} tok/s prompt)")
print(f"  warm  TTFT: {warm_pt:6.2f}s  ({pps2:6.0f} tok/s prompt)  follow-up {n2} tok")
print(f"  reuse gap : {cold_pt/max(warm_pt,0.001):.1f}x faster warm")
PY
