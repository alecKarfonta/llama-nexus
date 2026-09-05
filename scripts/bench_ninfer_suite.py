#!/usr/bin/env python3
# [2026-08-30 | INFO | bench_ninfer_suite | Reproducible 5-domain TPS harness for the
# ninfer backend. All perf experiments A/B against this script so results are
# comparable across days. Prints one line per domain per config tag.
#
# Usage:
#   python3 scripts/bench_ninfer_suite.py --tag baseline
#   python3 scripts/bench_ninfer_suite.py --tag kv-int8 --url http://localhost:8603/v1/chat/completions
#
# Domains cover the observed MTP-acceptance spectrum (json > code > chat > rp).

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

# [2026-08-28 | INFO | bench_ninfer_suite | prompt set measured across the 2026-08-28
# tuning session; acceptance ranged 24%-89% across these, so they stress the whole
# speculative-decode operating envelope.
PROMPTS: dict[str, str] = {
    "chat/tech": (
        "Write a detailed technical explanation of how modern GPUs execute matrix "
        "multiplication, covering tiling, memory hierarchy, and tensor cores."
    ),
    "chat/casual": (
        "I'm feeling a bit down today. Can you tell me an interesting story to cheer "
        "me up? Make it about a talking cat detective."
    ),
    "code/python": (
        "Write a Python function that implements a red-black tree with insert, delete, "
        "and search. Include docstrings and type hints."
    ),
    "roleplay": (
        "You are Detective Sarah Chen, a hardboiled cyberpunk investigator in Neo-Tokyo "
        "2087. I'm your informant Kaito. Roleplay a tense scene where I bring you "
        "evidence about a corporate conspiracy."
    ),
    "json": (
        "Output ONLY a JSON array of 20 fictional user objects. Each must have fields: "
        "id, name, email, age, city, roles (array of 1-3 strings)."
    ),
}


def run_once(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    presence_penalty: float | None,
    seed: int,
    api_key: str,
) -> tuple[float, int]:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "enable_thinking": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if top_k is not None:
        body["top_k"] = top_k
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers
    )
    t0 = time.perf_counter()
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    dt = time.perf_counter() - t0
    return resp["usage"]["completion_tokens"] / dt, resp["usage"]["completion_tokens"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8603/v1/chat/completions")
    ap.add_argument("--model", default="Huihui-Qwen3.8-27B-abliterated")
    ap.add_argument("--tag", default="run", help="config label printed on each row")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=None, help="e.g. 20 for the ninfer campaign preset")
    ap.add_argument("--presence-penalty", type=float, default=None, help="e.g. 1.0")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY") or os.environ.get("NINFER_API_KEY") or "",
        help="Bearer token; defaults to $API_KEY / $NINFER_API_KEY",
    )
    args = ap.parse_args()

    results: dict[str, list[float]] = {}
    for domain, prompt in PROMPTS.items():
        tps_list = []
        for run in range(args.runs):
            tps, _tok = run_once(
                args.url, args.model, prompt, args.max_tokens,
                args.temperature, args.top_k, args.presence_penalty,
                args.seed + run, args.api_key,
            )
            tps_list.append(tps)
        results[domain] = tps_list
        mean = statistics.mean(tps_list)
        spread = max(tps_list) - min(tps_list)
        print(f"[{args.tag}] {domain:13s} {mean:6.1f} TPS  (±{spread:.0f}  "
              f"runs: {[f'{t:.0f}' for t in tps_list]})", flush=True)

    all_means = [statistics.mean(v) for v in results.values()]
    print(f"[{args.tag}] GEOMEAN       {statistics.geometric_mean(all_means):6.1f} TPS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
