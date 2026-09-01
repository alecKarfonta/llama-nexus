#!/usr/bin/env python3
"""Benchmark Qwen3.8-Flash-Next decode/prefill tok/s across context depths.

Hits the live llama-server (default http://127.0.0.1:8600) with increasing
prompt sizes and reports TTFT, prefill tok/s, and decode tok/s at each depth.

Usage:
    python3 scripts/bench_context_tps.py [--url http://127.0.0.1:8600]
        [--depths 4k,16k,32k] [--gen 256] [--api-key KEY] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtp_bench_lib import run_chat_completion  # noqa: E402

FILLER_SENTENCES = [
    "The deployment pipeline stages the artifact bundle before promotion.",
    "Attention kernels dominate memory bandwidth at small batch sizes.",
    "Sparse routing reduces the activated parameter count per token.",
    "The registry merges configuration fragments from multiple sources.",
    "Checkpoint intervals trade recovery time against write amplification.",
]


def build_prompt(approx_tokens: int, seed: int = 7) -> str:
    """Synthesize a prompt of roughly the requested token count (~4 chars/tok)."""
    rng = random.Random(seed)
    target_chars = approx_tokens * 4
    parts: list[str] = []
    n = 0
    while n < target_chars:
        s = rng.choice(FILLER_SENTENCES)
        parts.append(s)
        n += len(s) + 1
    context = " ".join(parts)
    return (
        f"Below is background context.\n\n{context}\n\n"
        "Using only the number of sentences above modulo 10 as your answer, "
        "reason briefly then answer with a single digit."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600")
    ap.add_argument("--api-key", default="placeholder-api-key")
    ap.add_argument("--depths", default="1024,4096,16384,32768,49152",
                    help="comma-separated approx prompt token counts")
    ap.add_argument("--gen", type=int, default=200, help="completion tokens per depth")
    ap.add_argument("--repeats", type=int, default=2, help="runs per depth (median reported)")
    ap.add_argument("--max-ctx", type=int, default=65536, help="server ctx-size guard")
    ap.add_argument("--json", default=None, help="optional JSON output path")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    if any(d + args.gen > args.max_ctx - 512 for d in depths):
        print(f"error: depth+gen exceeds server ctx ({args.max_ctx}); lower --depths or --gen")
        return 2

    print(f"target={args.url} gen={args.gen} repeats={args.repeats}  (warmup...)")
    run_chat_completion(args.url, "Say OK.", max_tokens=8, temperature=0.0,
                        api_key=args.api_key, timeout=120.0)

    rows = []
    for depth in depths:
        prompt = build_prompt(depth)
        trials = []
        for rep in range(args.repeats):
            m, _ = run_chat_completion(
                args.url, prompt,
                max_tokens=args.gen, temperature=0.0, stream=False,
                api_key=args.api_key, timeout=900.0,
            )
            if m.error:
                print(f"  depth~{depth}: ERROR {m.error[:120]}")
                trials = None
                break
            trials.append(m)
            print(f"  depth~{depth:>6} run{rep + 1}: prompt={m.prompt_tokens:>6} "
                  f"pp={m.prompt_tokens_per_second:7.1f} tok/s  "
                  f"ttft={m.time_to_first_token_ms / 1000:6.2f}s  "
                  f"tg={m.decode_tokens_per_second:6.2f} tok/s")
        if trials:
            def med(f):
                return sorted(f(t) for t in trials)[len(trials) // 2]
            rows.append({
                "target_tokens": depth,
                "prompt_tokens": med(lambda t: t.prompt_tokens),
                "prefill_tps": round(med(lambda t: t.prompt_tokens_per_second), 2),
                "ttft_s": round(med(lambda t: t.time_to_first_token_ms) / 1000, 2),
                "decode_tps": round(med(lambda t: t.decode_tokens_per_second), 2),
            })

    if not rows:
        print("no successful trials")
        return 1

    print(f"\n{'ctx depth':>10} {'prompt_tok':>10} {'prefill':>9} {'TTFT':>7} {'decode':>8}")
    print(f"{'~tokens':>10} {'':>10} {'tok/s':>9} {'s':>7} {'tok/s':>8}")
    for r in rows:
        print(f"{r['target_tokens']:>10,} {r['prompt_tokens']:>10,} "
              f"{r['prefill_tps']:>9.1f} {r['ttft_s']:>7.2f} {r['decode_tps']:>8.2f}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
