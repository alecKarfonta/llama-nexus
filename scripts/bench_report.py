#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.BENCH_REPORT | regenerate the published benchmark tables
from capture-tee records.

Every number in docs/benchmarks/qwen3.8-flash-next-blackwell-vs-ampere.md comes
out of this script, so the report can be refreshed as more samples land instead
of being hand-transcribed.

Measurements are taken from production agent traffic (the MTP replay), not a
synthetic probe. That matters: a nonce-per-line synthetic filler measures ~840
tok/s prefill on the Blackwell card while real agent prompts of the same length
measure ~1340 tok/s, because the unique filler defeats the n-gram PLE path in a
way real code and prose do not. llama-server reports per-request `timings`, and
this arch runs with --parallel 1, so each record is an uncontended measurement
of one request even when a queue is backed up behind it.

Usage:
  python3 scripts/bench_report.py                      # markdown tables
  python3 scripts/bench_report.py --json               # raw numbers
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from typing import Any, Dict, List, Optional

LOCAL_DIR = "/mnt/comfyui-models/llama-nexus/datasets"
POD_DIR = "/mnt/comfyui-models/llama-nexus/datasets/pod"

SIZE_BUCKETS = [(2000, 10000), (10000, 20000), (20000, 30000), (30000, 60000)]
MIN_BUCKET_N = 5  # below this a median is reported but flagged as thin


def load(capture_dir: str, recursive: bool = False) -> List[Dict[str, Any]]:
    """Read capture records, keeping only those with usable server timings."""
    pattern = os.path.join(capture_dir, "captures-*.jsonl")
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("timings") or {}
                pn, pps = t.get("prompt_n"), t.get("prompt_per_second")
                if not isinstance(pn, (int, float)) or not isinstance(pps, (int, float)):
                    continue
                if pps <= 0:
                    continue
                rows.append({
                    "prompt_n": pn,
                    "prefill_tps": pps,
                    "cache_n": t.get("cache_n") or 0,
                    "decode_tps": t.get("predicted_per_second"),
                    "decode_n": t.get("predicted_n"),
                    "draft_accepted": t.get("draft_n_accepted"),
                    "draft_total": t.get("draft_n"),
                    "wall_s": (rec.get("wall_ms") or 0) / 1000.0,
                })
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cold = [r for r in rows if r["cache_n"] == 0]
    warm = [r for r in rows if r["cache_n"] > 0]

    prefill = {}
    for lo, hi in SIZE_BUCKETS:
        b = [r for r in cold if lo <= r["prompt_n"] < hi]
        if b:
            prefill[f"{lo // 1000}-{hi // 1000}k"] = {
                "n": len(b),
                "median_tps": st.median(r["prefill_tps"] for r in b),
                "thin": len(b) < MIN_BUCKET_N,
            }

    dec = [r["decode_tps"] for r in rows if r.get("decode_tps")]
    acc_pairs = [(r["draft_accepted"], r["draft_total"]) for r in rows
                 if r.get("draft_total")]
    walls = [r["wall_s"] for r in rows if r["wall_s"] > 0]

    out: Dict[str, Any] = {
        "n": len(rows),
        "cold_n": len(cold),
        "warm_n": len(warm),
        "cold_pct": 100 * len(cold) / len(rows) if rows else 0,
        "prefill": prefill,
        "decode_median_tps": st.median(dec) if dec else None,
        "decode_n": len(dec),
    }
    if acc_pairs:
        a = sum(x for x, _ in acc_pairs)
        d = sum(y for _, y in acc_pairs)
        out["mtp_acceptance"] = a / d if d else None
        out["mtp_accepted"] = a
        out["mtp_drafted"] = d
    if walls:
        out["wall_median_s"] = st.median(walls)
        out["wall_mean_s"] = sum(walls) / len(walls)
    if warm:
        out["warm_reused_median"] = st.median(r["cache_n"] for r in warm)
        out["warm_new_median"] = st.median(r["prompt_n"] for r in warm)
        out["warm_total_median"] = st.median(r["cache_n"] + r["prompt_n"] for r in warm)
    return out


def ratio(a: Optional[float], b: Optional[float]) -> str:
    if not a or not b:
        return "n/a"
    return f"{a / b:.2f}x"


def md(local: Dict[str, Any], pod: Dict[str, Any]) -> str:
    L: List[str] = []
    add = L.append

    add("### Prompt processing (prefill) — production traffic\n")
    add("Median `timings.prompt_per_second` for requests with `cache_n == 0`, "
        "bucketed by prompt length. Real agent prompts over a realistic length "
        "distribution, as a check on the matched synthetic probe above.\n")
    add("| Prompt tokens | 4x RTX 3090 Ti | 1x RTX PRO 6000 | Speedup |")
    add("|---|---|---|---|")
    for key in [f"{lo // 1000}-{hi // 1000}k" for lo, hi in SIZE_BUCKETS]:
        lb, pb = local["prefill"].get(key), pod["prefill"].get(key)
        if not lb and not pb:
            continue
        def cell(b):
            if not b:
                return "—"
            flag = "*" if b["thin"] else ""
            return f"{b['median_tps']:,.0f} tok/s{flag} (n={b['n']})"
        add(f"| {key} | {cell(lb)} | {cell(pb)} | "
            f"{ratio(pb['median_tps'] if pb else None, lb['median_tps'] if lb else None)} |")
    add(f"\n`*` fewer than {MIN_BUCKET_N} samples — indicative only.\n")

    add("### Token generation (decode)\n")
    add("| Metric | 4x RTX 3090 Ti | 1x RTX PRO 6000 | Speedup |")
    add("|---|---|---|---|")
    add(f"| Median decode | {local['decode_median_tps']:.1f} tok/s "
        f"| {pod['decode_median_tps']:.1f} tok/s "
        f"| {ratio(pod['decode_median_tps'], local['decode_median_tps'])} |")
    add(f"| MTP draft acceptance | {local.get('mtp_acceptance', 0):.3f} "
        f"| {pod.get('mtp_acceptance', 0):.3f} | — |")
    add(f"| Requests measured | {local['n']:,} | {pod['n']:,} | — |")

    add("\n### KV cache reuse on agent traffic\n")
    add("The replay walks each conversation turn by turn, so most requests extend "
        "the previous prompt and only the delta is prefilled.\n")
    add("| Metric | 4x RTX 3090 Ti | 1x RTX PRO 6000 |")
    add("|---|---|---|")
    add(f"| Cold requests (full prefill) | {local['cold_n']:,} "
        f"({local['cold_pct']:.0f}%) | {pod['cold_n']:,} ({pod['cold_pct']:.0f}%) |")
    add(f"| Warm requests (cache hit) | {local['warm_n']:,} | {pod['warm_n']:,} |")
    if "warm_reused_median" in local:
        add(f"| Median tokens reused (warm) | {local['warm_reused_median']:,.0f} "
            f"| {pod.get('warm_reused_median', 0):,.0f} |")
        add(f"| Median new tokens (warm) | {local['warm_new_median']:,.0f} "
            f"| {pod.get('warm_new_median', 0):,.0f} |")
    add("\nCold/warm mix differs between the two because each ran a different "
        "session shard, so end-to-end wall time is **not** comparable between "
        "columns — only the per-phase rates above are.\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default=LOCAL_DIR)
    ap.add_argument("--pod-dir", default=POD_DIR)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    local = summarize(load(args.local_dir))
    pod = summarize(load(args.pod_dir))
    if args.json:
        print(json.dumps({"local": local, "pod": pod}, indent=2, default=float))
    else:
        print(md(local, pod))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
