#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.DATASET_BUILD | turn capture-tee JSONL records into
train/val shards for the Cursor-specific MTP draft head (self-distillation).

Input : /var/lib/docker/volumes/llama-nexus_datasets/_data/captures/*.jsonl
Output: <out>/train.jsonl, <out>/val.jsonl + build_report.json

Each emitted sample:
  { prompt_token_ids, completion_text, sample_key, ts, tools_present, n_tokens }
Completion token ids are NOT stored — the training harness must tokenize the
completion with the same tokenizer at fit time (prompt ids were captured
server-side and ARE exact). Sample format kept minimal so the trainer owns
loss shaping (draft-head distillation over target tokens).

Filtering:
  - template_ok records only (token snapshot verified server-side)
  - finish_reason must be stop/tool_calls (drops length-truncated tails)
  - min completion chars (default 16)
  - tool-schema prefix dedupe: Cursor resends identical tool lists every
    request; hash (system_prefix + tools) and keep at most --max-per-prefix
    samples per distinct prefix so the set isn't dominated by one session
    shape. Full-prompt dedupe on top.
  - optional --scrub replaces absolute paths / emails / IPs with tokens

Usage:
  python3 scripts/dataset_build.py [--captures DIR] [--out DIR]
      [--val-frac 0.02] [--min-chars 16] [--max-per-prefix 4] [--scrub]
      [--since YYYY-MM-DD]
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

DEFAULT_CAPTURES = "/var/lib/docker/volumes/llama-nexus_datasets/_data/captures"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
ABSPATH_RE = re.compile(r"(?:/home|/Users|/root)/[\w./-]+")
KEY_RE = re.compile(r"\b(?:sk|hf)_[A-Za-z0-9]{20,}\b")


def scrub(text):
    text = KEY_RE.sub("<SECRET_KEY>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = ABSPATH_RE.sub("<PATH>", text)
    return IP_RE.sub("<IP>", text)


def record_sample_key(rec):
    """Identity of the training example itself (prompt tail + completion)."""
    msgs = rec.get("request", {}).get("messages") or []
    tail = ""
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("content"):
            tail = str(m["content"])[-2000:]
            break
    return hashlib.sha256(
        (tail + "\x00" + (rec.get("response", {}).get("content") or "")).encode()
    ).hexdigest()[:16]


def record_prefix_key(rec):
    """Cursor resends identical system+tools every turn; key on that prefix."""
    req = rec.get("request", {})
    sysmsg = ""
    for m in req.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "system":
            sysmsg = str(m.get("content"))[:4000]
            break
    tools_blob = json.dumps(req.get("tools") or [], sort_keys=True)[:8000]
    return hashlib.sha256((sysmsg + "\x00" + tools_blob).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=DEFAULT_CAPTURES)
    ap.add_argument("--out", default=None, help="default: <captures>/dataset-v1")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--min-chars", type=int, default=16)
    ap.add_argument("--max-per-prefix", type=int, default=4)
    ap.add_argument("--scrub", action="store_true")
    ap.add_argument("--since", default=None, help="only records from this date (YYYY-MM-DD)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.captures, "dataset-v1")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    files = sorted(glob.glob(os.path.join(args.captures, "captures-*.jsonl")))
    if args.since:
        files = [f for f in files if os.path.basename(f) >= f"captures-{args.since}"]

    stats = Counter()
    seen_samples, prefix_counts = set(), Counter()
    kept = []

    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    stats["drop:corrupt-line"] += 1
                    continue
                stats["read"] += 1

                if not rec.get("template_ok"):
                    stats["drop:template-unverified"] += 1
                    continue
                finish = rec.get("response", {}).get("finish_reason")
                if finish not in ("stop", "tool_calls"):
                    stats[f"drop:finish-{finish}"] += 1
                    continue
                content = rec.get("response", {}).get("content") or ""
                tool_calls = rec.get("response", {}).get("tool_calls") or []
                if len(content) < args.min_chars and not tool_calls:
                    stats["drop:too-short"] += 1
                    continue
                ids = rec.get("prompt_token_ids") or []
                if len(ids) < 8:
                    stats["drop:no-token-ids"] += 1
                    continue

                skey = record_sample_key(rec)
                if skey in seen_samples:
                    stats["drop:dupe-sample"] += 1
                    continue

                pkey = record_prefix_key(rec)
                if prefix_counts[pkey] >= args.max_per_prefix:
                    stats["drop:prefix-cap"] += 1
                    continue

                if args.scrub:
                    content = scrub(content)
                    msgs = rec.get("request", {}).get("messages") or []
                    for m in msgs if isinstance(msgs, list) else []:
                        if isinstance(m, dict) and m.get("content"):
                            m["content"] = scrub(str(m["content"]))

                seen_samples.add(skey)
                prefix_counts[pkey] += 1
                kept.append({
                    "prompt_token_ids": ids,
                    "completion_text": content,
                    "tool_calls": tool_calls,
                    "sample_key": skey,
                    "prefix_key": pkey,
                    "tools_present": bool(rec.get("request", {}).get("tools")),
                    "ts": rec.get("ts"),
                    "n_prompt_tokens": len(ids),
                })
                stats["kept"] += 1

    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_frac)) if kept else 0
    val, train = kept[:n_val], kept[n_val:]

    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "files_read": [os.path.basename(p) for p in files],
        "since": args.since,
        "scrub": args.scrub,
        "stats": dict(stats),
        "distinct_prefixes": len(prefix_counts),
        "train": len(train),
        "val": len(val),
        "total_kept": len(kept),
        "avg_prompt_tokens": (sum(r["n_prompt_tokens"] for r in kept) / len(kept)) if kept else 0,
        "tools_present_frac": (sum(1 for r in kept if r["tools_present"]) / len(kept)) if kept else 0,
    }
    with open(os.path.join(out_dir, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nout: {out_dir}/train.jsonl, val.jsonl, build_report.json")


if __name__ == "__main__":
    main()
