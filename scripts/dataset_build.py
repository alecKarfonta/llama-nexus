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

DO NOT train on `completion_render`. It is a dedupe/stats key, not a target:
render_completion() writes tool calls as `<tool_call>name(args)</tool_call>`,
while the model actually emits the template's XML form
(`<tool_call>\n<function=name>\n<parameter=k>\n...`), and the render also omits
the `\n</think>\n\n` that opens every assistant turn. ~80% of samples carry
tool calls, so a trainer using this field learns a token sequence that never
occurs in production. scripts/mtp_dataprep.py rebuilds the true sequence from
the server's own chat template and is the source of truth for training.

Filtering:
  - template_ok records only (token snapshot verified server-side)
  - finish_reason must be stop/tool_calls (drops length-truncated tails)
  - min completion chars (default 16)
  - tool-schema prefix dedupe: Cursor resends identical system+tools every
    request; hash (system_prefix + tools + session_discriminator) and keep at
    most --max-per-prefix samples per distinct prefix so one session does not
    dominate. Session discriminator is the first user message hash.
    Full-prompt dedupe on top.
  - optional --scrub replaces absolute paths / emails / IPs with tokens

Split:
  - train/val is prefix-disjoint: whole sessions are held out, so no val row
    has a sibling turn from the same conversation sitting in train.

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

DEFAULT_CAPTURES = "/mnt/comfyui-models/llama-nexus/datasets"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
ABSPATH_RE = re.compile(r"(?:/home|/Users|/root)/[\w./-]+")
KEY_RE = re.compile(r"\b(?:sk|hf)_[A-Za-z0-9]{20,}\b")


def scrub(text):
    text = KEY_RE.sub("<SECRET_KEY>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = ABSPATH_RE.sub("<PATH>", text)
    return IP_RE.sub("<IP>", text)


def render_completion(rec):
    """Full assistant turn as one string, for dedupe and stats only.

    Agent traffic is mostly tool calls with empty content, so the tool call
    payload has to be part of this string or identical-looking turns collapse
    onto each other during dedupe.

    This is deliberately NOT the training target — the shape below is not what
    the model emits (see the module docstring). scripts/mtp_dataprep.py derives
    the real token sequence from the live chat template.
    """
    resp = rec.get("response") or {}
    parts = []
    content = resp.get("content") or ""
    if content:
        parts.append(content)
    for tc in resp.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name") or "unknown"
        arguments = fn.get("arguments") or ""
        parts.append(f"<tool_call>{name}({arguments})</tool_call>")
    return "\n".join(parts)


def record_sample_key(rec):
    """Identity of the training example itself (prompt tail + rendered completion)."""
    msgs = rec.get("request", {}).get("messages") or []
    tail = ""
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("content"):
            tail = str(m["content"])[-2000:]
            break
    return hashlib.sha256(
        (tail + "\x00" + render_completion(rec)).encode()
    ).hexdigest()[:16]


def session_discriminator(rec):
    """First user message identifies the agent session within a shared envelope."""
    for m in rec.get("request", {}).get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                content = "\n".join(parts)
            return hashlib.sha256(str(content or "")[:8000].encode()).hexdigest()[:16]
    return "no-user"


def record_prefix_key(rec):
    """Cursor resends identical system+tools every turn; key on prefix + session."""
    req = rec.get("request", {})
    sysmsg = ""
    for m in req.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "system":
            sysmsg = str(m.get("content"))[:4000]
            break
    tools_blob = json.dumps(req.get("tools") or [], sort_keys=True)[:8000]
    session = session_discriminator(rec)
    return hashlib.sha256(
        (sysmsg + "\x00" + tools_blob + "\x00" + session).encode()
    ).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=DEFAULT_CAPTURES)
    ap.add_argument("--out", default=None, help="default: <captures>/dataset-v1")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--min-chars", type=int, default=16)
    ap.add_argument("--max-per-prefix", type=int, default=64)
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
                # Cursor writes "[REDACTED]" into transcripts where it scrubbed a
                # secret. capture_cursor_synth strips it from replayed history now,
                # but captures from before that fix (2026-09-04) have the model
                # echoing the placeholder as a trailing line. Training a draft head
                # on those teaches it to predict a token sequence that never occurs
                # in real traffic, which costs acceptance.
                if "[REDACTED]" in content:
                    stats["drop:redaction-echo"] += 1
                    continue
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

                rendered = render_completion(rec)
                if args.scrub:
                    rendered = scrub(rendered)
                if content and tool_calls:
                    kind = "text+tools"
                elif tool_calls:
                    kind = "tools-only"
                else:
                    kind = "text-only"
                stats[f"kind:{kind}"] += 1

                seen_samples.add(skey)
                prefix_counts[pkey] += 1
                kept.append({
                    "prompt_token_ids": ids,
                    "completion_text": content,
                    "completion_render": rendered,
                    "completion_kind": kind,
                    "tool_calls": tool_calls,
                    "sample_key": skey,
                    "prefix_key": pkey,
                    "tools_present": bool(rec.get("request", {}).get("tools")),
                    "ts": rec.get("ts"),
                    "n_prompt_tokens": len(ids),
                })
                stats["kept"] += 1

    # [TIME] | INFO | SCRIPTS.DATASET_BUILD.MAIN | hold out whole prefixes, not
    # random rows. One session contributes many turns sharing system+tools and
    # most of the conversation prefix, so a row-level shuffle lands siblings of
    # every val row in train and val loss then scores memorization inside known
    # conversations. Deal out entire prefix groups, skipping any group too large
    # for the remaining budget so val never overshoots --val-frac.
    by_prefix = {}
    for row in kept:
        by_prefix.setdefault(row["prefix_key"], []).append(row)
    groups = list(by_prefix.values())
    rng.shuffle(groups)

    n_val = max(1, int(len(kept) * args.val_frac)) if kept else 0
    val, train = [], []
    for group in groups:
        if len(val) + len(group) <= n_val:
            val.extend(group)
        else:
            train.extend(group)
    # Every group outgrew the budget (tiny corpus, or one giant session): give
    # up the smallest one so val is never empty.
    if not val and groups:
        smallest = min(groups, key=len)
        val = list(smallest)
        train = [r for r in train if r["prefix_key"] != smallest[0]["prefix_key"]]
    rng.shuffle(val)
    rng.shuffle(train)

    val_prefixes = {r["prefix_key"] for r in val}
    train_prefixes = {r["prefix_key"] for r in train}

    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    drop_stats = {k: v for k, v in sorted(stats.items()) if k.startswith("drop:")}
    kept_stats = {k: v for k, v in sorted(stats.items()) if not k.startswith("drop:")}

    report = {
        "files_read": [os.path.basename(p) for p in files],
        "since": args.since,
        "scrub": args.scrub,
        "max_per_prefix": args.max_per_prefix,
        "stats": dict(stats),
        "drop_histogram": drop_stats,
        "kept_histogram": kept_stats,
        "distinct_prefixes": len(prefix_counts),
        "train": len(train),
        "val": len(val),
        "total_kept": len(kept),
        "split": "prefix-disjoint",
        "train_prefixes": len(train_prefixes),
        "val_prefixes": len(val_prefixes),
        "split_prefix_overlap": len(train_prefixes & val_prefixes),
        "avg_prompt_tokens": (sum(r["n_prompt_tokens"] for r in kept) / len(kept)) if kept else 0,
        "tools_present_frac": (sum(1 for r in kept if r["tools_present"]) / len(kept)) if kept else 0,
        "completion_kinds": dict(Counter(r["completion_kind"] for r in kept)),
        "distinct_completions": len({r["completion_render"] for r in kept}),
        "avg_completion_chars": (
            sum(len(r["completion_render"]) for r in kept) / len(kept)
        ) if kept else 0,
    }
    with open(os.path.join(out_dir, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if drop_stats:
        print("\ndrop histogram:")
        for reason, count in drop_stats.items():
            print(f"  {reason}: {count}")
    print(f"\nout: {out_dir}/train.jsonl, val.jsonl, build_report.json")


if __name__ == "__main__":
    main()
