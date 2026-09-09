#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.MEASURE_ORGANIC_CURSOR | score live draft acceptance
during a real Cursor IDE session. You drive Cursor; this script opens a
time window and scores docker logs after.

Cursor must hit the capture-tee path (local :8601 or https://mlapi.us/cursor/v1),
not :8600 / /qwen/. e1 kstep @ 0.85 should already be live.

Usage:
  python3 scripts/measure_organic_cursor.py start
  # use Cursor for 15-20 real agent turns, then:
  python3 scripts/measure_organic_cursor.py score
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acceptance_log import LINE_RE  # noqa: E402

STATE = "/tmp/organic_cursor_window.json"


def accept_lines():
    p = subprocess.run(["docker", "logs", "llamacpp-api"],
                       capture_output=True, text=True, timeout=180)
    out = []
    for line in (p.stderr + p.stdout).splitlines():
        m = LINE_RE.search(line)
        if m:
            out.append((float(m.group("acc")), int(m.group("accepted")),
                        int(m.group("generated")), float(m.group("meanlen"))))
    return out


def draft_flags():
    p = subprocess.run(["docker", "inspect", "llamacpp-api", "--format",
                        "{{join .Args \" \"}}"], capture_output=True, text=True)
    args = p.stdout
    draft = p_min = None
    parts = args.split()
    for i, tok in enumerate(parts):
        if tok == "--model-draft" and i + 1 < len(parts):
            draft = parts[i + 1].rsplit("/", 1)[-1]
        if tok == "--spec-draft-p-min" and i + 1 < len(parts):
            p_min = parts[i + 1]
    return draft, p_min


def start():
    n = len(accept_lines())
    draft, p_min = draft_flags()
    state = {
        "t0": time.time(),
        "t0_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n0": n,
        "draft": draft,
        "p_min": p_min,
    }
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"[TIME] | INFO | ORGANIC | window open t0={state['t0_iso']} "
          f"draft={draft} p_min={p_min} accept_lines={n}")
    print("Drive Cursor against https://mlapi.us/cursor/v1  model qwen38-flash")
    print("(local equivalent: http://127.0.0.1:8601/v1 — not :8600)")
    print("When you have ~15-20 real agent turns: python3 scripts/measure_organic_cursor.py score")
    return 0


def score():
    if not os.path.isfile(STATE):
        raise SystemExit("no window — run start first")
    state = json.load(open(STATE))
    lines = accept_lines()
    new = lines[state["n0"]:]
    if not new:
        print("no new acceptance lines yet — keep using Cursor")
        return 1
    accepted = sum(r[1] for r in new)
    generated = sum(r[2] for r in new)
    acc = accepted / generated if generated else 0.0
    mlen = sum(r[3] for r in new) / len(new)
    draft, p_min = draft_flags()
    elapsed = time.time() - state["t0"]
    rep = {
        "window_start": state["t0_iso"],
        "elapsed_s": round(elapsed, 1),
        "draft_at_start": state["draft"],
        "draft_now": draft,
        "p_min": p_min,
        "n_requests": len(new),
        "acceptance": acc,
        "draft_len": mlen,
        "accepted_per_round": acc * mlen,
        "accepted": accepted,
        "generated": generated,
    }
    out = "/tmp/organic_cursor_score.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print(f"wrote {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("start", "score"))
    args = ap.parse_args()
    return start() if args.cmd == "start" else score()


if __name__ == "__main__":
    sys.exit(main())
