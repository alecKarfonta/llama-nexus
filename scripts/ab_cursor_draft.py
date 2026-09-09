#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.AB_CURSOR_DRAFT | A/B e1 vs stock on replayed Cursor turns.

Replays the same held-out Cursor agent transcript turns against two draft heads
at a fixed p_min, scores accepted/round from llama-server logs, and leaves the
winner live. Hits :8600 directly (not capture-tee).

Usage:
  python3 scripts/ab_cursor_draft.py
  python3 scripts/ab_cursor_draft.py --heads kstep kstep_pmin --sessions 20 --turns-per-session 5
  python3 scripts/ab_cursor_draft.py --sessions 10 --turns-per-session 3 --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acceptance_log import LINE_RE  # noqa: E402
from capture_cursor_synth import (  # noqa: E402
    _clip_text,
    _trim_messages,
    list_transcript_paths,
    load_envelopes,
    parse_transcript,
    reconstruct_tool_result,
)
from sweep_draft_knobs import (  # noqa: E402
    BACKEND,
    API_KEY,
    HEADS,
    LLAMA_URL,
    accept_lines,
    api,
    apply_point,
    container_id,
    llama_key,
    wait_for_point,
)

MODEL = "qwen38-flash"
P_MIN = 0.85
N_MIN = 0
MARGIN = 0.10
UA = "llama-nexus-cursor-ab/1.0"


def collect_turns(max_sessions: int, turns_per_session: int) -> List[Dict[str, Any]]:
    """Build fixed replay bodies from recent cursor transcripts."""
    paths = sorted(
        list_transcript_paths(["cursor"]),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    envelopes = load_envelopes(include_variants=False)
    env = envelopes[0]
    tools = env["tools"]
    turns: List[Dict[str, Any]] = []
    sessions_used = 0

    for path in paths:
        if sessions_used >= max_sessions:
            break
        events = parse_transcript(path)
        if not events:
            continue
        messages: List[Dict[str, Any]] = [{"role": "system", "content": env["system"]}]
        pending: Dict[str, Dict[str, Any]] = {}
        session_id = os.path.basename(path).replace(".jsonl", "")[:12]
        sent = 0
        session_turns: List[Dict[str, Any]] = []

        for ev in events:
            kind = ev["kind"]
            if kind == "user":
                messages.append({"role": "user", "content": _clip_text(ev["text"], 6000)})
                continue
            if kind == "tool_results":
                for call_id, result in ev["results"].items():
                    if call_id in pending:
                        pending[call_id]["content"] = _clip_text(result)
                continue

            if sent >= turns_per_session:
                break
            if messages[-1].get("role") not in ("user", "tool"):
                continue

            assistant = {"role": "assistant"}
            if ev["text"]:
                assistant["content"] = _clip_text(ev["text"], 4000)
            elif not ev.get("tool_calls"):
                assistant["content"] = ""
            if ev.get("tool_calls"):
                assistant["tool_calls"] = ev["tool_calls"]

            body = {
                "model": MODEL,
                "messages": copy.deepcopy(_trim_messages(messages)),
                "max_tokens": 400,
                "temperature": 0.0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"

            session_turns.append({
                "session": session_id,
                "turn": sent + 1,
                "path": path,
                "body": body,
            })
            sent += 1

            messages.append(assistant)
            pending = {}
            for tc in ev.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": reconstruct_tool_result(fn.get("name"), args),
                }
                pending[tc["id"]] = tool_msg
                messages.append(tool_msg)

        if session_turns:
            turns.extend(session_turns)
            sessions_used += 1

    return turns


def post_turn(key: str, body: Dict[str, Any]) -> tuple[float, tuple | None]:
    """POST one turn; return (tok/s, acceptance tuple or None)."""
    n0 = len(accept_lines())
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": UA,
        },
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.load(resp)
    tps = r.get("timings", {}).get("predicted_per_second", 0.0)

    lines = accept_lines()
    for _ in range(15):
        if len(lines) > n0:
            break
        time.sleep(1)
        lines = accept_lines()
    acc = lines[-1] if len(lines) > n0 else None
    return tps, acc


def run_head(head: str, turns: List[Dict[str, Any]], runs_warm: bool = True) -> Dict[str, Any]:
    """Restart to head, replay turns, aggregate acceptance."""
    print(f"\n=== head={head} p_min={P_MIN} turns={len(turns)} ===", flush=True)
    prev_id = container_id()
    apply_point(head, P_MIN, N_MIN)
    cid, up, probe = wait_for_point(head, P_MIN, N_MIN, prev_id)
    key = llama_key()
    print(f"    up in {up:.0f}s probe={probe:.0f}s container={cid[:12]}", flush=True)

    if runs_warm and turns:
        post_turn(key, turns[0]["body"])

    rows, tpss = [], []
    for i, turn in enumerate(turns):
        tps, acc = post_turn(key, turn["body"])
        tpss.append(tps)
        if acc:
            rows.append(acc)
        if (i + 1) % 5 == 0 or i + 1 == len(turns):
            print(f"    [{i+1}/{len(turns)}] session={turn['session']} turn={turn['turn']}",
                  flush=True)

    if container_id() != cid:
        raise RuntimeError("container replaced mid A/B")

    if not rows:
        return {"head": head, "n": 0, "error": "no acceptance lines"}

    accepted = sum(r[1] for r in rows)
    generated = sum(r[2] for r in rows)
    acc = accepted / generated if generated else 0.0
    mlen = sum(r[3] for r in rows) / len(rows)
    rep = {
        "head": head,
        "draft": HEADS[head],
        "p_min": P_MIN,
        "n": len(rows),
        "acceptance": acc,
        "draft_len": mlen,
        "accepted_per_round": acc * mlen,
        "tok_s": sum(tpss) / len(tpss),
        "accepted": accepted,
        "generated": generated,
    }
    print(f"    acc={acc:.3f} len={mlen:.2f} acc/round={acc * mlen:.2f} "
          f"tok/s={rep['tok_s']:.1f}", flush=True)
    return rep


def ship(head: str):
    api("PUT", "/api/v1/service/config", {
        "speculative": {"model_draft": HEADS[head], "gpu_layers_draft": 999},
        "mtp": {"draft_p_min": P_MIN, "draft_n_min": N_MIN},
    })
    api("POST", "/api/v1/service/restart", timeout=600)
    print(f"[TIME] | INFO | AB_CURSOR | shipped {head} @ p_min={P_MIN}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--turns-per-session", type=int, default=3)
    ap.add_argument("--heads", nargs=2, default=["kstep", "stock"],
                    metavar=("INCUMBENT", "CHALLENGER"),
                    help="sweep HEADS keys; challenger ships only if it beats "
                         "incumbent by more than --margin acc/round")
    ap.add_argument("--out", default="/tmp/cursor_ab_e1_stock.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="collect turns only, do not hit the server")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help="keep incumbent unless challenger lead exceeds this")
    args = ap.parse_args()

    incumbent, challenger = args.heads
    for h in (incumbent, challenger):
        if h not in HEADS:
            raise SystemExit(f"unknown head {h}; choose from {sorted(HEADS)}")

    turns = collect_turns(args.sessions, args.turns_per_session)
    if not turns:
        raise SystemExit("no replay turns collected from cursor transcripts")
    print(f"[TIME] | INFO | AB_CURSOR | {len(turns)} turns from "
          f"{len({t['session'] for t in turns})} sessions "
          f"({incumbent} vs {challenger})", flush=True)

    if args.dry_run:
        print(json.dumps({"turns": len(turns), "sessions": list({t['session'] for t in turns}),
                          "heads": [incumbent, challenger]}, indent=2))
        return 0

    results = {
        incumbent: run_head(incumbent, turns),
        challenger: run_head(challenger, turns),
    }

    inc = results[incumbent].get("accepted_per_round", 0.0)
    chal = results[challenger].get("accepted_per_round", 0.0)
    delta = chal - inc
    if delta > args.margin:
        winner = challenger
        reason = f"{challenger} wins by {delta:.2f} acc/round"
    else:
        winner = incumbent
        reason = (f"{incumbent} stays (challenger delta={delta:.2f} "
                  f"need > {args.margin})")

    report = {
        "turns": len(turns),
        "sessions": len({t["session"] for t in turns}),
        "p_min": P_MIN,
        "incumbent": incumbent,
        "challenger": challenger,
        "results": results,
        "winner": winner,
        "reason": reason,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[TIME] | INFO | AB_CURSOR | {incumbent}={inc:.2f} "
          f"{challenger}={chal:.2f} -> {winner}: {reason}", flush=True)
    print(f"wrote {args.out}", flush=True)

    ship(winner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
