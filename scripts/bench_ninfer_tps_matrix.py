#!/usr/bin/env python3
# [2026-09-09 | INFO | bench_ninfer_tps_matrix | Comprehensive TPS harness for the live
# Ninfer deployment. Complements bench_ninfer_suite.py (5-domain quick A/B) by adding
# context-depth scaling, concurrency scaling, sustained-generation, and prefix-cache
# phases, and by correlating every request with its ninfer-serve log line so we get
# engine-side decode tok/s, TTFT, and MTP acceptance (tok/round) per measurement.
#
# Usage:
#   python3 scripts/bench_ninfer_tps_matrix.py --phase all
#   python3 scripts/bench_ninfer_tps_matrix.py --phase scenario,depth --json results/x.json
#   NINFER_API_KEY=... python3 scripts/bench_ninfer_tps_matrix.py --container ninfer-api

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("TARGET_URL", "http://127.0.0.1:8603/v1/chat/completions")
MODEL = os.environ.get("TARGET_MODEL", "Huihui-Qwen3.8-27B-abliterated")
API_KEY = os.environ.get("NINFER_API_KEY") or os.environ.get("API_KEY") or ""
CONTAINER = os.environ.get("NINFER_CONTAINER", "ninfer-api")
MAX_CTX = 32768

# ninfer-serve "done" line, e.g.
#   [req 6572] done finish=stop_token prompt=621 gen=39 cache=614 reuse=...
#   ttft=50ms prefill=176.9tok/s decode=104.7tok/s wall=0.41s ...
#   speculative=mtp 1.73tok/round (18.2%)
DONE_RE = re.compile(
    r"\[req (?P<rid>\d+)\] done finish=(?P<finish>\S+) prompt=(?P<prompt>\d+) "
    r"gen=(?P<gen>\d+) cache=(?P<cache>\d+) reuse=(?P<reuse>\S+).*?"
    r"ttft=(?P<ttft>[\d.]+)ms prefill=(?P<prefill>[\d.]+)tok/s "
    r"decode=(?P<decode>[\d.]+)tok/s wall=(?P<wall>[\d.]+)s"
)
SPEC_RE = re.compile(r"speculative=\S+ (?P<tok_round>[\d.]+)tok/round \((?P<accept>[\d.]+)%\)")


@dataclass
class Sample:
    phase: str
    case: str
    run: int
    ok: bool = False
    error: str = ""
    client_tps: float = 0.0
    wall_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # engine-side, from the correlated ninfer-serve log line
    eng_ttft_ms: float = 0.0
    eng_prefill_tps: float = 0.0
    eng_decode_tps: float = 0.0
    eng_cache_tokens: int = 0
    eng_reuse: str = ""
    eng_finish: str = ""
    mtp_tok_per_round: float = 0.0
    mtp_accept_pct: float = 0.0
    note: str = ""


# ---------------------------------------------------------------------------
# Prompt synthesis
# ---------------------------------------------------------------------------

FILLER = [
    "The deployment pipeline stages the artifact bundle before promotion.",
    "Attention kernels dominate memory bandwidth at small batch sizes.",
    "Sparse routing reduces the activated parameter count per token.",
    "The registry merges configuration fragments from multiple sources.",
    "Checkpoint intervals trade recovery time against write amplification.",
    "Quantization error concentrates in outlier activation channels.",
    "The scheduler drains the queue before it rotates the weight set.",
    "Speculative decoding moves work from latency to verification width.",
]

SCENARIOS: dict[str, str] = {
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
    "code/debug": (
        "This async Python service deadlocks under load but works fine single-threaded:\n"
        "```python\nasync def worker(q, lock):\n    while True:\n        item = await q.get()\n"
        "        async with lock:\n            await handle(item)\n        q.task_done()\n"
        "```\nExplain every plausible cause and give the corrected code."
    ),
    "json": (
        "Output ONLY a JSON array of 20 fictional user objects. Each must have fields: "
        "id, name, email, age, city, roles (array of 1-3 strings)."
    ),
    "roleplay": (
        "You are Detective Sarah Chen, a hardboiled cyberpunk investigator in Neo-Tokyo "
        "2087. I'm your informant Kaito. Roleplay a tense scene where I bring you "
        "evidence about a corporate conspiracy."
    ),
    "summarize": (
        "Summarize the following passage in exactly five bullet points, then give a "
        "one-sentence TL;DR.\n\n"
        + " ".join(FILLER * 18)
    ),
    "translate": (
        "Translate the following technical paragraph into German, then into Japanese, "
        "keeping all identifiers unchanged:\n\n"
        "The KV cache is sharded across tensor-parallel ranks; each rank holds the "
        "key and value projections for its own attention heads, so the all-reduce "
        "after the output projection reconstructs the full attention state."
    ),
}


def build_context(approx_tokens: int, seed: int) -> str:
    """Synthesize ~approx_tokens of filler with a unique header to dodge prefix cache.

    Measured on this deployment: the filler corpus tokenizes at ~6.3 chars/token
    (a 4096-token target yielded ~2578 prompt tokens at the old 4 chars/token
    assumption), so scale accordingly and shave 3% for headroom.
    """
    rng = random.Random(seed)
    target_chars = int(approx_tokens * 6.3 * 0.97)
    parts: list[str] = []
    n = 0
    while n < target_chars:
        s = rng.choice(FILLER)
        parts.append(s)
        n += len(s) + 1
    return (
        f"[ctx-{seed}-{approx_tokens}] Reference notes:\n" + " ".join(parts) +
        "\n\nSummarize the reference notes above in detail. Cover every distinct claim "
        "about kernels, scheduling, routing, quantization, and speculative decoding, "
        "and write at least 250 words."
    )


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def post_chat(prompt: str, max_tokens: int, temperature: float, seed: int,
              timeout: float) -> tuple[dict, float]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "enable_thinking": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(BASE_URL, data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload, time.perf_counter() - t0


def log_window(since: dt.datetime) -> list[dict]:
    """Parse ninfer-serve 'done' lines emitted since a timestamp."""
    try:
        out = subprocess.run(
            ["docker", "logs", "--since", since.isoformat(), CONTAINER],
            capture_output=True, text=True, timeout=20,
        ).stderr or ""
    except (subprocess.SubprocessError, OSError):
        return []
    rows: list[dict] = []
    for line in out.splitlines():
        m = DONE_RE.search(line)
        if not m:
            continue
        row = m.groupdict()
        s = SPEC_RE.search(line)
        row["mtp_tok_per_round"] = float(s.group("tok_round")) if s else 0.0
        row["mtp_accept_pct"] = float(s.group("accept")) if s else 0.0
        rows.append(row)
    return rows


def correlate(rows: list[dict], prompt: int, gen: int, used: set[str]) -> Optional[dict]:
    """Match an unconsumed log line to this request by prompt/gen token counts."""
    for row in rows:
        if row["rid"] in used:
            continue
        if int(row["prompt"]) == prompt and int(row["gen"]) == gen:
            used.add(row["rid"])
            return row
    return None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure(phase: str, case: str, run: int, prompt: str, max_tokens: int,
            temperature: float, seed: int, rows: list[dict], used: set[str],
            timeout: float = 600.0, note: str = "") -> Sample:
    s = Sample(phase=phase, case=case, run=run, note=note)
    since = dt.datetime.now(dt.timezone.utc)
    try:
        payload, wall = post_chat(prompt, max_tokens, temperature, seed, timeout)
    except urllib.error.HTTPError as exc:
        s.error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:120]}"
        return s
    except Exception as exc:  # noqa: BLE001
        s.error = f"{type(exc).__name__}: {exc}"
        return s
    usage = payload.get("usage") or {}
    s.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    s.completion_tokens = int(usage.get("completion_tokens") or 0)
    s.wall_s = wall
    s.client_tps = s.completion_tokens / wall if wall > 0 and s.completion_tokens else 0.0
    s.ok = s.completion_tokens > 0
    time.sleep(0.12)  # let the log line flush
    for r in log_window(since):
        if r["rid"] in used:
            continue
        if int(r["gen"]) != s.completion_tokens:
            continue
        if s.prompt_tokens and int(r["prompt"]) not in (s.prompt_tokens, s.prompt_tokens + 1):
            continue
        log = r
        break
    else:
        log = None
    if log:
        used.add(log["rid"])
        s.eng_ttft_ms = float(log["ttft"])
        s.eng_prefill_tps = float(log["prefill"])
        s.eng_decode_tps = float(log["decode"])
        s.eng_cache_tokens = int(log["cache"])
        s.eng_reuse = log["reuse"]
        s.eng_finish = log["finish"]
        s.mtp_tok_per_round = log["mtp_tok_per_round"]
        s.mtp_accept_pct = log["mtp_accept_pct"]
    return s


def aggregate(samples: list[Sample]) -> dict[str, Any]:
    ok = [s for s in samples if s.ok]
    if not ok:
        return {"n": 0, "errors": sorted({s.error for s in samples})[:2]}
    med = lambda f: statistics.median([f(s) for s in ok])  # noqa: E731
    out = {
        "n": len(ok),
        "client_tps": round(med(lambda s: s.client_tps), 1),
        "decode_tps": round(med(lambda s: s.eng_decode_tps), 1) if any(s.eng_decode_tps for s in ok) else None,
        "ttft_ms": round(med(lambda s: s.eng_ttft_ms), 0) if any(s.eng_ttft_ms for s in ok) else None,
        "mtp_tok_round": round(med(lambda s: s.mtp_tok_per_round), 2) if any(s.mtp_tok_per_round for s in ok) else None,
        "mtp_accept_pct": round(med(lambda s: s.mtp_accept_pct), 1) if any(s.mtp_accept_pct for s in ok) else None,
        "prompt_tok": int(med(lambda s: s.prompt_tokens)),
        "gen_tok": int(med(lambda s: s.completion_tokens)),
    }
    spread = max(s.client_tps for s in ok) - min(s.client_tps for s in ok)
    out["client_tps_spread"] = round(spread, 1)
    return out


def show(phase: str, case: str, agg: dict[str, Any]) -> None:
    if not agg.get("n"):
        print(f"  {phase:14s} {case:22s} FAILED {agg.get('errors')}", flush=True)
        return
    print(
        f"  {phase:14s} {case:22s} "
        f"decode={_f(agg['decode_tps']):>6} tps  client={_f(agg['client_tps']):>6} tps  "
        f"ttft={_f(agg['ttft_ms']):>6} ms  mtp={_f(agg['mtp_tok_round']):>4} tok/rd "
        f"({_f(agg['mtp_accept_pct']):>4}%)  p={agg['prompt_tok']:>6} g={agg['gen_tok']:>4}",
        flush=True,
    )


def _f(v: Any) -> str:
    return "  n/a" if v is None else f"{v:g}"


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_scenarios(args, rows: list[dict], used: set[str], out: list[Sample]) -> None:
    print("\n== Phase 1: scenario mix (single stream, 400 tok gen, temp 0.7) ==", flush=True)
    for name, prompt in SCENARIOS.items():
        samples = []
        for run in range(args.runs):
            s = measure("scenario", name, run, prompt, args.gen, 0.7, args.seed + run,
                        rows, used, note=f"sampler=campaign")
            samples.append(s)
            out.append(s)
            time.sleep(0.4)
        show("scenario", name, aggregate(samples))


def phase_depth(args, rows: list[dict], used: set[str], out: list[Sample]) -> None:
    print("\n== Phase 2: context depth (cold prefix each run, 200 tok gen) ==", flush=True)
    for depth in args.depths:
        samples = []
        for run in range(args.depth_runs):
            prompt = build_context(depth, seed=args.seed + run * 101)
            s = measure("depth", f"~{depth // 1024}k", run, prompt, 200, 0.0,
                        args.seed + run, rows, used)
            samples.append(s)
            out.append(s)
            time.sleep(0.4)
        show("depth", f"~{depth // 1024}k", aggregate(samples))


def phase_concurrency(args, rows: list[dict], used: set[str], out: list[Sample]) -> None:
    print("\n== Phase 3: concurrency scaling (shared medium prompt, 256 tok gen) ==", flush=True)
    prompt = SCENARIOS["chat/tech"]
    for level in args.concurrency:
        since = dt.datetime.now(dt.timezone.utc)
        t0 = time.perf_counter()

        def one(i: int) -> Sample:
            body_prompt = prompt if i % 2 else prompt + f" (variant {i})"
            return measure("concurrency", f"c{level}", i, body_prompt, 256, 0.7,
                           args.seed + i, rows, used)

        with cf.ThreadPoolExecutor(max_workers=level) as ex:
            samples = list(ex.map(one, range(level * args.concurrency_rounds)))
        batch_wall = time.perf_counter() - t0
        total_tok = sum(s.completion_tokens for s in samples)
        for s in samples:
            out.append(s)
        agg = aggregate(samples)
        agg["batch_tps"] = round(total_tok / batch_wall, 1)
        show("concurrency", f"c{level} (agg {agg.get('batch_tps')})", agg)


def phase_sustained(args, rows: list[dict], used: set[str], out: list[Sample]) -> None:
    print("\n== Phase 4: sustained generation (decode stability over long outputs) ==", flush=True)
    for gen in (256, 1024, 2048):
        samples = []
        for run in range(2):
            s = measure("sustained", f"gen{gen}", run, SCENARIOS["chat/tech"], gen, 0.7,
                        args.seed + run, rows, used)
            samples.append(s)
            out.append(s)
            time.sleep(0.4)
        show("sustained", f"gen{gen}", aggregate(samples))


def phase_prefix_cache(args, rows: list[dict], used: set[str], out: list[Sample]) -> None:
    print("\n== Phase 5: prefix-cache effect (same 12k context repeated) ==", flush=True)
    prompt = build_context(12288, seed=4242)
    for run in range(4):
        s = measure("prefix_cache", f"hit{run}", run, prompt, 128, 0.0, args.seed, rows, used)
        s.note = f"cache={s.eng_cache_tokens}/{s.prompt_tokens}"
        show("prefix_cache", f"turn{run}", aggregate([s]))
        out.append(s)
        time.sleep(0.4)


def phase_stream_ttft(args, rows: list[dict], used: set[str], out: list[Sample]) -> None:
    print("\n== Phase 6: streaming TTFT (time to first token, short + long prompt) ==", flush=True)
    cases = {
        "short": "Say OK.",
        "~2k": build_context(2048, seed=11),
        "~16k": build_context(16384, seed=12),
    }
    for name, prompt in cases.items():
        ttfts = []
        for run in range(3):
            p = prompt if run == 0 else prompt + f" v{run}"
            s = measure("stream_ttft", name, run, p, 32, 0.0, args.seed + run, rows, used)
            out.append(s)
            if s.ok:
                ttfts.append(s.eng_ttft_ms)
            time.sleep(0.3)
        if ttfts:
            print(f"  {'stream_ttft':14s} {name:22s} ttft median={statistics.median(ttfts):.0f} ms "
                  f"(min {min(ttfts):.0f} / max {max(ttfts):.0f})", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all",
                    help="comma list: scenario,depth,concurrency,sustained,prefix_cache,stream_ttft")
    ap.add_argument("--runs", type=int, default=3, help="repeats per scenario")
    ap.add_argument("--gen", type=int, default=400, help="completion tokens per scenario run")
    ap.add_argument("--depths", default="1024,4096,8192,16384,24576,28672")
    ap.add_argument("--depth-runs", type=int, default=2)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--concurrency-rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if not API_KEY:
        print("warn: no API key; relying on proxy injection" if "8700" in BASE_URL
              else "error: NINFER_API_KEY required for direct :8603", file=sys.stderr)
        if "8700" not in BASE_URL:
            return 2

    args.depths = [int(d) for d in args.depths.split(",") if d.strip()]
    args.concurrency = [int(c) for c in args.concurrency.split(",") if c.strip()]
    phases = args.phase.split(",") if args.phase != "all" else [
        "scenario", "depth", "concurrency", "sustained", "prefix_cache", "stream_ttft"]

    print(f"target={BASE_URL} model={MODEL} max_ctx={MAX_CTX} "
          f"container={CONTAINER} phases={phases}", flush=True)

    # Warm the engine so the first measured request is not a cold-start outlier.
    try:
        post_chat("Say OK.", 8, 0.0, 1, timeout=180)
    except Exception as exc:  # noqa: BLE001
        print(f"warmup failed: {exc}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    used: set[str] = set()
    out: list[Sample] = []
    t_start = time.perf_counter()

    if "scenario" in phases:
        phase_scenarios(args, rows, used, out)
    if "depth" in phases:
        phase_depth(args, rows, used, out)
    if "concurrency" in phases:
        phase_concurrency(args, rows, used, out)
    if "sustained" in phases:
        phase_sustained(args, rows, used, out)
    if "prefix_cache" in phases:
        phase_prefix_cache(args, rows, used, out)
    if "stream_ttft" in phases:
        phase_stream_ttft(args, rows, used, out)

    ok = [s for s in out if s.ok]
    print(f"\ncompleted {len(ok)}/{len(out)} samples in {time.perf_counter() - t_start:.0f}s")
    if ok:
        decodes = [s.eng_decode_tps for s in ok if s.eng_decode_tps]
        accepts = [s.mtp_accept_pct for s in ok if s.mtp_accept_pct]
        if decodes:
            print(f"overall decode median {statistics.median(decodes):.1f} tok/s "
                  f"(min {min(decodes):.1f} / max {max(decodes):.1f})")
        if accepts:
            print(f"overall MTP median {statistics.median(accepts):.1f}% "
                  f"({statistics.median([s.mtp_tok_per_round for s in ok if s.mtp_tok_per_round]):.2f} tok/round)")

    if args.json:
        payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "target": BASE_URL,
            "model": MODEL,
            "deployment": {
                "max_context": MAX_CTX,
                "speculative": "mtp draft=5 lm_head_draft=true",
            },
            "samples": [s.__dict__ for s in out],
            "summary": {
                f"{s.phase}/{s.case}": aggregate([x for x in out if x.phase == s.phase and x.case == s.case])
                for s in out
            },
        }
        path = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
