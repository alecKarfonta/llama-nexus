#!/usr/bin/env python3
"""bench_tools.py — tool-call path benchmark (grammar + MTP experiment).

Measures decode tok/s on streaming tool-call completions and records whether
valid tool_calls were returned. Pair with docker logs for MTP acceptance rate.

Usage:
  ./bench_tools.py --url http://localhost:8603 --label tools-mtp-n6
  ./bench_tools.py --compare results/tools-*.json
"""
import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

import aiohttp

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_overlay",
            "description": "Change a stream overlay element",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {"type": "string"},
                    "action": {"type": "string", "enum": ["show", "hide", "toggle"]},
                    "text": {"type": "string"},
                },
                "required": ["element", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command on the host",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]

# Prompts designed to hit the tool-call / grammar-constrained path
PROMPTS = {
    "weather": "What is the weather in Tokyo right now? You must call get_weather.",
    "overlay": (
        'A viewer asked to hide the donation goal overlay. '
        'Call set_overlay with element "donation_goal" and action "hide".'
    ),
    "command": (
        "Check disk usage on the streaming PC. "
        'Call run_command with command "df -h /".'
    ),
}

MAX_TOKENS = 128
ACCEPT_RE = re.compile(
    r"draft\s+acceptance\s+rate\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\s*\)",
    re.I,
)


async def one_request(session, url, model, prompt_key, use_tools, temperature):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPTS[prompt_key]}],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "stream": False,
    }
    if use_tools:
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"

    t0 = time.perf_counter()
    last_err = None
    obj = None
    for attempt in range(3):
        try:
            async with session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                headers={"Connection": "close"},
            ) as resp:
                resp.raise_for_status()
                obj = await resp.json()
            break
        except aiohttp.ClientError as e:
            last_err = e
            await asyncio.sleep(3)
    if obj is None:
        raise last_err

    total = time.perf_counter() - t0
    timings = obj.get("timings") or {}
    usage = obj.get("usage") or {}
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    predicted_ms = timings.get("predicted_ms", 0)
    ttft = (prompt_ms / 1000.0) if prompt_ms else None
    decode_tps = timings.get("predicted_per_second")
    if not decode_tps and predicted_ms and completion_tokens:
        decode_tps = completion_tokens / (predicted_ms / 1000.0)
    elif not decode_tps and completion_tokens and total > 0:
        decode_tps = completion_tokens / total

    choices = obj.get("choices") or []
    ch = choices[0] if choices else {}
    finish_reason = ch.get("finish_reason")
    msg = ch.get("message") or {}
    tool_calls = msg.get("tool_calls") or []

    return {
        "ttft_s": ttft,
        "total_s": total,
        "tokens": completion_tokens,
        "decode_tps": decode_tps or 0.0,
        "tool_call_deltas": len(tool_calls),
        "finish_reason": finish_reason,
        "valid_tool_call": finish_reason == "tool_calls" or len(tool_calls) > 0,
    }


def parse_acceptance_from_logs(log_text: str):
    rates = []
    for m in ACCEPT_RE.finditer(log_text):
        rates.append({
            "rate": float(m.group(1)),
            "accepted": int(m.group(2)),
            "generated": int(m.group(3)),
        })
    if not rates:
        return None
    last = rates[-1]
    return {
        "acceptance_rate": last["rate"],
        "tokens_accepted": last["accepted"],
        "tokens_generated": last["generated"],
        "samples": len(rates),
    }


async def run(args):
    results = {}
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=600, sock_read=300),
    ) as session:
        model = args.model
        if not model:
            async with session.get(f"{args.url}/v1/models") as r:
                model = (await r.json())["data"][0]["id"]

        print(
            f"endpoint={args.url} model={model} tools={args.use_tools} "
            f"runs={args.runs}\n"
        )

        for name in PROMPTS:
            runs = []
            await one_request(
                session, args.url, model, name, args.use_tools, args.temperature
            )
            for _ in range(args.runs):
                runs.append(
                    await one_request(
                        session, args.url, model, name, args.use_tools, args.temperature
                    )
                )
            tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
            ttft = [r["ttft_s"] for r in runs if r["ttft_s"]]
            valid = sum(1 for r in runs if r["valid_tool_call"])
            results[name] = {
                "decode_tps_mean": round(statistics.mean(tps), 1) if tps else 0,
                "decode_tps_p50": round(statistics.median(tps), 1) if tps else 0,
                "ttft_ms_mean": round(statistics.mean(ttft) * 1000, 0) if ttft else 0,
                "tool_call_valid_pct": round(100 * valid / len(runs), 0),
                "n": len(runs),
            }
            r = results[name]
            print(
                f"{name:<10} {r['decode_tps_mean']:>7.1f} tok/s  "
                f"TTFT {r['ttft_ms_mean']:.0f} ms  "
                f"valid_tools {r['tool_call_valid_pct']:.0f}%"
            )

    mtp_stats = None
    if args.container_logs:
        mtp_stats = parse_acceptance_from_logs(Path(args.container_logs).read_text())

    out = Path("results")
    out.mkdir(exist_ok=True)
    path = out / f"{args.label}.json"
    path.write_text(
        json.dumps(
            {
                "label": args.label,
                "url": args.url,
                "model": model,
                "use_tools": args.use_tools,
                "runs": args.runs,
                "results": results,
                "mtp_stats": mtp_stats,
            },
            indent=2,
        )
    )
    if mtp_stats:
        print(
            f"\nMTP acceptance: {mtp_stats['acceptance_rate']:.1%} "
            f"({mtp_stats['tokens_accepted']}/{mtp_stats['tokens_generated']})"
        )
    print(f"saved -> {path}")


def compare(paths):
    rows = [json.loads(Path(p).read_text()) for p in paths]
    profiles = list(PROMPTS)
    hdr = f"{'label':<22}" + "".join(f"{p+' tok/s':>16}" for p in profiles)
    if any(r.get("mtp_stats") for r in rows):
        hdr += f"{'accept%':>10}"
    print(hdr)
    print("-" * len(hdr))
    base = None
    for r in rows:
        line = f"{r['label']:<22}"
        for p in profiles:
            v = r["results"][p]["decode_tps_mean"]
            cell = f"{v:.1f}"
            if base:
                cell += f" ({v / base['results'][p]['decode_tps_mean']:.2f}x)"
            line += f"{cell:>16}"
        if any(x.get("mtp_stats") for x in rows):
            acc = r.get("mtp_stats", {}).get("acceptance_rate")
            line += f"{(acc * 100 if acc else 0):>9.1f}%"
        print(line)
        base = base or r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8603")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="tools-run")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--use-tools", action="store_true", default=True)
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--container-logs", default=None, help="docker logs file for MTP acceptance")
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()
    if args.no_tools:
        args.use_tools = False
    if args.compare:
        compare(args.compare)
        sys.exit(0)
    asyncio.run(run(args))
