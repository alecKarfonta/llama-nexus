#!/usr/bin/env python3
"""bench.py — A/B benchmark for OpenAI-compatible endpoints (llama.cpp, vLLM).

Measures, per run: TTFT, decode tok/s (client-side, streaming), total wall time.
Run it against the same server with SPEC=on vs SPEC=off to quantify your MTP win.

Usage:
  ./bench.py --url http://localhost:8080 --label mtp-on
  ./bench.py --url http://localhost:8080 --label mtp-off
  ./bench.py --url http://localhost:8000 --label vllm-tp2 --concurrency 4
  ./bench.py --compare results/*.json
"""
import argparse, asyncio, json, statistics, sys, time
from pathlib import Path

import aiohttp

# Three workload profiles: spec-decode acceptance differs wildly between them.
PROMPTS = {
    "code": (
        "Write a Python class implementing a thread-safe LRU cache with TTL "
        "expiry, max-size eviction, and hit/miss statistics. Include type "
        "hints and a small usage example."
    ),
    "structured": (
        "Produce a JSON array of 20 objects describing fictional GPU SKUs with "
        "fields: name, vram_gb, tdp_w, fp16_tflops, msrp_usd. Valid JSON only."
    ),
    "chat": (
        "You're hanging out on a Twitch stream. A viewer asks what the "
        "difference is between a turbo and a supercharger and whether they "
        "should swap one into a stock NA Miata. Answer casually but accurately."
    ),
}
MAX_TOKENS = 512


async def one_request(session, url, model, prompt, temperature):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    completion_tokens = None
    async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                completion_tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            token = delta.get("content") or delta.get("reasoning_content")
            if token:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_chunks += 1
    total = time.perf_counter() - t0
    # Fall back to chunk count if the server doesn't report usage on stream.
    toks = completion_tokens if completion_tokens else n_chunks
    decode_time = total - (ttft or 0)
    return {
        "ttft_s": ttft,
        "total_s": total,
        "tokens": toks,
        "decode_tps": toks / decode_time if decode_time > 0 and toks else 0.0,
    }


async def run(args):
    results = {}
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        # discover model name
        model = args.model
        if not model:
            async with session.get(f"{args.url}/v1/models") as r:
                model = (await r.json())["data"][0]["id"]
        print(f"endpoint={args.url} model={model} runs={args.runs} "
              f"concurrency={args.concurrency}\n")

        for name, prompt in PROMPTS.items():
            temp = 0.7 if name == "chat" else 0.2
            runs = []
            # warmup (populates prefix cache / JIT / cuda graphs)
            await one_request(session, args.url, model, prompt, temp)
            for i in range(args.runs):
                batch = await asyncio.gather(*[
                    one_request(session, args.url, model, prompt, temp)
                    for _ in range(args.concurrency)
                ])
                runs.extend(batch)
            tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
            ttft = [r["ttft_s"] for r in runs if r["ttft_s"]]
            results[name] = {
                "decode_tps_mean": round(statistics.mean(tps), 1),
                "decode_tps_p50": round(statistics.median(tps), 1),
                "ttft_ms_mean": round(statistics.mean(ttft) * 1000, 0),
                "n": len(runs),
            }
            r = results[name]
            print(f"{name:<11} {r['decode_tps_mean']:>7.1f} tok/s mean  "
                  f"(p50 {r['decode_tps_p50']:.1f})   TTFT {r['ttft_ms_mean']:.0f} ms")

    out = Path("results"); out.mkdir(exist_ok=True)
    path = out / f"{args.label}.json"
    path.write_text(json.dumps(
        {"label": args.label, "url": args.url, "model": model,
         "concurrency": args.concurrency, "results": results}, indent=2))
    print(f"\nsaved -> {path}")


def compare(paths):
    rows = [json.loads(Path(p).read_text()) for p in paths]
    profiles = list(PROMPTS)
    hdr = f"{'label':<16}" + "".join(f"{p+' tok/s':>18}" for p in profiles)
    print(hdr); print("-" * len(hdr))
    base = None
    for r in rows:
        line = f"{r['label']:<16}"
        for p in profiles:
            v = r["results"][p]["decode_tps_mean"]
            cell = f"{v:.1f}"
            if base:
                cell += f" ({v / base['results'][p]['decode_tps_mean']:.2f}x)"
            line += f"{cell:>18}"
        print(line)
        base = base or r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="run")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
        sys.exit(0)
    asyncio.run(run(args))
