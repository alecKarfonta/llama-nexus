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
import argparse, asyncio, json, os, statistics, sys, time
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

# Filler building block for ctx sweeps: repetitive-but-tokenized-dense prose.
# Repeated verbatim would be prefix-cache friendly, so we vary each block with
# an index; prompts stay realistic (meeting-notes style) and non-degenerate.
FILLER_BLOCK = (
    "Meeting notes, item {i}: the team reviewed the latency budget for the "
    "inference path. Prefill dominates the first response, so chunked batching "
    "was enabled. The cache hit rate improved after the scheduler change. "
    "Action item: rerun the throughput matrix on the Blackwell cards before "
    "the next review. Owner: platform. Status: open. Priority: medium. "
    "Follow-up scheduled for next Tuesday at the weekly sync. "
)


async def one_request(session, url, model, prompt, temperature, max_tokens=MAX_TOKENS,
                      chat_template_kwargs=None, headers=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    completion_tokens = None
    prompt_tokens = None
    async with session.post(f"{url}/v1/chat/completions", json=payload,
                            headers=headers) as resp:
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
                prompt_tokens = obj["usage"].get("prompt_tokens")
            choices = obj.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            # newer vLLM emits reasoning as `reasoning`; older as `reasoning_content`
            token = (delta.get("content") or delta.get("reasoning_content")
                     or delta.get("reasoning"))
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
        "prompt_tokens": prompt_tokens,
        "decode_tps": toks / decode_time if decode_time > 0 and toks else 0.0,
        "prefill_tps": (prompt_tokens / ttft) if (ttft and prompt_tokens) else None,
    }


def build_padded_prompt(profile_prompt, target_tokens, tokens_per_block):
    """Prepend filler blocks until the prompt is ~target_tokens long."""
    n_blocks = max(0, round((target_tokens - 120) / tokens_per_block))
    if n_blocks == 0:
        return profile_prompt, 0
    filler = "\n".join(
        FILLER_BLOCK.format(i=i) for i in range(n_blocks)
    )
    return (
        f"Below is a log of {n_blocks} meeting-note entries. Read them, then "
        f"do the task at the very end.\n\n{filler}\n\n---\n\nTask: {profile_prompt}"
    ), n_blocks


async def run(args):
    results = {}
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    # [2026-09-05 | INFO | bench.run | Thinking-off so decode tok/s is content tokens, not Qwen3.8 reasoning]
    chat_kwargs = {"enable_thinking": False} if args.no_thinking else None
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600),
        headers=headers,
    ) as session:
        # discover model name
        model = args.model
        if not model:
            async with session.get(f"{args.url}/v1/models", headers=headers) as r:
                model = (await r.json())["data"][0]["id"]
        print(f"endpoint={args.url} model={model} runs={args.runs} "
              f"concurrency={args.concurrency} thinking={not args.no_thinking}\n")

        for name, prompt in (
            (k, PROMPTS[k]) for k in args.profiles.split(",") if k in PROMPTS
        ):
            temp = 0.7 if name == "chat" else 0.2
            runs = []
            # warmup (populates prefix cache / JIT / cuda graphs)
            await one_request(session, args.url, model, prompt, temp,
                              chat_template_kwargs=chat_kwargs, headers=headers)
            for i in range(args.runs):
                batch = await asyncio.gather(*[
                    one_request(session, args.url, model, prompt, temp,
                                chat_template_kwargs=chat_kwargs, headers=headers)
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


async def calibrate_block_tokens(session, url, model, headers=None):
    """Measure tokens per FILLER_BLOCK with one probe request."""
    probe = FILLER_BLOCK.format(i=0) * 10
    r = await one_request(session, url, model, probe, 0.1, max_tokens=1,
                          headers=headers)
    pt = r.get("prompt_tokens")
    if not pt:
        raise RuntimeError("server does not report prompt_tokens; cannot calibrate")
    return pt / 10


async def run_ctx_sweep(args):
    """Decode tok/s / TTFT / prefill tok/s vs. prompt context size."""
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=1800)
    ) as session:
        model = args.model
        if not model:
            async with session.get(f"{args.url}/v1/models", headers=headers) as r:
                model = (await r.json())["data"][0]["id"]
        per_block = await calibrate_block_tokens(session, args.url, model, headers)
        print(f"endpoint={args.url} model={model}")
        print(f"filler calibration: ~{per_block:.0f} tokens/block\n")

        results = {}
        print(f"{'ctx':>8} {'prompt_tok':>10} {'ttft_s':>8} {'prefill t/s':>11} "
              f"{'decode t/s':>10} {'runs':>5}")
        for target in args.ctx_sweep:
            runs = []
            for _ in range(args.runs):
                # vary block count slightly per run to defeat prefix caching
                prompt, _ = build_padded_prompt(
                    PROMPTS["code"], target + 200 * len(runs), per_block
                )
                r = await one_request(
                    session, args.url, model, prompt, 0.2,
                    chat_template_kwargs={"enable_thinking": False},
                    headers=headers,
                )
                runs.append(r)
            tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
            ttft = [r["ttft_s"] for r in runs if r["ttft_s"]]
            prefill = [r["prefill_tps"] for r in runs if r["prefill_tps"]]
            ptoks = [r["prompt_tokens"] for r in runs if r["prompt_tokens"]]
            agg = {
                "prompt_tokens_mean": round(statistics.mean(ptoks)) if ptoks else None,
                "ttft_s_mean": round(statistics.mean(ttft), 2) if ttft else None,
                "prefill_tps_mean": round(statistics.mean(prefill), 1) if prefill else None,
                "decode_tps_mean": round(statistics.mean(tps), 1) if tps else None,
                "decode_tps_p50": round(statistics.median(tps), 1) if tps else None,
                "n": len(runs),
            }
            results[str(target)] = agg
            print(f"{target:>8} {agg['prompt_tokens_mean'] or 0:>10} "
                  f"{agg['ttft_s_mean'] or 0:>8} {agg['prefill_tps_mean'] or 0:>11} "
                  f"{agg['decode_tps_mean'] or 0:>10} {agg['n']:>5}")

        out = Path("results"); out.mkdir(exist_ok=True)
        path = out / f"{args.label}.json"
        path.write_text(json.dumps(
            {"label": args.label, "url": args.url, "model": model,
             "mode": "ctx-sweep", "concurrency": args.concurrency,
             "results": results}, indent=2))
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--api-key", default=os.environ.get("LLAMACPP_API_KEY", ""))
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="run")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--profiles", default="code,structured,chat",
                    help="Comma-separated workload profiles (code, structured, chat)")
    ap.add_argument("--compare", nargs="+", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--ctx-sweep", nargs="+", type=int, default=None,
                    help="measure decode tok/s vs prompt context sizes (tokens), "
                         "e.g. --ctx-sweep 512 2048 8192 32768 65536")
    ap.add_argument("--no-thinking", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false (Qwen3.8)")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
        sys.exit(0)
    if args.ctx_sweep:
        asyncio.run(run_ctx_sweep(args))
    else:
        asyncio.run(run(args))
