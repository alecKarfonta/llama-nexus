#!/usr/bin/env python3
# [2026-08-31 | INFO | generate_huihui_corpus | Self-generates a Huihui chat corpus for
# MTP-head distillation. Writes JSONL of {prompt, completion} pairs. Resumable.
# Single-stream only — do not add concurrent workers.
#
# Usage:
#   python3 scripts/generate_huihui_corpus.py --out /home/alec/mtp-chat-runs/selfgen \
#       --target-tokens 500000

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

SEEDS = [
    "Write a detailed technical explanation of how modern GPUs execute matrix multiplication.",
    "I'm feeling a bit down today. Tell me an interesting story about a talking cat detective.",
    "Explain transformers to a high-school student, then to a compiler engineer.",
    "You are Detective Sarah Chen in Neo-Tokyo 2087. Roleplay a tense informant scene.",
    "Write a Python function that implements a red-black tree with insert, delete, and search.",
    "Give a step-by-step solution: a box has surface area 384 cm^2 and volume 384 cm^3.",
    "Draft a polite email declining a meeting, then a blunt version of the same email.",
    "Compare KV-cache quantization schemes (bf16, int8, fp8) for LLM decode bandwidth.",
    "Help me plan a weekend trip to Montreal with food, museums, and a rainy-day backup.",
    "Continue this scene: the airlock hissed and the lights flickered once, then twice.",
    "What is the difference between speculative decoding, MTP, and EAGLE? Use examples.",
    "Write a C function for a lock-free SPSC ring buffer using C11 atomics.",
    "Roleplay as a tired starship medic treating a crewmember who shouldn't have survived.",
    "Summarize the last decade of GPU tensor-core evolution in two pages of prose.",
    "Teach me how to debug a hanging asyncio event loop. Walk through a real example.",
    "Write a short essay arguing both sides of 'open-weight models should ship MTP heads'.",
    "I'm cooking for four vegetarians tonight. Give me a menu and a shopping list.",
    "Explain attention sinks and why they matter for long-context streaming decode.",
    "A user says their chatbot is too sycophantic. Give concrete rewriting advice.",
    "Describe a heist from the perspective of the getaway driver who is also the planner.",
    "Walk me through how a Blackwell tensor-core MMA instruction retires a 4-bit weight tile.",
    "My friend ghosted after a fight. Help me write a message that is honest but not needy.",
    "Implement Dijkstra with a binary heap in Python, then explain the complexity.",
    "You are a cynical ship's AI. I just asked you to open the airlock. Roleplay the refusal.",
    "Output ONLY a JSON object describing a fictional startup: name, pitch, team, risks.",
    "Explain RoPE, NTK-aware scaling, and YaRN like I already know softmax attention.",
    "Give me a 20-minute home workout with no equipment, then a grocery list for recovery food.",
    "Write a bash script that tails docker logs and highlights OOM and CUDA errors.",
    "I'm the bartender. A time traveler just sat down and ordered 'the usual.' Continue.",
    "Compare Qwen3.8, Llama 4, and DeepSeek-V3 as local coding models. Be opinionated.",
    "Debug this: a FastAPI app returns 200 but the client hangs on chunked streaming.",
    "Write a short folk tale about a village that taxes people in unfinished sentences.",
    "Explain how CUDA graphs help decode and when they stop helping.",
    "Roleplay a code review where you are kind but refuse to rubber-stamp a race condition.",
    "Produce a JSON array of 12 log lines from a failing Kubernetes job. No prose.",
    "Teach me Kalman filters with a one-dimensional robot and then a 2D extension.",
    "Draft a postmortem for an outage caused by a bad MTP accept-rate dashboard.",
    "You are a tired librarian in a library of deleted tweets. I ask for a love story.",
    "Write Rust that parses a tiny subset of JSON without a crate. Include tests as comments.",
    "What should I say to my manager when I want to stop on-call after a burnout scare?",
    "Explain prefix caching, paged KV, and why int8 group-64 KV is a bandwidth win.",
    "Continue: the negotiation is going well until they mention the second suitcase.",
    "Give a worked example of backprop through a two-layer MLP with real numbers.",
    "Write SQL for a slowly-changing-dimension type 2 table and a query that as-of joins it.",
    "You are my debate partner. Argue that local models should never ship a thinking mode.",
    "Describe a rainy Tuesday in Osaka from the point of view of a vending machine.",
    "Help me design a chat UX that shows tokens/sec without making users feel rushed.",
    "Implement a minimal byte-pair tokenizer in Python on a 20-line corpus.",
    "A junior engineer asks why we don't just set temperature to 0. How do you answer?",
    "Write an angry-but-professional letter to a landlord about a broken heater.",
    "Explain speculative decoding acceptance curves and why draft=5 can lose to draft=4.",
    "Roleplay a heist planner who keeps getting interrupted by their cat.",
    "Output a YAML Kubernetes Deployment for a GPU inference pod. Comments allowed.",
    "I failed an interview on system design. Coach me through a URL shortener redesign.",
    "Tell me about a friendship that ended because of a shared playlist. Make it funny.",
    "Compare flash attention, paged attention, and sliding-window attention for chat.",
    "Write a Python context manager that times GPU kernels with CUDA events.",
    "You are a medieval monk copying CUDA headers. Complain beautifully.",
    "Give me a weeknight dinner plan that uses leftover rice and one sad zucchini.",
    "Explain why a 600 W board cap can hide a memory-bandwidth optimization.",
]


def _post(url: str, model: str, prompt: str, max_tokens: int, temperature: float, seed: int, api_key: str) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "enable_thinking": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return resp["choices"][0]["message"]["content"] or ""


def _count_existing(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                n += int(rec.get("completion_chars", 0))
            except json.JSONDecodeError:
                continue
    return n // 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8603/v1/chat/completions")
    ap.add_argument("--model", default="Huihui-Qwen3.8-27B-abliterated")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-tokens", type=int, default=1_000_000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY") or os.environ.get("NINFER_API_KEY") or "",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "corpus.jsonl"
    produced = _count_existing(jsonl)
    rng = random.Random(args.seed + produced)
    n = 0
    t0 = time.perf_counter()
    print(f"[gen] resume_tokens≈{produced} target={args.target_tokens}", flush=True)

    with jsonl.open("a") as f:
        while produced < args.target_tokens:
            prompt = rng.choice(SEEDS)
            if rng.random() < 0.4:
                prompt = prompt + f" Keep it under {rng.choice([200, 400, 600])} words."
            try:
                text = _post(
                    args.url, args.model, prompt, args.max_tokens, args.temperature,
                    args.seed + n, args.api_key,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[gen] request failed: {e}; sleeping 5s", flush=True)
                time.sleep(5)
                continue
            rec = {
                "prompt": prompt,
                "completion": text,
                "completion_chars": len(text),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            produced += max(1, len(text) // 4)
            n += 1
            if n % 10 == 0:
                rate = produced / max(1e-6, time.perf_counter() - t0)
                print(f"[gen] n={n} tokens≈{produced} ({rate:.0f} tok/s wall)", flush=True)

    print(f"[done] wrote {jsonl} tokens≈{produced}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
