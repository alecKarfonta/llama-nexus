#!/usr/bin/env python3
"""
Fixed-seed output diff: MTP off vs MTP on.

MTP verification is lossless when implemented correctly — at temperature 0 with a
fixed seed, completions should match the non-MTP baseline byte-for-byte.

Usage:
  python3 scripts/mtp_quality_diff.py --model models/Qwen3.6-27B-MTP-Q6_K.gguf
  python3 scripts/mtp_quality_diff.py --model models/x.gguf --output results/mtp_quality.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from mtp_bench_lib import (  # noqa: E402
    LlamaServerRunner,
    MtpServerConfig,
    expand_env_path,
    find_llama_server,
    infer_model_family,
    run_chat_completion,
)

CODING_PROMPTS = [
    "Implement `binary_search(arr, x)` in Python with type hints and two asserts.",
    "Write a Rust function that reverses a linked list in-place. Show only code.",
    "Given JSON `{\"a\":[1,2],\"b\":{\"c\":3}}`, write JavaScript to flatten keys as 'a.0', 'b.c'.",
    "Explain quicksort partition in 4 lines, then give Python code for `partition(arr, lo, hi)`.",
    "Write SQL to find duplicate emails in table `users(email)` using GROUP BY.",
]


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _diff_summary(a: str, b: str) -> Dict[str, Any]:
    a_n, b_n = _normalize(a), _normalize(b)
    if a_n == b_n:
        return {"match": True, "length_a": len(a_n), "length_b": len(b_n)}
    # First differing line index
    a_lines = a_n.split("\n")
    b_lines = b_n.split("\n")
    first_diff = 0
    for i, (la, lb) in enumerate(zip(a_lines, b_lines)):
        if la != lb:
            first_diff = i + 1
            break
    else:
        first_diff = min(len(a_lines), len(b_lines)) + 1
    return {
        "match": False,
        "length_a": len(a_n),
        "length_b": len(b_n),
        "first_diff_line": first_diff,
        "snippet_a": a_n[max(0, first_diff - 2) : first_diff + 2][:400],
        "snippet_b": b_n[max(0, first_diff - 2) : first_diff + 2][:400],
    }


def run_prompt_pair(
    base_url: str,
    prompt: str,
    *,
    max_tokens: int,
    seed: int,
) -> Dict[str, Any]:
    metrics_off, text_off = run_chat_completion(
        base_url, prompt, max_tokens=max_tokens, temperature=0.0, seed=seed, stream=True
    )
    metrics_on, text_on = run_chat_completion(
        base_url, prompt, max_tokens=max_tokens, temperature=0.0, seed=seed, stream=True
    )
    # Note: both calls hit the same server config — caller must restart server between modes.
    return {
        "prompt": prompt[:120],
        "off": {"text": text_off, "metrics": metrics_off.__dict__},
        "on": {"text": text_on, "metrics": metrics_on.__dict__},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MTP fixed-seed quality diff suite")
    parser.add_argument("--model", required=True, help="MTP-capable GGUF path")
    parser.add_argument("--output", default="", help="JSONL output (default: results/mtp_quality_<ts>.jsonl)")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draft-n-max", type=int, default=3)
    parser.add_argument("--draft-p-min", type=float, default=0.75)
    parser.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER"))
    args = parser.parse_args()

    model_path = expand_env_path(args.model)
    server_bin = args.llama_server or find_llama_server()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output or f"results/mtp_quality_{ts}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    port = args.port

    for mode, mtp_enabled in (("baseline", False), ("mtp", True)):
        cfg = MtpServerConfig(
            model_path=model_path,
            port=port,
            mtp_enabled=mtp_enabled,
            draft_n_max=args.draft_n_max,
            draft_p_min=args.draft_p_min,
        )
        runner = LlamaServerRunner(cfg, server_bin=server_bin)
        print(f"Starting {mode} server on port {port}…", flush=True)
        runner.start()
        mode_outputs: Dict[str, str] = {}
        try:
            for i, prompt in enumerate(CODING_PROMPTS):
                _, text = run_chat_completion(
                    runner.base_url,
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    seed=args.seed + i,
                    stream=True,
                )
                mode_outputs[str(i)] = text
                print(f"  [{mode}] prompt {i + 1}/{len(CODING_PROMPTS)}", flush=True)
        finally:
            runner.stop()
            time.sleep(2)
        results.append({"mode": mode, "outputs": mode_outputs})
        port += 1

    baseline = results[0]["outputs"]
    mtp_out = results[1]["outputs"]
    diffs = []
    for key, prompt in zip(baseline.keys(), CODING_PROMPTS):
        summary = _diff_summary(baseline[key], mtp_out[key])
        diffs.append({"prompt_index": int(key), "prompt": prompt[:120], **summary})

    passed = sum(1 for d in diffs if d["match"])
    report = {
        "type": "mtp_quality_summary",
        "model_path": model_path,
        "family": infer_model_family(model_path),
        "seed": args.seed,
        "draft_n_max": args.draft_n_max,
        "draft_p_min": args.draft_p_min,
        "prompts": len(CODING_PROMPTS),
        "matches": passed,
        "pass_rate": passed / len(CODING_PROMPTS) if CODING_PROMPTS else 0.0,
        "all_match": passed == len(CODING_PROMPTS),
        "diffs": diffs,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
        for d in diffs:
            f.write(json.dumps({"type": "diff", **d}) + "\n")

    print()
    print(f"Matches: {passed}/{len(CODING_PROMPTS)}")
    print(f"{'✅ PASS' if report['all_match'] else '❌ FAIL — outputs diverged (investigate MTP correctness)'}")
    print(f"Results: {out_path}")
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
