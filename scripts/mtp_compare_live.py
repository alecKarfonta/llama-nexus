#!/usr/bin/env python3
"""
Compare baseline vs MTP on the live stack (or a local llama-server URL).

Measures decode tok/s, TTFT, completion tokens, and optionally output text equality
at temperature=0 on the same model file (MTP off vs on).

Usage:
  # Against running llamacpp-api (default :8600)
  python3 scripts/mtp_compare_live.py --url http://127.0.0.1:8600

  # Standalone two-server comparison (needs 2+ GPUs for 27B)
  python3 scripts/mtp_compare_live.py --model /path/to/MTP-Q6_K.gguf --standalone \\
    --tensor-split 2,1 --cuda-devices 0,2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mtp_bench_lib import (  # noqa: E402
    DEFAULT_DECODE_PROMPT,
    LlamaServerRunner,
    MtpServerConfig,
    expand_env_path,
    find_llama_server,
    run_chat_completion,
)

PROMPT = DEFAULT_DECODE_PROMPT


def bench_url(base_url: str, api_key: str, label: str, seed: int = 42) -> Dict[str, Any]:
    warm, _ = run_chat_completion(
        base_url, "Say OK.", max_tokens=4, temperature=0.0, seed=seed, api_key=api_key or None
    )
    metrics, text = run_chat_completion(
        base_url,
        PROMPT,
        max_tokens=128,
        temperature=0.0,
        seed=seed,
        stream=False,
        api_key=api_key or None,
    )
    return {
        "label": label,
        "metrics": metrics.__dict__,
        "text_preview": text[:200],
        "text_len": len(text),
        "warmup_error": warm.error,
    }


def bench_standalone(
    model_path: str,
    mtp_enabled: bool,
    *,
    port: int,
    n_ctx: int,
    tensor_split: str,
    draft_n_max: int,
    draft_p_min: float,
    server_bin: str,
) -> Dict[str, Any]:
    cfg = MtpServerConfig(
        model_path=model_path,
        port=port,
        n_ctx=n_ctx,
        mtp_enabled=mtp_enabled,
        draft_n_max=draft_n_max,
        draft_p_min=draft_p_min,
        tensor_split=tensor_split or None,
    )
    runner = LlamaServerRunner(cfg, server_bin=server_bin)
    label = "mtp" if mtp_enabled else "baseline"
    try:
        runner.start(timeout_seconds=600)
        time.sleep(1)
        result = bench_url(runner.base_url, os.environ.get("LLAMACPP_API_KEY", ""), label)
        acc = runner.aggregate_acceptance()
        result["acceptance"] = acc
        return result
    finally:
        runner.stop()
        time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser(description="MTP vs baseline comparison")
    parser.add_argument("--url", default="http://127.0.0.1:8600", help="Running llama-server base URL")
    parser.add_argument("--api-key", default=os.environ.get("LLAMACPP_API_KEY", "placeholder-api-key"))
    parser.add_argument("--model", help="GGUF path for standalone mode (MTP-capable file)")
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER"))
    parser.add_argument("--port", type=int, default=18120)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--tensor-split", default=os.environ.get("TENSOR_SPLIT", "2,1"))
    parser.add_argument("--draft-n-max", type=int, default=3)
    parser.add_argument("--draft-p-min", type=float, default=0.75)
    parser.add_argument("--output", default="results/mtp_compare_live.json")
    parser.add_argument("--restart-mtp-via-api", action="store_true",
                        help="Use backend API to toggle MTP and restart (experimental)")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []

    if args.standalone:
        if not args.model:
            parser.error("--model required for --standalone")
        model_path = expand_env_path(args.model)
        server_bin = args.llama_server or find_llama_server()
        print(f"Standalone compare: {model_path}")
        print(f"  tensor_split={args.tensor_split}")
        for mtp in (False, True):
            print(f"\n=== {'MTP' if mtp else 'baseline'} ===")
            results.append(
                bench_standalone(
                    model_path,
                    mtp,
                    port=args.port + (1 if mtp else 0),
                    n_ctx=args.n_ctx,
                    tensor_split=args.tensor_split,
                    draft_n_max=args.draft_n_max,
                    draft_p_min=args.draft_p_min,
                    server_bin=server_bin,
                )
            )
    else:
        print(f"Live URL: {args.url}")
        results.append(bench_url(args.url, args.api_key, "current_server"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"prompt": PROMPT[:120], "runs": results}
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        m = r.get("metrics") or {}
        print(f"\n{r['label']}:")
        if m.get("error"):
            print(f"  ERROR: {m['error']}")
            continue
        print(f"  decode tok/s:     {m.get('decode_tokens_per_second')}")
        print(f"  prompt tok/s:     {m.get('prompt_tokens_per_second')}")
        print(f"  TTFT ms:          {m.get('time_to_first_token_ms')}")
        print(f"  completion toks:  {m.get('completion_tokens')}")
        print(f"  text length:      {r.get('text_len')}")
        if r.get("acceptance"):
            acc = r["acceptance"]
            print(f"  acceptance:       {acc.get('acceptance_rate_last')}")
    if len(results) == 2:
        b, t = results[0].get("metrics", {}), results[1].get("metrics", {})
        if b.get("decode_tokens_per_second") and t.get("decode_tokens_per_second"):
            speedup = t["decode_tokens_per_second"] / b["decode_tokens_per_second"]
            print(f"\nMTP speedup (decode): {speedup:.2f}x")
        if results[0].get("text_preview") and results[1].get("text_preview"):
            same = results[0].get("text_preview") == results[1].get("text_preview")
            print(f"Output preview match (t=0): {same}")

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
