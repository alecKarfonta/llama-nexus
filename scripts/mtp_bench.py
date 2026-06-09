#!/usr/bin/env python3
"""
MTP benchmark harness (llama-server HTTP).

Runs baseline (no MTP) and MTP parameter sweeps, recording decode/prompt throughput,
TTFT, acceptance rate from server logs, and VRAM delta.

Usage:
  python3 scripts/mtp_bench.py --model models/MyModel-MTP-Q6_K.gguf
  python3 scripts/mtp_bench.py --matrix scripts/mtp_benchmark_matrix.yaml
  python3 scripts/mtp_bench.py --model models/x.gguf --sweep --parallel 1,4

Output: results/mtp_bench_<timestamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from mtp_bench_lib import (
    DEFAULT_DECODE_PROMPT,
    LlamaServerRunner,
    MtpServerConfig,
    expand_env_path,
    find_llama_server,
    infer_model_family,
    iter_matrix_cases,
    load_yaml_matrix,
    run_benchmark_trial,
)


def _system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        info["gpus"] = [line.strip() for line in out.strip().splitlines() if line.strip()]
    except (subprocess.SubprocessError, OSError):
        pass
    return info


def _run_single_case(case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = MtpServerConfig(
        model_path=case["model_path"],
        port=int(case.get("port", args.port)),
        n_ctx=int(case.get("n_ctx", args.n_ctx)),
        n_gpu_layers=int(case.get("n_gpu_layers", args.n_gpu_layers)),
        parallel_slots=int(case.get("parallel_slots", args.parallel)),
        cache_type_k=str(case.get("cache_type_k", args.cache_type_k)),
        cache_type_v=str(case.get("cache_type_v", args.cache_type_v)),
        mtp_enabled=bool(case.get("mtp_enabled")),
        draft_n_max=int(case.get("draft_n_max", args.draft_n_max)),
        draft_n_min=int(case.get("draft_n_min", args.draft_n_min)),
        draft_p_min=float(case.get("draft_p_min", args.draft_p_min)),
    )
    if args.tensor_split:
        cfg.tensor_split = args.tensor_split
    cfg.main_gpu = args.main_gpu
    record: Dict[str, Any] = {
        "case_id": case.get("case_id"),
        "model_id": case.get("model_id") or Path(cfg.model_path).stem,
        "model_path": cfg.model_path,
        "family": case.get("family") or infer_model_family(cfg.model_path),
        "quant": case.get("quant"),
        "parallel_slots": cfg.parallel_slots,
        "cache_type_k": cfg.cache_type_k,
        "cache_type_v": cfg.cache_type_v,
        "mtp_enabled": cfg.mtp_enabled,
        "draft_n_max": cfg.draft_n_max,
        "draft_n_min": cfg.draft_n_min,
        "draft_p_min": cfg.draft_p_min,
        "repetition": case.get("repetition", 0),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    runner = LlamaServerRunner(cfg, server_bin=args.llama_server)
    try:
        print(
            f"  → {record['case_id']} "
            f"({'MTP' if cfg.mtp_enabled else 'baseline'}) port={cfg.port}",
            flush=True,
        )
        runner.start(timeout_seconds=args.server_timeout)
        trial = run_benchmark_trial(
            runner,
            max_tokens=int(case.get("max_tokens", args.max_tokens)),
            seed=args.seed,
        )
        record["metrics"] = trial
        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)
        print(f"    ✗ {exc}", flush=True)
    finally:
        runner.stop()
        time.sleep(args.cooldown)
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    return record


def _build_sweep_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    model_path = expand_env_path(args.model)
    family = infer_model_family(model_path)
    parallel_values = [int(x) for x in args.parallel.split(",")]
    draft_n_max_values = [int(x) for x in args.draft_n_max_list.split(",")]
    draft_p_min_values = [float(x) for x in args.draft_p_min_list.split(",")]
    port = args.port
    cases: List[Dict[str, Any]] = []
    for parallel in parallel_values:
        if args.baseline:
            cases.append(
                {
                    "case_id": f"{Path(model_path).stem}_np{parallel}_baseline",
                    "model_path": model_path,
                    "family": family,
                    "parallel_slots": parallel,
                    "mtp_enabled": False,
                    "port": port,
                }
            )
            port += 1
        if args.sweep:
            for n_max in draft_n_max_values:
                for p_min in draft_p_min_values:
                    cases.append(
                        {
                            "case_id": f"{Path(model_path).stem}_np{parallel}_n{n_max}_p{p_min}",
                            "model_path": model_path,
                            "family": family,
                            "parallel_slots": parallel,
                            "mtp_enabled": True,
                            "draft_n_max": n_max,
                            "draft_p_min": p_min,
                            "port": port,
                        }
                    )
                    port += 1
        elif args.mtp:
            cases.append(
                {
                    "case_id": f"{Path(model_path).stem}_np{parallel}_mtp",
                    "model_path": model_path,
                    "family": family,
                    "parallel_slots": parallel,
                    "mtp_enabled": True,
                    "draft_n_max": args.draft_n_max,
                    "draft_p_min": args.draft_p_min,
                    "port": port,
                }
            )
            port += 1
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="MTP llama-server benchmark harness")
    parser.add_argument("--model", help="Path to MTP-capable GGUF")
    parser.add_argument("--matrix", help="YAML matrix (see scripts/mtp_benchmark_matrix.yaml)")
    parser.add_argument("--output", help="JSONL output path (default: results/mtp_bench_<ts>.jsonl)")
    parser.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER"), help="llama-server binary")
    parser.add_argument("--sweep", action="store_true", help="Full MTP grid (draft_n_max × draft_p_min)")
    parser.add_argument("--mtp", action="store_true", help="Single MTP run (use with --draft-n-max/--draft-p-min)")
    parser.add_argument("--baseline", action="store_true", default=True, help="Include non-MTP baseline")
    parser.add_argument("--no-baseline", action="store_false", dest="baseline")
    parser.add_argument("--parallel", default="1", help="Comma-separated --parallel values")
    parser.add_argument("--draft-n-max-list", default="2,3,4,5")
    parser.add_argument("--draft-p-min-list", default="0.5,0.75,0.9")
    parser.add_argument("--draft-n-max", type=int, default=3)
    parser.add_argument("--draft-n-min", type=int, default=0)
    parser.add_argument("--draft-p-min", type=float, default=0.75)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--tensor-split", default=os.environ.get("TENSOR_SPLIT", ""))
    parser.add_argument("--main-gpu", type=int, default=int(os.environ.get("MAIN_GPU", "0")))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-timeout", type=float, default=300.0)
    parser.add_argument("--cooldown", type=float, default=3.0, help="Seconds between cases")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.matrix:
        matrix = load_yaml_matrix(Path(args.matrix))
        cases = iter_matrix_cases(matrix)
    elif args.model:
        if not args.sweep and not args.mtp and not args.baseline:
            parser.error("Specify --sweep, --mtp, or --baseline (default)")
        cases = _build_sweep_cases(args)
    else:
        parser.error("Provide --model or --matrix")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output or f"results/mtp_bench_{ts}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              MTP Benchmark (llama-server)                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Server:  {args.llama_server}")
    print(f"Cases:   {len(cases)}")
    print(f"Output:  {out_path}")
    print()

    if args.dry_run:
        for case in cases:
            print(json.dumps(case, indent=2))
        return 0

    if args.llama_server:
        args.llama_server = expand_env_path(args.llama_server)
    else:
        try:
            args.llama_server = find_llama_server()
        except FileNotFoundError as exc:
            parser.error(str(exc))

    meta = {"system": _system_info(), "llama_server": args.llama_server, "case_count": len(cases)}
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(json.dumps({"type": "meta", **meta}) + "\n")
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case.get('case_id')}")
            record = _run_single_case(case, args)
            out.write(json.dumps(record) + "\n")
            out.flush()

    print()
    print(f"✅ Wrote {out_path}")
    print(f"   Report: python3 scripts/mtp_report.py --bench {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
