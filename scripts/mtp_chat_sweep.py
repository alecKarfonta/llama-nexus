#!/usr/bin/env python3
"""Quick chat-profile MTP sweep on the live llamacpp-api stack.

Restarts via backend API for each (draft_n_max, draft_p_min), runs chat-only
bench (temp=0.7), then restores the original deploy config.

Usage:
  python3 scripts/mtp_chat_sweep.py
  python3 scripts/mtp_chat_sweep.py --n-max 1,2,3,4 --p-min 0.5,0.75,0.9 --runs 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "fast-inference" / "bench" / "bench.py"


def http_json(method: str, url: str, body: Optional[Dict[str, Any]] = None, timeout: float = 600.0) -> Any:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(url: str, timeout_s: float = 900.0) -> None:
    deadline = time.time() + timeout_s
    health = url.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(3)
    raise TimeoutError(f"Server not ready at {health} after {timeout_s}s")


def run_chat_bench(
    infer_url: str,
    api_key: str,
    label: str,
    runs: int,
) -> Dict[str, Any]:
    bench_args = [
        "--url",
        infer_url,
        "--api-key",
        api_key,
        "--profiles",
        "chat",
        "--runs",
        str(runs),
        "--label",
        label,
    ]
    try:
        import aiohttp  # noqa: F401

        proc = subprocess.run(
            [sys.executable, str(BENCH), *bench_args],
            cwd=BENCH.parent,
            capture_output=True,
            text=True,
        )
    except ModuleNotFoundError:
        inner = "pip install -q aiohttp && python3 bench.py " + " ".join(
            subprocess.list2cmdline([a]) if " " in a else a for a in bench_args
        )
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "host",
                "-v",
                f"{BENCH.parent}:/bench",
                "-w",
                "/bench",
                "python:3.12-slim",
                "bash",
                "-c",
                f"pip install -q aiohttp && python3 bench.py {' '.join(bench_args)}",
            ],
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"bench failed ({label}): {proc.stderr or proc.stdout}")
    result_path = BENCH.parent / "results" / f"{label}.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing bench output: {result_path}")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return data["results"]["chat"]


def available_cuda_devices() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        ids = [line.strip() for line in out.splitlines() if line.strip().isdigit()]
        if ids:
            return ",".join(ids)
    except (subprocess.SubprocessError, OSError, ValueError):
        pass
    return "0"


def build_config_patch(
    base: Dict[str, Any],
    *,
    model_name: str,
    model_variant: str,
    mtp_enabled: bool,
    draft_n_max: int,
    draft_p_min: float,
) -> Dict[str, Any]:
    cfg = deepcopy(base)
    cfg.setdefault("model", {})
    cfg["model"]["name"] = model_name
    cfg["model"]["variant"] = model_variant
    cfg.setdefault("mtp", {})
    cfg["mtp"]["enabled"] = mtp_enabled
    cfg["mtp"]["workload_profile"] = "custom"
    cfg["mtp"]["draft_n_max"] = max(draft_n_max, 1) if mtp_enabled else 2
    cfg["mtp"]["draft_n_min"] = 0
    cfg["mtp"]["draft_p_min"] = draft_p_min if mtp_enabled else 0.75
    cfg.setdefault("performance", {})
    cfg["performance"]["parallel_slots"] = 1
    cfg.setdefault("execution", {})
    devices = available_cuda_devices()
    cfg["execution"]["cuda_devices"] = devices
    if "," not in devices:
        cfg["performance"]["tensor_split"] = ""
    return cfg


def restart_with_config(backend_url: str, config: Dict[str, Any]) -> None:
    url = backend_url.rstrip("/") + "/api/v1/service/action"
    http_json("POST", url, {"action": "restart", "backend": "llamacpp", "config": config})


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat MTP parameter sweep (live stack)")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8700")
    parser.add_argument("--infer-url", default="http://127.0.0.1:8600")
    parser.add_argument("--api-key", default="placeholder-api-key")
    parser.add_argument("--model-name", default="Qwen3.6-35B-A3B-uncensored-heretic-MTP")
    parser.add_argument("--model-variant", default="Q6_K")
    parser.add_argument("--n-max", default="1,2,3,4", help="Comma-separated draft_n_max values")
    parser.add_argument("--p-min", default="0.5,0.75,0.9", help="Comma-separated draft_p_min values")
    parser.add_argument("--runs", type=int, default=2, help="Bench runs per case")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    n_max_values = parse_int_list(args.n_max)
    p_min_values = parse_float_list(args.p_min)

    status = http_json("GET", args.backend_url.rstrip("/") + "/v1/service/status")
    original_config = deepcopy(status["config"])
    cases: List[Tuple[str, bool, int, float]] = []
    if not args.skip_baseline:
        cases.append(("baseline", False, 0, 0.0))
    for n in n_max_values:
        for p in p_min_values:
            cases.append((f"n{n}_p{str(p).replace('.', '')}", True, n, p))

    print("MTP chat sweep")
    print(f"  model: {args.model_name} / {args.model_variant}")
    print(f"  infer: {args.infer_url}")
    print(f"  cases: {len(cases)}  runs/case: {args.runs}")
    print(f"  n_max: {n_max_values}  p_min: {p_min_values}")
    print()

    if args.dry_run:
        for label, enabled, n, p in cases:
            print(f"  {label}: mtp={enabled} n={n} p={p}")
        return 0

    results: List[Dict[str, Any]] = []
    baseline_tps: Optional[float] = None

    try:
        for i, (label, mtp_enabled, n_max, p_min) in enumerate(cases, 1):
            patch = build_config_patch(
                original_config,
                model_name=args.model_name,
                model_variant=args.model_variant,
                mtp_enabled=mtp_enabled,
                draft_n_max=n_max if mtp_enabled else 0,
                draft_p_min=p_min if mtp_enabled else 0.75,
            )
            print(f"[{i}/{len(cases)}] {label} — restarting…", flush=True)
            t0 = time.time()
            restart_with_config(args.backend_url, patch)
            wait_ready(args.infer_url)
            print(f"  ready in {time.time() - t0:.0f}s, benchmarking…", flush=True)
            chat = run_chat_bench(args.infer_url, args.api_key, f"chat_sweep_{label}", args.runs)
            row = {
                "label": label,
                "mtp_enabled": mtp_enabled,
                "draft_n_max": n_max if mtp_enabled else None,
                "draft_p_min": p_min if mtp_enabled else None,
                "chat_decode_tps_mean": chat["decode_tps_mean"],
                "chat_decode_tps_p50": chat["decode_tps_p50"],
                "ttft_ms_mean": chat["ttft_ms_mean"],
            }
            if baseline_tps is None and not mtp_enabled:
                baseline_tps = chat["decode_tps_mean"]
            if baseline_tps and mtp_enabled:
                row["speedup_vs_baseline"] = round(chat["decode_tps_mean"] / baseline_tps, 3)
            results.append(row)
            speed = row.get("speedup_vs_baseline")
            extra = f"  ({speed:.2f}x)" if speed else ""
            print(f"  chat {chat['decode_tps_mean']:.1f} tok/s{extra}", flush=True)
    finally:
        print("\nRestoring original deploy config…", flush=True)
        try:
            restore = deepcopy(original_config)
            restore.setdefault("execution", {})
            restore["execution"]["cuda_devices"] = available_cuda_devices()
            if "," not in restore["execution"]["cuda_devices"]:
                restore.setdefault("performance", {})
                restore["performance"]["tensor_split"] = ""
            restart_with_config(args.backend_url, restore)
            wait_ready(args.infer_url, timeout_s=900)
        except Exception as exc:
            print(f"  WARNING: restore failed: {exc}", file=sys.stderr)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output or ROOT / "results" / f"mtp_chat_sweep_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "model_variant": args.model_variant,
        "infer_url": args.infer_url,
        "runs_per_case": args.runs,
        "cases": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"{'label':<14} {'n':>3} {'p':>5} {'chat tok/s':>11} {'vs base':>8}")
    print("-" * 48)
    for row in results:
        n = row["draft_n_max"] if row["mtp_enabled"] else "-"
        p = row["draft_p_min"] if row["mtp_enabled"] else "-"
        sp = row.get("speedup_vs_baseline")
        sp_s = f"{sp:.2f}x" if sp else ("1.00x" if not row["mtp_enabled"] else "-")
        print(
            f"{row['label']:<14} {str(n):>3} {str(p):>5} "
            f"{row['chat_decode_tps_mean']:>11.1f} {sp_s:>8}"
        )

    best = max(
        (r for r in results if r["mtp_enabled"]),
        key=lambda r: r["chat_decode_tps_mean"],
        default=None,
    )
    if best and baseline_tps:
        print()
        print(
            f"Best chat MTP: draft_n_max={best['draft_n_max']}, "
            f"draft_p_min={best['draft_p_min']} → "
            f"{best['chat_decode_tps_mean']:.1f} tok/s "
            f"({best['speedup_vs_baseline']:.2f}x vs baseline {baseline_tps:.1f})"
        )
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
