#!/usr/bin/env python3
"""
MTP benchmark report generator.

Reads JSONL from mtp_bench.py, picks recommended tunables (acceptance ≥ 70%,
best decode tok/s), and writes markdown + UI defaults JSON.

Usage:
  python3 scripts/mtp_report.py --bench results/mtp_bench_*.jsonl
  python3 scripts/mtp_report.py --bench results/mtp_bench.jsonl --write-ui-defaults
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_UI_DEFAULTS_PATH = _REPO_ROOT / "frontend" / "src" / "config" / "mtpRecommendedDefaults.json"
_RESULTS_DEFAULTS_PATH = _REPO_ROOT / "results" / "mtp_recommended_defaults.json"

ACCEPTANCE_TARGET = 0.70


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                continue
            if obj.get("status") == "ok":
                records.append(obj)
    return records


def _decode_tps(record: Dict[str, Any]) -> Optional[float]:
    try:
        return float(record["metrics"]["decode"]["decode_tokens_per_second"])
    except (KeyError, TypeError, ValueError):
        return None


def _acceptance(record: Dict[str, Any]) -> Optional[float]:
    try:
        acc = record["metrics"]["acceptance"]
        rate = acc.get("acceptance_rate_last") or acc.get("acceptance_rate_mean")
        return float(rate) if rate is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def _baseline_tps_by_group(records: List[Dict[str, Any]]) -> Dict[Tuple, float]:
    out: Dict[Tuple, float] = {}
    for r in records:
        if r.get("mtp_enabled"):
            continue
        tps = _decode_tps(r)
        if tps is None:
            continue
        key = (r.get("model_id"), r.get("parallel_slots"), r.get("cache_type_k"))
        out[key] = max(out.get(key, 0.0), tps)
    return out


def pick_recommendations(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per model family, pick MTP settings with best decode TPS among acceptance ≥ target."""
    baselines = _baseline_tps_by_group(records)
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if not r.get("mtp_enabled"):
            continue
        by_family[str(r.get("family") or "default")].append(r)

    recommendations: Dict[str, Dict[str, Any]] = {}
    for family, group in by_family.items():
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for r in group:
            acc = _acceptance(r)
            tps = _decode_tps(r)
            if acc is None or tps is None or acc < ACCEPTANCE_TARGET:
                continue
            key = (r.get("model_id"), r.get("parallel_slots"), r.get("cache_type_k"))
            base = baselines.get(key)
            speedup = (tps / base) if base and base > 0 else None
            candidates.append((tps, {**r, "speedup_vs_baseline": speedup, "acceptance_rate": acc}))

        if not candidates:
            recommendations[family] = {
                "enabled": False,
                "note": f"No sweep met {ACCEPTANCE_TARGET:.0%} acceptance — keep MTP disabled or re-tune.",
            }
            continue

        _, best = max(candidates, key=lambda x: x[0])
        recommendations[family] = {
            "enabled": True,
            "draft_n_max": int(best.get("draft_n_max", 3)),
            "draft_n_min": int(best.get("draft_n_min", 0)),
            "draft_p_min": float(best.get("draft_p_min", 0.75)),
            "acceptance_rate": best.get("acceptance_rate"),
            "decode_tokens_per_second": _decode_tps(best),
            "speedup_vs_baseline": best.get("speedup_vs_baseline"),
            "model_id": best.get("model_id"),
            "parallel_slots": best.get("parallel_slots"),
        }
    return recommendations


def generate_markdown(
    records: List[Dict[str, Any]],
    recommendations: Dict[str, Dict[str, Any]],
    bench_path: Optional[Path],
) -> str:
    lines = [
        "# MTP Benchmark Results",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]
    if bench_path:
        lines.append(f"Source: `{bench_path}`")
        lines.append("")

    # Summary table
    lines.extend(
        [
            "## Summary",
            "",
            "| Model | Parallel | MTP | draft_n_max | draft_p_min | Accept % | Decode tok/s | Speedup |",
            "|-------|----------|-----|-------------|-------------|----------|--------------|---------|",
        ]
    )
    baselines = _baseline_tps_by_group(records)
    for r in sorted(records, key=lambda x: (x.get("model_id", ""), x.get("case_id", ""))):
        acc = _acceptance(r)
        tps = _decode_tps(r)
        key = (r.get("model_id"), r.get("parallel_slots"), r.get("cache_type_k"))
        base = baselines.get(key)
        speedup = f"{tps / base:.2f}×" if tps and base and base > 0 and r.get("mtp_enabled") else "—"
        acc_s = f"{acc * 100:.1f}" if acc is not None else "—"
        tps_s = f"{tps:.1f}" if tps is not None else "—"
        lines.append(
            f"| {r.get('model_id')} | {r.get('parallel_slots')} | "
            f"{'yes' if r.get('mtp_enabled') else 'no'} | "
            f"{r.get('draft_n_max', '—')} | {r.get('draft_p_min', '—')} | "
            f"{acc_s} | {tps_s} | {speedup} |"
        )

    lines.extend(["", "## Recommended defaults (≥ 70% acceptance)", ""])
    for family, rec in sorted(recommendations.items()):
        lines.append(f"### `{family}`")
        if not rec.get("enabled"):
            lines.append(f"- {rec.get('note', 'MTP disabled')}")
        else:
            lines.append(
                f"- Enable MTP: `draft_n_max={rec['draft_n_max']}`, "
                f"`draft_p_min={rec['draft_p_min']}`"
            )
            if rec.get("acceptance_rate") is not None:
                lines.append(f"- Acceptance: {rec['acceptance_rate'] * 100:.1f}%")
            if rec.get("speedup_vs_baseline"):
                lines.append(f"- Speedup vs baseline: {rec['speedup_vs_baseline']:.2f}×")
        lines.append("")

    lines.append("## Methodology")
    lines.extend(
        [
            "",
            "- Harness: `scripts/mtp_bench.py` (llama-server HTTP, not llama-bench — MTP flags unsupported there).",
            "- Metrics: decode/prompt tok/s, TTFT, log acceptance rate, VRAM delta.",
            "- Quality: `scripts/mtp_quality_diff.py` fixed-seed coding prompts (MTP off vs on).",
            "",
        ]
    )
    return "\n".join(lines)


def write_defaults_json(recommendations: Dict[str, Dict[str, Any]], paths: List[Path]) -> None:
    payload = {
        "_comment": "Generated by scripts/mtp_report.py — family keys match inferMtpFamily() in mtpModelSettings.ts",
        "default": {
            "enabled": False,
            "draft_n_max": 3,
            "draft_n_min": 0,
            "draft_p_min": 0.75,
        },
    }
    for family, rec in recommendations.items():
        if rec.get("enabled"):
            payload[family] = {
                "enabled": True,
                "draft_n_max": rec["draft_n_max"],
                "draft_n_min": rec.get("draft_n_min", 0),
                "draft_p_min": rec["draft_p_min"],
            }
        else:
            payload[family] = {
                "enabled": False,
                "draft_n_max": 3,
                "draft_n_min": 0,
                "draft_p_min": 0.75,
                "note": rec.get("note"),
            }

    text = json.dumps(payload, indent=2) + "\n"
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote defaults: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MTP benchmark report")
    parser.add_argument("--bench", required=True, help="JSONL path or glob")
    parser.add_argument("--output", default=str(_REPO_ROOT / "docs" / "mtp_benchmark_results.md"))
    parser.add_argument("--write-ui-defaults", action="store_true", help="Write mtpRecommendedDefaults.json")
    args = parser.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.bench))
    if not paths:
        parser.error(f"No files match: {args.bench}")

    records: List[Dict[str, Any]] = []
    for p in paths:
        records.extend(load_jsonl(p))

    if not records:
        print("No successful benchmark records found.", file=sys.stderr)
        return 1

    recommendations = pick_recommendations(records)
    md = generate_markdown(records, recommendations, paths[0] if len(paths) == 1 else None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote report: {out}")

    if args.write_ui_defaults:
        write_defaults_json(
            recommendations,
            [_UI_DEFAULTS_PATH, _RESULTS_DEFAULTS_PATH],
        )
    else:
        write_defaults_json(recommendations, [_RESULTS_DEFAULTS_PATH])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
