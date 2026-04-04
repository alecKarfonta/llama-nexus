#!/usr/bin/env python3
"""
TurboQuant Benchmark Report Generator

Parses outputs from the benchmark scripts and generates a comprehensive
markdown comparison report with go/no-go recommendations.

Usage:
    python3 scripts/turboquant_report.py \
        --speed results/turboquant_speed_*.csv \
        --quality results/turboquant_quality_*.json \
        --functional results/turboquant_functional.jsonl \
        --output docs/turboquant_benchmark_results.md
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "ppl_pass": 0.01,       # ΔPPL < 1% → PASS
    "ppl_warn": 0.03,       # ΔPPL 1-3% → WARN
    "kld_p999_pass": 0.01,  # 99.9th KLD < 0.01 → PASS
    "speed_pass": 0.90,     # ≥ 90% of q4_0 speed → PASS
    "speed_warn": 0.70,     # 70-90% → WARN
    "functional_pass": 0.80, # ≥ 80% success rate → PASS
}


def parse_speed_csv(path: str) -> list[dict]:
    """Parse llama-bench CSV output."""
    records = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        print(f"⚠️  Could not parse {path}: {e}", file=sys.stderr)
    return records


def parse_quality_json(path: str) -> list[dict]:
    """Parse quality benchmark JSON output."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not parse {path}: {e}", file=sys.stderr)
        return []


def parse_functional_jsonl(path: str) -> list[dict]:
    """Parse functional test JSONL output."""
    records = []
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except Exception as e:
        print(f"⚠️  Could not parse {path}: {e}", file=sys.stderr)
    return records


def verdict(kv_type: str, quality: list, speed: list, functional: list) -> str:
    """Determine PASS/WARN/FAIL for a KV type."""
    issues = []
    
    # Quality check
    q_match = [q for q in quality if q.get("kv_type") == kv_type and "error" not in q]
    if q_match:
        q = q_match[0]
        ppl = q.get("perplexity")
        kld_p999 = q.get("kld_p999")
        
        if kld_p999 and kld_p999 != "null":
            if float(kld_p999) > THRESHOLDS["kld_p999_pass"]:
                issues.append(f"KLD 99.9th ({kld_p999}) > {THRESHOLDS['kld_p999_pass']}")
    
    # Functional check
    f_match = [f for f in functional if f.get("kv_type") == kv_type]
    if f_match:
        avg_success = sum(f["success_ratio"] for f in f_match) / len(f_match)
        if avg_success < THRESHOLDS["functional_pass"]:
            issues.append(f"Functional success ({avg_success:.0%}) < {THRESHOLDS['functional_pass']:.0%}")
    
    if not issues:
        return "✅ PASS"
    elif len(issues) <= 1:
        return f"⚠️ WARN: {issues[0]}"
    else:
        return f"❌ FAIL: {'; '.join(issues)}"


def generate_report(speed_files, quality_files, functional_files, output_path):
    """Generate the full markdown report."""
    
    speed_data = []
    for f in (speed_files or []):
        speed_data.extend(parse_speed_csv(f))
    
    quality_data = []
    for f in (quality_files or []):
        quality_data.extend(parse_quality_json(f))
    
    functional_data = []
    for f in (functional_files or []):
        functional_data.extend(parse_functional_jsonl(f))
    
    lines = []
    lines.append("# TurboQuant Benchmark Results")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    
    # ── Verdicts ──────────────────────────────────────────────────────────
    kv_types_tested = set()
    for d in quality_data:
        kv_types_tested.add(d.get("kv_type"))
    for d in functional_data:
        kv_types_tested.add(d.get("kv_type"))
    
    if kv_types_tested:
        lines.append("## Verdict")
        lines.append("")
        lines.append("| KV Type | Status |")
        lines.append("|---------|--------|")
        for kv in sorted(kv_types_tested):
            v = verdict(kv, quality_data, speed_data, functional_data)
            lines.append(f"| `{kv}` | {v} |")
        lines.append("")
    
    # ── Speed Results ─────────────────────────────────────────────────────
    if speed_data:
        lines.append("## Speed Results")
        lines.append("")
        
        # Group by (type_k, type_v, n_prompt, flash_attn) and show avg tok/s
        cols = ["type_k", "type_v", "n_prompt", "flash_attn", "avg_ts", "stddev_ts"]
        available_cols = [c for c in cols if c in speed_data[0]]
        
        if available_cols:
            lines.append("| " + " | ".join(available_cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(available_cols)) + " |")
            for row in speed_data:
                values = [str(row.get(c, "N/A")) for c in available_cols]
                lines.append("| " + " | ".join(values) + " |")
        else:
            # Fallback: dump all columns from CSV
            if speed_data:
                all_cols = list(speed_data[0].keys())
                lines.append("| " + " | ".join(all_cols) + " |")
                lines.append("| " + " | ".join(["---"] * len(all_cols)) + " |")
                for row in speed_data[:50]:  # Cap at 50 rows
                    values = [str(row.get(c, "")) for c in all_cols]
                    lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    
    # ── Quality Results ───────────────────────────────────────────────────
    if quality_data:
        lines.append("## Quality Results (KL Divergence)")
        lines.append("")
        lines.append("Lower KLD = closer to FP16 baseline = better quality preservation.")
        lines.append("")
        lines.append("| KV Type | KLD Mean | KLD Max | KLD 99.9th | Perplexity | Duration |")
        lines.append("|---------|----------|---------|------------|------------|----------|")
        for q in quality_data:
            if "error" in q:
                lines.append(f"| `{q['kv_type']}` | ❌ {q['error']} | — | — | — | — |")
            else:
                lines.append(
                    f"| `{q['kv_type']}` "
                    f"| {q.get('kld_mean', 'N/A')} "
                    f"| {q.get('kld_max', 'N/A')} "
                    f"| {q.get('kld_p999', 'N/A')} "
                    f"| {q.get('perplexity', 'N/A')} "
                    f"| {q.get('duration_s', q.get('duration_seconds', 'N/A'))}s |"
                )
        lines.append("")
    
    # ── Functional Results ────────────────────────────────────────────────
    if functional_data:
        lines.append("## Functional Quality Results")
        lines.append("")
        lines.append("Live server tests measuring coherence, reasoning, and instruction following.")
        lines.append("")
        
        # Build pivot table: tests × kv_types
        kv_types = sorted(set(r["kv_type"] for r in functional_data))
        test_names = sorted(set(r["test"] for r in functional_data))
        
        header = "| Test | " + " | ".join(f"`{kv}`" for kv in kv_types) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(kv_types)) + " |"
        lines.append(header)
        lines.append(sep)
        
        for test in test_names:
            row = f"| {test} |"
            for kv in kv_types:
                match = [r for r in functional_data if r["test"] == test and r["kv_type"] == kv]
                if match:
                    ratio = match[0]["success_ratio"]
                    emoji = "✅" if ratio >= 0.8 else "⚠️" if ratio >= 0.5 else "❌"
                    row += f" {ratio:.0%} {emoji} |"
                else:
                    row += " N/A |"
            lines.append(row)
        lines.append("")
    
    # ── Thresholds Reference ──────────────────────────────────────────────
    lines.append("## Evaluation Thresholds")
    lines.append("")
    lines.append("| Metric | PASS | WARN | FAIL |")
    lines.append("|--------|------|------|------|")
    lines.append(f"| ΔPPL vs FP16 | < {THRESHOLDS['ppl_pass']:.0%} | {THRESHOLDS['ppl_pass']:.0%}–{THRESHOLDS['ppl_warn']:.0%} | > {THRESHOLDS['ppl_warn']:.0%} |")
    lines.append(f"| KLD 99.9th | < {THRESHOLDS['kld_p999_pass']} | — | ≥ {THRESHOLDS['kld_p999_pass']} |")
    lines.append(f"| Speed vs q4_0 | ≥ {THRESHOLDS['speed_pass']:.0%} | {THRESHOLDS['speed_warn']:.0%}–{THRESHOLDS['speed_pass']:.0%} | < {THRESHOLDS['speed_warn']:.0%} |")
    lines.append(f"| Functional tests | ≥ {THRESHOLDS['functional_pass']:.0%} | — | < {THRESHOLDS['functional_pass']:.0%} |")
    lines.append("")
    
    report = "\n".join(lines)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"✅ Report written to {output_path}")
    print(f"   ({len(speed_data)} speed records, {len(quality_data)} quality records, {len(functional_data)} functional records)")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Benchmark Report Generator")
    parser.add_argument("--speed", nargs="*", help="Speed benchmark CSV files from llama-bench")
    parser.add_argument("--quality", nargs="*", help="Quality benchmark JSON files from llama-perplexity")
    parser.add_argument("--functional", nargs="*", help="Functional test JSONL files")
    parser.add_argument("--output", default="docs/turboquant_benchmark_results.md",
                        help="Output markdown report path")
    args = parser.parse_args()
    
    if not any([args.speed, args.quality, args.functional]):
        parser.error("At least one of --speed, --quality, or --functional is required")
    
    generate_report(args.speed, args.quality, args.functional, args.output)


if __name__ == "__main__":
    main()
