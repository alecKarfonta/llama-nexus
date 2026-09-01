#!/usr/bin/env python3
"""
[TIME] | INFO | SCRIPTS.ACCEPTANCE_LOG | trend MTP draft acceptance over time
from llama-server logs, so the dataset's value is measurable: acceptance on
code/JSON should rise after the Cursor-tuned head replaces the stock sidecar.

Each `docker logs` line like:
  ... draft acceptance = 0.95055 (  346 accepted /   364 generated), mean len = 8.69
becomes a timestamped sample. Acceptance samples are per-request (server logs
one per finished request with speculative decoding on).

Modes:
  tail      — print latest sample (one-shot, for scripts/cron)
  trend     — parse full logs, aggregate per day, optionally append to a CSV
  live      — stream docker logs and append to CSV continuously

Usage:
  python3 scripts/acceptance_log.py tail
  python3 scripts/acceptance_log.py trend --csv /data/captures/acceptance.csv
  python3 scripts/acceptance_log.py live  --csv /data/captures/acceptance.csv
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict

# llama-server log line format (speculative decoding summary per request)
LINE_RE = re.compile(
    r"(?P<acc>[01]\.\d+) \(\s*(?P<accepted>\d+) accepted /\s*(?P<generated>\d+) generated\)"
    r", mean len =\s*(?P<meanlen>[\d.]+)"
)
TS_RE = re.compile(r"\d{2}:(?P<ts>\d{4}-\d{2}-\d{2})")  # llama-server: hh.mm.ss.uuu yyyy-mm-dd


def parse_logs(since=None):
    """Yield (date, acc, accepted, generated, meanlen) from docker logs."""
    cmd = ["docker", "logs", "llamacpp-api"]
    if since:
        cmd += ["--since", since]
    try:
        # [TIME] | INFO | SCRIPTS.ACCEPTANCE_LOG | llama-server logs to stderr;
        # merge both streams or acceptance lines never appear
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        print(f"docker logs failed: {e}", file=sys.stderr)
        return
    for line in (proc.stderr + proc.stdout).splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        ts = TS_RE.match(line)
        yield (
            ts.group("ts") if ts else time.strftime("%Y-%m-%d"),
            float(m.group("acc")),
            int(m.group("accepted")),
            int(m.group("generated")),
            float(m.group("meanlen")),
        )


def write_csv_row(path, row):
    import csv
    import os
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["date", "acceptance", "accepted", "generated", "mean_len"])
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tail", "trend", "live"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--since", default=None, help="docker logs --since value")
    args = ap.parse_args()

    samples = list(parse_logs(args.since))

    if args.mode == "tail":
        if not samples:
            print("no acceptance lines in logs")
            return
        _, acc, accd, gen, ml = samples[-1]
        print(f"acceptance={acc:.3f} accepted={accd} generated={gen} mean_len={ml:.2f}")
        return

    if args.mode == "trend":
        by_day = defaultdict(list)
        for date, acc, accd, gen, ml in samples:
            by_day[date].append((acc, accd, gen, ml))
        if not by_day:
            print("no acceptance samples")
            return
        print(f"{'date':12s} {'n':>5s} {'acc_avg':>8s} {'mean_len':>9s} {'tokens':>8s}")
        for date in sorted(by_day):
            rows = by_day[date]
            n = len(rows)
            acc_avg = sum(r[0] for r in rows) / n
            ml_avg = sum(r[3] for r in rows) / n
            tokens = sum(r[2] for r in rows)
            print(f"{date:12s} {n:5d} {acc_avg:8.3f} {ml_avg:9.2f} {tokens:8d}")
            if args.csv:
                write_csv_row(args.csv, [date, f"{acc_avg:.4f}", n, ml_avg, tokens])
        return

    if args.mode == "live":
        print("streaming (ctrl-c to stop)...")
        proc = subprocess.Popen(
            ["docker", "logs", "-f", "llamacpp-api"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            for line in proc.stdout:
                m = LINE_RE.search(line)
                if m and args.csv:
                    ts = TS_RE.match(line)
                    write_csv_row(args.csv, [
                        ts.group("ts") if ts else time.strftime("%Y-%m-%d"),
                        m.group("acc"), m.group("accepted"),
                        m.group("generated"), m.group("meanlen"),
                    ])
        except KeyboardInterrupt:
            proc.terminate()
        return


if __name__ == "__main__":
    main()
