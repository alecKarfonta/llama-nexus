#!/usr/bin/env bash
# [TIME] | INFO | BENCH_WIDE_TPS | wider STRUCT + Cursor TPS/acc-round for e1 vs stock.
set -u

ROOT=/home/alec/git/llama-nexus
RUNS=${RUNS:-5}
SESSIONS=${SESSIONS:-20}
TURNS=${TURNS:-5}
OUT=${OUT:-/tmp/wide_tps_report.json}
PMIN=0.85

say() { echo "[$(date -Is)] $*"; }

say "STRUCT sweep kstep @ $PMIN ($RUNS runs/workload)"
python3 "$ROOT/scripts/sweep_draft_knobs.py" --head kstep --p-min "$PMIN" \
  --runs "$RUNS" --no-restore --out /tmp/wide_struct_kstep.json

say "STRUCT sweep stock @ $PMIN ($RUNS runs/workload)"
python3 "$ROOT/scripts/sweep_draft_knobs.py" --head stock --p-min "$PMIN" \
  --runs "$RUNS" --no-restore --out /tmp/wide_struct_stock.json

say "Cursor replay A/B ($SESSIONS sessions x $TURNS turns)"
python3 "$ROOT/scripts/ab_cursor_draft.py" --sessions "$SESSIONS" \
  --turns-per-session "$TURNS" --out /tmp/cursor_ab_wide.json

python3 - "$OUT" <<PY
import json, sys
out = sys.argv[1]
def load(p):
    return json.load(open(p))
def struct_summary(path, label):
    rows = load(path)
    pt = rows[0] if rows else {}
    per = pt.get("workloads") or {}
    tok = {w: per[w]["tok_s"] for w in per}
    apr = {w: per[w]["accepted_per_round"] for w in per}
    struct = sum(apr.get(w, 0) for w in ("code", "json")) / 2 if per else 0
    all_tok = sum(tok.values()) / len(tok) if tok else 0
    return {"label": label, "p_min": pt.get("p_min"), "workloads": per,
            "struct_acc_round": struct, "avg_tok_s": all_tok, "tok_s": tok, "acc_round": apr}
sk = struct_summary("/tmp/wide_struct_kstep.json", "e1_kstep")
st = struct_summary("/tmp/wide_struct_stock.json", "stock")
cur = load("/tmp/cursor_ab_wide.json")
report = {
    "struct": {"e1": sk, "stock": st},
    "cursor": cur,
    "winner_cursor": cur.get("winner"),
}
json.dump(report, open(out, "w"), indent=2)
print(json.dumps(report, indent=2))
print(f"\nwrote {out}")
PY

# Restore e1 winner from cursor A/B (ab_cursor_draft ships winner already)
say "done — report at $OUT"
