#!/usr/bin/env bash
# [TIME] | INFO | DIAGNOSE_PMIN | stock vs e1 E[len_pmin] at p_min 0 / 0.80 / 0.85
set -u

ROOT=/home/alec/git/llama-nexus
EVAL=/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-eval
E1=/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-h-pod/mtp_lora_r32_kstep.pt
BUILD=/mnt/comfyui-models/llama-nexus/build
STOCK_BF16=$BUILD/mtp-bf16.gguf
OUT=/tmp/pmin_diagnostic
mkdir -p "$OUT"

run_one() {
  local label=$1 adapter=$2 pmin=$3
  local extra=()
  [ -n "$adapter" ] && extra=(--adapter "$adapter")
  [ "$pmin" != "0" ] && extra+=(--p-min "$pmin")
  OMP_NUM_THREADS=44 python3 -u "$ROOT/scripts/mtp_train_head.py" --rollout --fp32 --device cpu \
    --prep-dir "$EVAL" --head-gguf "$STOCK_BF16" --split val \
    --per-span 3 --k-max 16 --window 64 --batch 8 "${extra[@]}" \
    > "$OUT/${label}_p${pmin}.json" 2> "$OUT/${label}_p${pmin}.log"
}

for p in 0 0.80 0.85; do
  run_one stock "" "$p"
  run_one e1 "$E1" "$p"
done

python3 - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out, "*.json"))):
    base = os.path.basename(path).replace(".json", "")
    t = open(path).read()
    if "{" not in t:
        continue
    rep = json.loads(t[t.index("{"):])["rollout"]
    rows.append((base, rep.get("expected_accepted_len"), rep.get("expected_accepted_len_pmin", rep.get("expected_accepted_len"))))
print(f"{'label':<20} {'E[len]':>8} {'E[len_pmin]':>12}")
for label, raw, pmin in rows:
    print(f"{label:<20} {raw:8.4f} {pmin:12.4f}")
PY

echo "diagnostic written to $OUT"
