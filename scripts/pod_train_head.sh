#!/usr/bin/env bash
# Waits for extraction to finish, then trains the LoRA on the pod.
#
# Separate from pod_extract_chain.sh on purpose: that script is already running
# and bash reads a script incrementally, so editing it mid-flight would splice
# new bytes into the running interpreter. This watches from outside instead.
#
# Completion is judged by the index row count, not by process exit: a chain that
# died mid-split leaves a plausible-looking partial output, and training on a
# fraction of the data while believing it is the whole set is the kind of error
# that only shows up as a disappointing number much later.
set -u

PREP=${PREP:-/workspace/prep}
OUT=${OUT:-/workspace/out}
DATA=${DATA:-/workspace/data}
HEAD=${HEAD:-/workspace/models/mtp-bf16.gguf}
TRAIN_EXPECT=${TRAIN_EXPECT:-2191}
VAL_EXPECT=${VAL_EXPECT:-200}
# Wait for the full set, but if the extractor exits early accept a near-complete
# run rather than throwing away eight GPU-hours over a couple of lost samples.
# Set high enough that a genuinely truncated run (a crash halfway) still aborts.
TRAIN_MIN=${TRAIN_MIN:-$((TRAIN_EXPECT * 99 / 100))}
VAL_MIN=${VAL_MIN:-$((VAL_EXPECT * 95 / 100))}

rows() { [ -s "$1" ] && wc -l < "$1" || echo 0; }

echo "=== waiting for extraction (train $TRAIN_EXPECT, val $VAL_EXPECT) $(date -Is) ==="
while :; do
  t=$(rows "$OUT/train.h.index.jsonl"); v=$(rows "$OUT/val.h.index.jsonl")
  [ "$t" -ge "$TRAIN_EXPECT" ] && [ "$v" -ge "$VAL_EXPECT" ] && break
  if ! pgrep -f mtp_extract_h.py > /dev/null; then
    if [ "$t" -ge "$TRAIN_MIN" ] && [ "$v" -ge "$VAL_MIN" ]; then
      echo "extractor exited at train=$t/$TRAIN_EXPECT val=$v/$VAL_EXPECT -- "\
           "within tolerance (>=$TRAIN_MIN/$VAL_MIN), proceeding"
      break
    fi
    echo "ABORT: extractor gone with train=$t/$TRAIN_EXPECT val=$v/$VAL_EXPECT"\
         "(needed >=$TRAIN_MIN/$VAL_MIN)"
    exit 1
  fi
  sleep 120
done
echo "=== extraction complete: train=$t val=$v $(date -Is) ==="

# Same guard the extractor runs under: the states are only meaningful paired
# with the exact token stream they were decoded from.
( cd "$PREP" && md5sum -c --quiet manifest.md5 ) || { echo "ABORT: prep checksums moved"; exit 1; }

mkdir -p "$DATA"
for f in "$PREP"/*.tokens.bin "$OUT"/*.h.bf16 "$OUT"/*.h.index.jsonl; do ln -sf "$f" "$DATA"/; done

echo "=== training $(date -Is) ==="
TRAIN_CMD=(python3 /workspace/scripts/mtp_train_head.py
  --prep-dir "$DATA" --split train --val-split val
  --head-gguf "$HEAD" --device cuda
  --rank 32 --alpha 64 --accum 8 --epochs 1
  --rollout-lambda "${ROLL_LAMBDA:-1.0}" --rollout-k "${ROLL_K:-8}"
  --rollout-window "${ROLL_WINDOW:-64}" --rollout-per-span "${ROLL_PER_SPAN:-2}"
  --eval-samples 60 --eval-every "${EVAL_EVERY:-100}" --eval-rollout-k 8
  --log-every 20
  --out "${OUT:-/workspace/mtp_lora_r32_kstep.pt}")
[ -n "${RESUME:-}" ] && TRAIN_CMD+=(--resume "$RESUME")
"${TRAIN_CMD[@]}"
echo "=== training done $(date -Is) exit=$? ==="
