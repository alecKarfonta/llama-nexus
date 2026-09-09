#!/usr/bin/env bash
# [TIME] | INFO | POD_TRAIN_KSTEP | k-step rollout loss, continue from r=32 LoRA.
# Extraction is already done locally; this pod only needs the BF16 head GGUF,
# the staged prep+states, torch, and the saved adapter.
set -u

PREP=${PREP:-/workspace/prep}
STATES=${STATES:-/workspace/states}
DATA=${DATA:-/workspace/data}
HEAD=${HEAD:-/workspace/models/mtp-bf16.gguf}
RESUME=${RESUME:-/workspace/mtp_lora_r32.pt}
OUT=${OUT:-/workspace/mtp_lora_r32_kstep.pt}
LR=${LR:-}
LR_SCHEDULE=${LR_SCHEDULE:-onecycle}
ROLL_TEACHER_P=${ROLL_TEACHER_P:-}
ROLL_TEACHER_P_END=${ROLL_TEACHER_P_END:-}
ABORT_ELEN=${ABORT_ELEN:-}
ABORT_AT=${ABORT_AT:-}
ROLL_P_MIN=${ROLL_P_MIN:-}
ROLL_CONF_MARGIN=${ROLL_CONF_MARGIN:-}
RESUME_ELEN=${RESUME_ELEN:-}

echo "=== k-step train setup $(date -Is) ==="
mkdir -p "$DATA"

if [ -s "$PREP/manifest.md5" ]; then
  # manifest lists the full prep subset; only the token streams are required
  # for training (indices live in h.index.jsonl). Verify tokens if present.
  if [ -f "$PREP/train.tokens.bin" ] && [ -f "$PREP/val.tokens.bin" ]; then
    echo "prep tokens present ($(wc -c < "$PREP/train.tokens.bin") train bytes)"
  else
    ( cd "$PREP" && md5sum -c --quiet manifest.md5 ) \
      || { echo "ABORT: prep checksums failed"; exit 1; }
    echo "prep verified"
  fi
fi

for f in "$PREP"/*.tokens.bin "$STATES"/*.h.bf16 "$STATES"/*.h.index.jsonl; do
  [ -e "$f" ] || continue
  ln -sf "$f" "$DATA"/
done

for need in train.tokens.bin train.h.bf16 train.h.index.jsonl \
            val.tokens.bin val.h.bf16 val.h.index.jsonl "$HEAD" "$RESUME"; do
  if [ ! -e "$DATA/$need" ] && [ ! -e "$need" ]; then
    echo "ABORT: missing $need"
    exit 1
  fi
done

echo "=== training $(date -Is) ==="
cmd=(python3 -u /workspace/scripts/mtp_train_head.py
  --prep-dir "$DATA" --split train --val-split val
  --head-gguf "$HEAD" --device cuda
  --resume "$RESUME"
  --rank 32 --alpha 64 --accum 8 --epochs 1
  --lr-schedule "$LR_SCHEDULE"
  --rollout-lambda 1.0 --rollout-k 8 --rollout-window 64
  --rollout-per-span 2 --rollout-batch 8
  --eval-samples 60 --eval-every 100 --eval-rollout-k 8
  --log-every 20
  --out "$OUT")
[ -n "$LR" ] && cmd+=(--lr "$LR")
[ -n "$ROLL_TEACHER_P" ] && cmd+=(--rollout-teacher-p "$ROLL_TEACHER_P")
[ -n "$ROLL_TEACHER_P_END" ] && cmd+=(--rollout-teacher-p-end "$ROLL_TEACHER_P_END")
[ -n "$ABORT_ELEN" ] && cmd+=(--abort-elen "$ABORT_ELEN")
[ -n "$ABORT_AT" ] && cmd+=(--abort-at $ABORT_AT)
[ -n "$ROLL_P_MIN" ] && cmd+=(--rollout-p-min "$ROLL_P_MIN")
[ -n "$ROLL_CONF_MARGIN" ] && cmd+=(--rollout-conf-margin "$ROLL_CONF_MARGIN")
[ -n "$ROLL_CONF_LAMBDA" ] && cmd+=(--rollout-conf-lambda "$ROLL_CONF_LAMBDA")
[ -n "$RESUME_ELEN" ] && cmd+=(--resume-elen "$RESUME_ELEN")
"${cmd[@]}" 2>&1 | tee /workspace/train_kstep.log
