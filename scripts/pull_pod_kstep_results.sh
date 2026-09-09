#!/usr/bin/env bash
# Pull k-step adapter, run final k=16 rollout gate, merge+quantize if it wins.
set -u

ROOT=/home/alec/git/llama-nexus
BUILD=/mnt/comfyui-models/llama-nexus/build
POD=${POD:-$(cat /tmp/pod_kstep_id 2>/dev/null)}
SSHCFG=${SSHCFG:-/tmp/pod_kstep_ssh_config}
HOSTALIAS=${HOSTALIAS:-ksteppod}
DEST=${DEST:-/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-h-pod}
ADAPTER_REMOTE=${ADAPTER_REMOTE:-/workspace/mtp_lora_r32_kstep_e2.pt}
ADAPTER_LOCAL=${ADAPTER_LOCAL:-$DEST/$(basename "$ADAPTER_REMOTE")}
EVAL=/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-eval
STOCK_ELEN=${STOCK_ELEN:-5.72}
STEP1_MIN=${STEP1_MIN:-0.85}
STOCK_BF16=$BUILD/mtp-bf16.gguf

SSH="ssh -F $SSHCFG -o ConnectTimeout=20 -n $HOSTALIAS"
say() { echo "[$(date -Is)] $*"; }
running() { $SSH "pgrep -f '[${1:0:1}]${1:1}' > /dev/null" 2>/dev/null; }

mkdir -p "$DEST"
[ -f "$SSHCFG" ] || { say "ABORT: no $SSHCFG — set SSHCFG or run launch_pod_kstep_train.sh first"; exit 1; }

say "waiting for $ADAPTER_REMOTE"
while ! $SSH "test -s $ADAPTER_REMOTE" 2>/dev/null; do
  if ! running pod_train_kstep.sh && ! running mtp_train_head.py; then
    say "ABORT: trainer gone and no adapter"
    exit 1
  fi
  sleep 180
done
while running mtp_train_head.py; do sleep 60; done
say "trainer finished"

scp -F "$SSHCFG" "$HOSTALIAS:$ADAPTER_REMOTE" "$ADAPTER_LOCAL" \
  || { say "ABORT: adapter pull failed"; exit 1; }
scp -F "$SSHCFG" "$HOSTALIAS:/workspace/train_kstep.log" "$DEST/train_kstep.log" 2>/dev/null || true
say "adapter local: $(ls -la "$ADAPTER_LOCAL" | awk '{print $5" bytes"}')"

OUT=/tmp/kstep_final_gate
mkdir -p "$OUT"

say "stock baseline rollout (k=16)"
OMP_NUM_THREADS=44 python3 -u "$ROOT/scripts/mtp_train_head.py" --rollout --fp32 --device cpu \
  --prep-dir "$EVAL" --head-gguf "$STOCK_BF16" --split val \
  --per-span 3 --k-max 16 --window 64 --batch 8 \
  > "$OUT/stock.json" 2> "$OUT/stock.log"

say "k-step adapter rollout (k=16)"
OMP_NUM_THREADS=44 python3 -u "$ROOT/scripts/mtp_train_head.py" --rollout --fp32 --device cpu \
  --prep-dir "$EVAL" --head-gguf "$STOCK_BF16" --adapter "$ADAPTER_LOCAL" --split val \
  --per-span 3 --k-max 16 --window 64 --batch 8 \
  > "$OUT/kstep.json" 2> "$OUT/kstep.log"

python3 - "$OUT/stock.json" "$OUT/kstep.json" "$STOCK_ELEN" "$STEP1_MIN" <<'PY'
import json, sys
def load(p):
    t=open(p).read(); return json.loads(t[t.index('{'):])['rollout']
s, k = load(sys.argv[1]), load(sys.argv[2])
bar, step1_min = float(sys.argv[3]), float(sys.argv[4])
step1 = k['steps'][0]['acc']
elen = k['expected_accepted_len']
print(f"stock E[len]={s['expected_accepted_len']:.4f}  kstep E[len]={elen:.4f}  step1={step1:.4f}")
ok = elen > bar and step1 >= step1_min
print(f"gate: {'PASS' if ok else 'FAIL'} (need E[len]>{bar} and step1>={step1_min})")
sys.exit(0 if ok else 1)
PY
GATE_RC=$?

if [ "$GATE_RC" -eq 0 ]; then
  say "gate passed — merge + quantize"
  MERGED=$BUILD/mtp-lora-kstep-bf16.gguf
  QUANT=$BUILD/mtp-lora-kstep-Q4_K_M.gguf
  python3 "$ROOT/scripts/mtp_train_head.py" --merge "$ADAPTER_LOCAL" \
    --export-gguf "$MERGED" --head-gguf "$STOCK_BF16" --device cpu
  "$BUILD/llama.cpp/build-quantize/bin/llama-quantize" "$MERGED" "$QUANT" Q4_K_M
  say "merged $MERGED quant $QUANT — swap live draft manually if desired"
else
  say "gate failed — leaving stock head live"
fi

if [ "${DELETE_POD:-1}" = 1 ] && [ -n "${POD:-}" ]; then
  export PATH="$HOME/.local/bin:$PATH"
  export RUNPOD_API_KEY="$(grep -m1 '^RUNPOD_API_KEY=' "$ROOT/.env" | cut -d= -f2- | tr -d '"'\''\r' | xargs)"
  runpodctl pod delete "$POD" < /dev/null && say "pod $POD deleted" \
    || say "WARNING: pod delete failed"
fi
exit "$GATE_RC"
