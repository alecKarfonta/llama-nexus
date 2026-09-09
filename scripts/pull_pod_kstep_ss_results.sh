#!/usr/bin/env bash
# Pull SS adapter, offline k=16 gate, merge+quantize, live STRUCT gate vs stock 7.38.
set -u

ROOT=/home/alec/git/llama-nexus
BUILD=/mnt/comfyui-models/llama-nexus/build
POD=${POD:-$(cat /tmp/pod_kstep_ss_id 2>/dev/null)}
SSHCFG=${SSHCFG:-/tmp/pod_kstep_ss_ssh_config}
HOSTALIAS=${HOSTALIAS:-kstepsspod}
DEST=${DEST:-/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-h-pod}
ADAPTER_REMOTE=${ADAPTER_REMOTE:-/workspace/mtp_lora_r32_kstep_ss.pt}
ADAPTER_LOCAL=${ADAPTER_LOCAL:-$DEST/$(basename "$ADAPTER_REMOTE")}
EVAL=/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-eval
STOCK_ELEN=${STOCK_ELEN:-5.72}
STEP1_MIN=${STEP1_MIN:-0.85}
LIVE_STRUCT_BAR=${LIVE_STRUCT_BAR:-7.38}
STOCK_BF16=$BUILD/mtp-bf16.gguf
MERGED=$BUILD/mtp-lora-kstep-ss-bf16.gguf
QUANT=$BUILD/Qwen3.8-Flash-Next-MTP-kstep-ss-Q4_K_M.gguf
QUANT_LINK=$BUILD/mtp-lora-kstep-ss-Q4_K_M.gguf

SSH="ssh -F $SSHCFG -o ConnectTimeout=20 -n $HOSTALIAS"
say() { echo "[$(date -Is)] $*"; }
running() { $SSH "pgrep -f '[${1:0:1}]${1:1}' > /dev/null" 2>/dev/null; }

mkdir -p "$DEST"
[ -f "$SSHCFG" ] || { say "ABORT: no $SSHCFG — run launch_pod_kstep_ss.sh first"; exit 1; }

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
scp -F "$SSHCFG" "$HOSTALIAS:/workspace/train_kstep.log" "$DEST/train_kstep_ss.log" 2>/dev/null || true
say "adapter local: $(ls -la "$ADAPTER_LOCAL" | awk '{print $5" bytes"}')"

OUT=/tmp/kstep_ss_final_gate
mkdir -p "$OUT"

say "stock baseline rollout (k=16)"
OMP_NUM_THREADS=44 python3 -u "$ROOT/scripts/mtp_train_head.py" --rollout --fp32 --device cpu \
  --prep-dir "$EVAL" --head-gguf "$STOCK_BF16" --split val \
  --per-span 3 --k-max 16 --window 64 --batch 8 \
  > "$OUT/stock.json" 2> "$OUT/stock.log"

say "SS adapter rollout (k=16, own-token)"
OMP_NUM_THREADS=44 python3 -u "$ROOT/scripts/mtp_train_head.py" --rollout --fp32 --device cpu \
  --prep-dir "$EVAL" --head-gguf "$STOCK_BF16" --adapter "$ADAPTER_LOCAL" --split val \
  --per-span 3 --k-max 16 --window 64 --batch 8 \
  > "$OUT/kstep_ss.json" 2> "$OUT/kstep_ss.log"

python3 - "$OUT/stock.json" "$OUT/kstep_ss.json" "$STOCK_ELEN" "$STEP1_MIN" <<'PY'
import json, sys
def load(p):
    t=open(p).read(); return json.loads(t[t.index('{'):])['rollout']
s, k = load(sys.argv[1]), load(sys.argv[2])
bar, step1_min = float(sys.argv[3]), float(sys.argv[4])
step1 = k['steps'][0]['acc']
elen = k['expected_accepted_len']
print(f"stock E[len]={s['expected_accepted_len']:.4f}  ss E[len]={elen:.4f}  step1={step1:.4f}")
ok = elen > bar and step1 >= step1_min
print(f"offline gate: {'PASS' if ok else 'FAIL'} (need E[len]>{bar} and step1>={step1_min})")
sys.exit(0 if ok else 1)
PY
GATE_RC=$?

if [ "$GATE_RC" -ne 0 ]; then
  say "offline gate failed — not merging or sweeping live"
  [ "${DELETE_POD:-1}" = 1 ] && [ -n "${POD:-}" ] && runpodctl pod delete "$POD" < /dev/null || true
  exit 1
fi

say "offline gate passed — merge + quantize (new name, do not overwrite e1)"
python3 "$ROOT/scripts/mtp_train_head.py" --merge "$ADAPTER_LOCAL" \
  --export-gguf "$MERGED" --head-gguf "$STOCK_BF16" --device cpu
"$BUILD/llama.cpp/build-quantize/bin/llama-quantize" "$MERGED" "$QUANT" Q4_K_M
ln -sf "$(basename "$QUANT")" "$QUANT_LINK"
say "quantized $QUANT"
MODELS=/mnt/comfyui-models/llama-nexus/gpt_oss_models
if cp -f "$QUANT" "$MODELS/$(basename "$QUANT")" 2>/dev/null; then
  say "staged $(basename "$QUANT") -> $MODELS"
else
  sudo cp -f "$QUANT" "$MODELS/$(basename "$QUANT")"
  sudo chown "$(id -un):$(id -gn)" "$MODELS/$(basename "$QUANT")" 2>/dev/null || true
  say "staged $(basename "$QUANT") -> $MODELS (sudo)"
fi

SWEEP_OUT=/tmp/kstep_ss_live_sweep.json
say "live STRUCT sweep at p_min 0.80 and 0.85 (bar=$LIVE_STRUCT_BAR vs stock)"
python3 "$ROOT/scripts/sweep_draft_knobs.py" --head kstep_ss \
  --p-min 0.80 0.85 --runs 2 --no-restore --out "$SWEEP_OUT" \
  || { say "WARNING: sweep failed"; exit 1; }

python3 - "$SWEEP_OUT" "$LIVE_STRUCT_BAR" <<'PY'
import json, sys
results = json.load(open(sys.argv[1]))
bar = float(sys.argv[2])
best = 0.0
best_p = None
for pt in results:
    per = pt.get("workloads") or {}
    vals = [per[w]["accepted_per_round"] for w in ("code", "json") if w in per]
    if not vals:
        continue
    struct = sum(vals) / len(vals)
    print(f"p_min={pt['p_min']:.2f} STRUCT acc/round={struct:.2f}")
    if struct > best:
        best, best_p = struct, pt["p_min"]
print(f"best STRUCT={best:.2f} @ p_min={best_p}  (bar>{bar})")
sys.exit(0 if best > bar else 1)
PY
LIVE_RC=$?

if [ "$LIVE_RC" -eq 0 ]; then
  say "LIVE GATE PASS — ship kstep_ss @ best p_min from sweep"
  BEST_P=$(python3 - "$SWEEP_OUT" <<'PY'
import json, sys
results = json.load(open(sys.argv[1]))
best = (-1.0, None)
for pt in results:
    per = pt.get("workloads") or {}
    vals = [per[w]["accepted_per_round"] for w in ("code", "json") if w in per]
    if not vals: continue
    s = sum(vals)/len(vals)
    if s > best[0]: best = (s, pt["p_min"])
print(best[1])
PY
)
  python3 - <<PY
import json, os, urllib.request
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8700")
API_KEY = os.getenv("API_KEY", "placeholder-api-key")
def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BACKEND}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)
api("PUT", "/api/v1/service/config", {
    "speculative": {"model_draft": "Qwen3.8-Flash-Next-MTP-kstep-ss-Q4_K_M.gguf", "gpu_layers_draft": 999},
    "mtp": {"draft_p_min": float("$BEST_P"), "draft_n_min": 0},
})
api("POST", "/api/v1/service/restart")
print(f"shipped kstep_ss p_min=$BEST_P")
PY
else
  say "LIVE GATE FAIL — keeping e1 kstep @ 0.85 live"
  python3 - <<'PY'
import json, os, urllib.request
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8700")
API_KEY = os.getenv("API_KEY", "placeholder-api-key")
def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BACKEND}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)
api("PUT", "/api/v1/service/config", {
    "speculative": {"model_draft": "Qwen3.8-Flash-Next-MTP-kstep-Q4_K_M.gguf", "gpu_layers_draft": 999},
    "mtp": {"draft_p_min": 0.85, "draft_n_min": 0},
})
api("POST", "/api/v1/service/restart")
print("restored e1 kstep @ p_min=0.85")
PY
fi

if [ "${DELETE_POD:-1}" = 1 ] && [ -n "${POD:-}" ]; then
  export PATH="$HOME/.local/bin:$PATH"
  export RUNPOD_API_KEY="$(grep -m1 '^RUNPOD_API_KEY=' "$ROOT/.env" | cut -d= -f2- | tr -d '"'\''\r' | xargs)"
  runpodctl pod delete "$POD" < /dev/null && say "pod $POD deleted" \
    || say "WARNING: pod delete failed"
fi
exit "$LIVE_RC"
