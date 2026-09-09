#!/usr/bin/env bash
# Pull pmin adapter, offline k=16 p_min=0.85 gate, merge+quantize, live STRUCT gate vs 7.38.
set -u

ROOT=/home/alec/git/llama-nexus
BUILD=/mnt/comfyui-models/llama-nexus/build
POD=${POD:-$(cat /tmp/pod_kstep_pmin_id 2>/dev/null)}
SSHCFG=${SSHCFG:-/tmp/pod_kstep_pmin_ssh_config}
HOSTALIAS=${HOSTALIAS:-ksteppminpod}
DEST=${DEST:-/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-h-pod}
ADAPTER_REMOTE=${ADAPTER_REMOTE:-/workspace/mtp_lora_r32_kstep_pmin.pt}
ADAPTER_LOCAL=${ADAPTER_LOCAL:-$DEST/$(basename "$ADAPTER_REMOTE")}
E1=${E1:-$DEST/mtp_lora_r32_kstep.pt}
EVAL=/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-eval
PMIN=${PMIN:-0.85}
LIVE_STRUCT_BAR=${LIVE_STRUCT_BAR:-7.38}
STOCK_BF16=$BUILD/mtp-bf16.gguf
MERGED=$BUILD/mtp-lora-kstep-pmin-bf16.gguf
QUANT=$BUILD/Qwen3.8-Flash-Next-MTP-kstep-pmin-Q4_K_M.gguf
QUANT_LINK=$BUILD/mtp-lora-kstep-pmin-Q4_K_M.gguf
DIAG=/tmp/pmin_diagnostic

SSH="ssh -F $SSHCFG -o ConnectTimeout=20 -n $HOSTALIAS"
say() { echo "[$(date -Is)] $*"; }
running() { $SSH "pgrep -f '[${1:0:1}]${1:1}' > /dev/null" 2>/dev/null; }

mkdir -p "$DEST"
[ -f "$SSHCFG" ] || { say "ABORT: no $SSHCFG"; exit 1; }

say "waiting for $ADAPTER_REMOTE"
while ! $SSH "test -s $ADAPTER_REMOTE" 2>/dev/null; do
  if ! running pod_train_kstep.sh && ! running mtp_train_head.py; then
    say "ABORT: trainer gone and no adapter"; exit 1
  fi
  sleep 180
done
while running mtp_train_head.py; do sleep 60; done
say "trainer finished"

scp -F "$SSHCFG" "$HOSTALIAS:$ADAPTER_REMOTE" "$ADAPTER_LOCAL"
scp -F "$SSHCFG" "$HOSTALIAS:/workspace/train_kstep.log" "$DEST/train_kstep_pmin.log" 2>/dev/null || true

OUT=/tmp/kstep_pmin_final_gate
mkdir -p "$OUT"

rollout() {
  local label=$1 adapter=$2
  local extra=()
  [ -n "$adapter" ] && extra=(--adapter "$adapter")
  OMP_NUM_THREADS=44 python3 -u "$ROOT/scripts/mtp_train_head.py" --rollout --fp32 --device cpu \
    --prep-dir "$EVAL" --head-gguf "$STOCK_BF16" --split val \
    --per-span 3 --k-max 16 --window 64 --batch 8 --p-min "$PMIN" "${extra[@]}" \
    > "$OUT/${label}.json" 2> "$OUT/${label}.log"
}

say "offline rollout k=16 p_min=$PMIN"
rollout stock ""
rollout e1 "$E1"
rollout pmin "$ADAPTER_LOCAL"

python3 - "$OUT" "$PMIN" <<'PY'
import json, sys, os
out, pmin = sys.argv[1], float(sys.argv[2])
def load(name):
    t=open(os.path.join(out, f"{name}.json")).read()
    return json.loads(t[t.index('{'):])['rollout']
s, e, p = load("stock"), load("e1"), load("pmin")
for label, rep in [("stock", s), ("e1", e), ("pmin", p)]:
    print(f"{label}: E[len]={rep['expected_accepted_len']:.4f} E[len_pmin]={rep['expected_accepted_len_pmin']:.4f}")
bar_stock = s["expected_accepted_len_pmin"]
bar_e1 = e["expected_accepted_len_pmin"]
elen = p["expected_accepted_len_pmin"]
ok = elen > bar_stock and elen > bar_e1
print(f"gate: {'PASS' if ok else 'FAIL'} (need > stock {bar_stock:.4f} and e1 {bar_e1:.4f})")
sys.exit(0 if ok else 1)
PY
GATE_RC=$?

if [ "$GATE_RC" -ne 0 ]; then
  say "offline gate failed"
  [ "${DELETE_POD:-1}" = 1 ] && [ -n "${POD:-}" ] && runpodctl pod delete "$POD" < /dev/null || true
  exit 1
fi

say "merge + quantize"
python3 "$ROOT/scripts/mtp_train_head.py" --merge "$ADAPTER_LOCAL" \
  --export-gguf "$MERGED" --head-gguf "$STOCK_BF16" --device cpu
"$BUILD/llama.cpp/build-quantize/bin/llama-quantize" "$MERGED" "$QUANT" Q4_K_M
ln -sf "$(basename "$QUANT")" "$QUANT_LINK"
MODELS=/mnt/comfyui-models/llama-nexus/gpt_oss_models
if cp -f "$QUANT" "$MODELS/$(basename "$QUANT")" 2>/dev/null; then
  say "staged $(basename "$QUANT")"
else
  sudo cp -f "$QUANT" "$MODELS/$(basename "$QUANT")"
  sudo chown "$(id -un):$(id -gn)" "$MODELS/$(basename "$QUANT")" 2>/dev/null || true
fi

SWEEP_OUT=/tmp/kstep_pmin_live_sweep.json
python3 "$ROOT/scripts/sweep_draft_knobs.py" --head kstep_pmin \
  --p-min 0.80 0.85 --runs 2 --no-restore --out "$SWEEP_OUT" || exit 1

python3 - "$SWEEP_OUT" "$LIVE_STRUCT_BAR" <<'PY'
import json, sys
results = json.load(open(sys.argv[1]))
bar = float(sys.argv[2])
best = (-1.0, None)
for pt in results:
    per = pt.get("workloads") or {}
    vals = [per[w]["accepted_per_round"] for w in ("code", "json") if w in per]
    if not vals: continue
    s = sum(vals)/len(vals)
    print(f"p_min={pt['p_min']:.2f} STRUCT={s:.2f}")
    if s > best[0]: best = (s, pt["p_min"])
print(f"best={best[0]:.2f} bar={bar}")
sys.exit(0 if best[0] > bar else 1)
PY
LIVE_RC=$?

if [ "$LIVE_RC" -eq 0 ]; then
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
    req = urllib.request.Request(f"{BACKEND}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)
api("PUT", "/api/v1/service/config", {
    "speculative": {"model_draft": "Qwen3.8-Flash-Next-MTP-kstep-pmin-Q4_K_M.gguf", "gpu_layers_draft": 999},
    "mtp": {"draft_p_min": float("$BEST_P"), "draft_n_min": 0},
})
api("POST", "/api/v1/service/restart")
print("shipped kstep_pmin p_min=$BEST_P")
PY
else
  python3 - <<'PY'
import json, os, urllib.request
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8700")
API_KEY = os.getenv("API_KEY", "placeholder-api-key")
def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BACKEND}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)
api("PUT", "/api/v1/service/config", {
    "speculative": {"model_draft": "Qwen3.8-Flash-Next-MTP-kstep-Q4_K_M.gguf", "gpu_layers_draft": 999},
    "mtp": {"draft_p_min": 0.85, "draft_n_min": 0},
})
api("POST", "/api/v1/service/restart")
print("restored e1 @ 0.85")
PY
fi

[ "${DELETE_POD:-1}" = 1 ] && [ -n "${POD:-}" ] && runpodctl pod delete "$POD" < /dev/null || true
exit "$LIVE_RC"
