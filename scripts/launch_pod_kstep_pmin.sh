#!/usr/bin/env bash
# [TIME] | INFO | LAUNCH_POD_KSTEP_PMIN | p_min confidence-margin loss on k-step e1.
set -euo pipefail

ROOT=/home/alec/git/llama-nexus
DATA_ROOT=/mnt/comfyui-models/llama-nexus/datasets/dataset-v2
PREP_SUB=$DATA_ROOT/mtp-prep-subset
STATES=$DATA_ROOT/mtp-h-pod
ADAPTER=${ADAPTER:-$STATES/mtp_lora_r32_kstep.pt}
REMOTE_RESUME=${REMOTE_RESUME:-/workspace/states/$(basename "$ADAPTER")}
REMOTE_OUT=${REMOTE_OUT:-/workspace/mtp_lora_r32_kstep_pmin.pt}
LR=${LR:-1e-5}
LR_SCHEDULE=${LR_SCHEDULE:-constant}
ROLL_P_MIN=${ROLL_P_MIN:-0.85}
ROLL_CONF_MARGIN=${ROLL_CONF_MARGIN:-0.05}
ROLL_CONF_LAMBDA=${ROLL_CONF_LAMBDA:-1.0}
ABORT_ELEN=${ABORT_ELEN:-}
ABORT_AT=${ABORT_AT:-100 200}
SSHKEY=$HOME/.ssh/id_ed25519_runpod
INIT_B64="$(base64 -w0 "$ROOT/scripts/pod_init_train.sh")"

export PATH="$HOME/.local/bin:$PATH"
export RUNPOD_API_KEY="$(grep -m1 '^RUNPOD_API_KEY=' "$ROOT/.env" | cut -d= -f2- | tr -d '"'\''\r' | xargs)"
[ -n "$RUNPOD_API_KEY" ] || { echo "ABORT: no RUNPOD_API_KEY"; exit 1; }
[ -s "$ADAPTER" ] || { echo "ABORT: missing $ADAPTER"; exit 1; }

# Default abort bar from e1 diagnostic at p_min=0.85 if unset.
if [ -z "$ABORT_ELEN" ] && [ -f /tmp/pmin_diagnostic/e1_p0.85.json ]; then
  ABORT_ELEN=$(python3 - <<'PY'
import json
t=open("/tmp/pmin_diagnostic/e1_p0.85.json").read()
rep=json.loads(t[t.index("{"):])["rollout"]
print(f"{rep.get('expected_accepted_len_pmin', rep['expected_accepted_len']):.4f}")
PY
)
fi
ABORT_ELEN=${ABORT_ELEN:-4.0}

GPU='NVIDIA RTX PRO 6000 Blackwell Server Edition'
NAME="mtp-kstep-pmin-$(date +%m%d-%H%M)"

say() { echo "[$(date -Is)] $*"; }

say "creating pod $NAME (p_min=$ROLL_P_MIN margin=$ROLL_CONF_MARGIN abort=$ABORT_ELEN)"
CREATE_OUT=$(runpodctl pod create \
  --name "$NAME" \
  --image nvidia/cuda:12.8.1-devel-ubuntu22.04 \
  --gpu-id "$GPU" \
  --gpu-count 1 \
  --container-disk-in-gb 80 \
  --volume-in-gb 0 \
  --ports 22/tcp \
  --docker-args "bash -lc 'echo $INIT_B64 | base64 -d | bash'" \
  --wait 2>&1) || { echo "$CREATE_OUT"; exit 1; }

POD_ID=$(echo "$CREATE_OUT" | python3 -c "
import json,sys,re
t=sys.stdin.read().strip()
try:
  d=json.loads(t)
  if isinstance(d,list): d=d[0]
  print(d.get('id') or d.get('podId') or '')
except json.JSONDecodeError:
  m=re.search(r'\"id\"\\s*:\\s*\"([a-z0-9]+)\"', t)
  print(m.group(1) if m else '')
")
[ -n "$POD_ID" ] || { echo "ABORT: could not parse pod id"; exit 1; }
say "pod id $POD_ID"

sleep 15
POD_JSON=$(runpodctl pod get "$POD_ID" 2>/dev/null)
HOST=$(echo "$POD_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('ssh') or {}).get('ip') or d.get('publicIp') or '')")
PORT=$(echo "$POD_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('ssh') or {}).get('port') or '')")
[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "ABORT: no ssh"; exit 1; }

SSHCFG=/tmp/pod_kstep_pmin_ssh_config
cat > "$SSHCFG" <<EOF
Host ksteppminpod
  HostName $HOST
  Port $PORT
  User root
  IdentityFile $SSHKEY
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
  ServerAliveInterval 30
  ServerAliveCountMax 10
EOF

for i in $(seq 1 60); do
  ssh -F "$SSHCFG" -o ConnectTimeout=8 ksteppminpod 'echo ok' 2>/dev/null && break
  sleep 10
done
ssh -F "$SSHCFG" ksteppminpod 'echo ok' || exit 1

ssh -F "$SSHCFG" ksteppminpod 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; apt-get install -y -qq python3-pip >/dev/null; pip install -q --no-cache-dir numpy; pip install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch'
ssh -F "$SSHCFG" ksteppminpod 'mkdir -p /workspace/{scripts,prep,states,models,logs}'
scp -F "$SSHCFG" -r "$ROOT/scripts/mtp_train_head.py" "$ROOT/scripts/qwen4exp_nextn.py" \
  "$ROOT/scripts/pod_train_kstep.sh" ksteppminpod:/workspace/scripts/
scp -F "$SSHCFG" "$PREP_SUB/train.tokens.bin" "$PREP_SUB/val.tokens.bin" \
  "$PREP_SUB/manifest.md5" ksteppminpod:/workspace/prep/
scp -F "$SSHCFG" \
  "$STATES/train.h.bf16" "$STATES/train.h.index.jsonl" \
  "$STATES/val.h.bf16" "$STATES/val.h.index.jsonl" \
  "$ADAPTER" ksteppminpod:/workspace/states/

HF_TOKEN="$(grep -m1 '^HUGGINGFACE_TOKEN=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'\''\r' | xargs || true)"
ssh -F "$SSHCFG" ksteppminpod "pip install -q --no-cache-dir huggingface_hub && \
  ${HF_TOKEN:+HF_TOKEN='$HF_TOKEN' }hf download dzannotti/Qwen3.8-Flash-Next-MTP-GGUF \
  Qwen3.8-Flash-Next-MTP-BF16.gguf --local-dir /workspace/models && \
  ln -sf /workspace/models/Qwen3.8-Flash-Next-MTP-BF16.gguf /workspace/models/mtp-bf16.gguf"

ssh -F "$SSHCFG" ksteppminpod "ln -sf $REMOTE_RESUME /workspace/$(basename "$ADAPTER"); chmod +x /workspace/scripts/pod_train_kstep.sh"

ssh -F "$SSHCFG" ksteppminpod "RESUME=$REMOTE_RESUME OUT=$REMOTE_OUT LR=$LR LR_SCHEDULE=$LR_SCHEDULE \
  ROLL_P_MIN=$ROLL_P_MIN ROLL_CONF_MARGIN=$ROLL_CONF_MARGIN ROLL_CONF_LAMBDA=$ROLL_CONF_LAMBDA \
  ABORT_ELEN=$ABORT_ELEN ABORT_AT='$ABORT_AT' \
  setsid nohup bash /workspace/scripts/pod_train_kstep.sh \
  > /workspace/logs/train_kstep.log 2>&1 < /dev/null & disown; sleep 2; tail -8 /workspace/logs/train_kstep.log"

echo "$POD_ID" > /tmp/pod_kstep_pmin_id
say "pod $POD_ID started. When done: bash $ROOT/scripts/pull_pod_kstep_pmin_results.sh"
