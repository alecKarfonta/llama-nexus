#!/usr/bin/env bash
# [TIME] | INFO | LAUNCH_POD_KSTEP | create RTX 6000 pod, stage data, start k-step train.
set -euo pipefail

ROOT=/home/alec/git/llama-nexus
DATA_ROOT=/mnt/comfyui-models/llama-nexus/datasets/dataset-v2
PREP_SUB=$DATA_ROOT/mtp-prep-subset
STATES=$DATA_ROOT/mtp-h-pod
ADAPTER=${ADAPTER:-$STATES/mtp_lora_r32_kstep.pt}
REMOTE_RESUME=${REMOTE_RESUME:-/workspace/states/$(basename "$ADAPTER")}
REMOTE_OUT=${REMOTE_OUT:-/workspace/mtp_lora_r32_kstep_e2.pt}
LR=${LR:-1e-5}
LR_SCHEDULE=${LR_SCHEDULE:-constant}
SSHKEY=$HOME/.ssh/id_ed25519_runpod
PUBKEY="$(cat "${SSHKEY}.pub")"
INIT_B64="$(base64 -w0 "$ROOT/scripts/pod_init_train.sh")"

export PATH="$HOME/.local/bin:$PATH"
export RUNPOD_API_KEY="$(grep -m1 '^RUNPOD_API_KEY=' "$ROOT/.env" | cut -d= -f2- | tr -d '"'\''\r' | xargs)"
[ -n "$RUNPOD_API_KEY" ] || { echo "ABORT: no RUNPOD_API_KEY"; exit 1; }
[ -s "$ADAPTER" ] || { echo "ABORT: missing $ADAPTER"; exit 1; }

GPU='NVIDIA RTX PRO 6000 Blackwell Server Edition'
NAME="mtp-kstep-$(date +%m%d-%H%M)"

say() { echo "[$(date -Is)] $*"; }

say "creating pod $NAME"
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
for chunk in (t,):
  try:
    d=json.loads(chunk)
    if isinstance(d,list): d=d[0]
    pid=d.get('id') or d.get('podId') or ''
    if pid: print(pid); raise SystemExit
  except json.JSONDecodeError:
    pass
  m=re.search(r'\"id\"\\s*:\\s*\"([a-z0-9]+)\"', t)
  if m: print(m.group(1))
" )
[ -n "$POD_ID" ] || POD_ID=$(runpodctl pod list 2>/dev/null | python3 -c "
import json,sys
pods=json.load(sys.stdin)
for p in pods:
  if p.get('name')=='$NAME': print(p['id']); break
")
[ -n "$POD_ID" ] || { echo "ABORT: could not parse pod id from: $CREATE_OUT"; exit 1; }
say "pod id $POD_ID"

sleep 15
POD_JSON=$(runpodctl pod get "$POD_ID" 2>/dev/null)
HOST=$(echo "$POD_JSON" | python3 -c "
import json,sys
d=json.load(sys.stdin)
ssh=d.get('ssh') or {}
print(ssh.get('ip') or d.get('publicIp') or '')
")
PORT=$(echo "$POD_JSON" | python3 -c "
import json,sys
d=json.load(sys.stdin)
ssh=d.get('ssh') or {}
print(ssh.get('port') or '')
")
[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "ABORT: no ssh endpoint yet"; echo "$POD_JSON"; exit 1; }

SSHCFG=/tmp/pod_kstep_ssh_config
cat > "$SSHCFG" <<EOF
Host ksteppod
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

say "waiting for ssh $HOST:$PORT"
for i in $(seq 1 60); do
  if ssh -F "$SSHCFG" -o ConnectTimeout=8 ksteppod 'echo ok' 2>/dev/null; then break; fi
  sleep 10
done
ssh -F "$SSHCFG" ksteppod 'echo ok' || { echo "ABORT: ssh failed"; exit 1; }

say "installing python deps on pod"
ssh -F "$SSHCFG" ksteppod 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; apt-get install -y -qq python3-pip >/dev/null; pip install -q --no-cache-dir numpy; pip install -q --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch'

say "staging scripts"
ssh -F "$SSHCFG" ksteppod 'mkdir -p /workspace/{scripts,prep,states,models,logs}'
scp -F "$SSHCFG" -r "$ROOT/scripts/mtp_train_head.py" "$ROOT/scripts/qwen4exp_nextn.py" \
  "$ROOT/scripts/pod_train_kstep.sh" ksteppod:/workspace/scripts/

say "staging prep tokens + manifest"
scp -F "$SSHCFG" "$PREP_SUB/train.tokens.bin" "$PREP_SUB/val.tokens.bin" \
  "$PREP_SUB/manifest.md5" ksteppod:/workspace/prep/

say "staging hidden states + adapter (this takes a while)"
scp -F "$SSHCFG" \
  "$STATES/train.h.bf16" "$STATES/train.h.index.jsonl" \
  "$STATES/val.h.bf16" "$STATES/val.h.index.jsonl" \
  "$ADAPTER" ksteppod:/workspace/states/

say "downloading BF16 head on pod"
HF_TOKEN="$(grep -m1 '^HUGGINGFACE_TOKEN=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'\''\r' | xargs || true)"
ssh -F "$SSHCFG" ksteppod "pip install -q --no-cache-dir 'huggingface_hub[cli]' && \
  ${HF_TOKEN:+HF_TOKEN='$HF_TOKEN' }hf download dzannotti/Qwen3.8-Flash-Next-MTP-GGUF \
  Qwen3.8-Flash-Next-MTP-BF16.gguf --local-dir /workspace/models && \
  ln -sf /workspace/models/Qwen3.8-Flash-Next-MTP-BF16.gguf /workspace/models/mtp-bf16.gguf"

ssh -F "$SSHCFG" ksteppod "ln -sf $REMOTE_RESUME /workspace/$(basename "$ADAPTER"); chmod +x /workspace/scripts/pod_train_kstep.sh"

say "launching training detached (lr=$LR schedule=$LR_SCHEDULE out=$REMOTE_OUT)"
ssh -F "$SSHCFG" ksteppod "RESUME=$REMOTE_RESUME OUT=$REMOTE_OUT LR=$LR LR_SCHEDULE=$LR_SCHEDULE \
  setsid nohup bash /workspace/scripts/pod_train_kstep.sh \
  > /workspace/logs/train_kstep.log 2>&1 < /dev/null & disown; sleep 2; tail -8 /workspace/logs/train_kstep.log"

echo "$POD_ID" > /tmp/pod_kstep_id
say "pod $POD_ID training started. Monitor: ssh -F $SSHCFG ksteppod tail -f /workspace/train_kstep.log"
say "When done: POD=$POD_ID SSHCFG=$SSHCFG ADAPTER_REMOTE=$REMOTE_OUT STOCK_ELEN=5.72 bash $ROOT/scripts/pull_pod_kstep_results.sh"
