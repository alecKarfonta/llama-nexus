#!/usr/bin/env bash
# [TIME] | INFO | RUNPOD_BOOTSTRAP | stand up Qwen3.8-Flash-Next (qwen4exp) on a
# single RTX PRO 6000 Blackwell pod.
#
# Why this exists: the arch needs llama.cpp PR #27742 (unslothai fork, orphaned
# SHA) plus the local MTP draft-head patch, and the repo image is compiled for
# CUDA_ARCH=86 (Ampere). Blackwell is sm_120, so the binary must be rebuilt on
# the pod. Build and the 111 GB model pull run concurrently since both are long.
#
# Memory plan on one 96 GB card:
#   ~78 GB  GPU weights
#   ~33 GB  per_layer_token_embd (51B n-gram PLE) -> host RAM via mmap
#    <3 GB  q8_0 KV at 192K ctx (only 12/48 layers hold KV: QSA + GDN)
# No layer split and no n_cpu_moe expert offload, unlike the 4x3090 Ti rig.

set -uo pipefail

WORK=/workspace
LOGS="$WORK/logs"
mkdir -p "$WORK" "$LOGS"
exec > >(tee -a "$LOGS/bootstrap.log") 2>&1

say() { echo "[$(date -u +%H:%M:%S)] | INFO | bootstrap | $*"; }

LLAMA_REMOTE="https://github.com/unslothai/llama.cpp.git"
LLAMA_REF="bea3b12daee45876b0129a3602dc8f534ce30bf0"
CUDA_ARCH="${CUDA_ARCH:-120}"
MODEL_REPO="unsloth/Qwen3.8-Flash-Next-GGUF"
MODEL_GLOB="UD-Q4_K_XL/*"
DRAFT_REPO="dzannotti/Qwen3.8-Flash-Next-MTP-GGUF"
DRAFT_FILE="Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf"

say "start on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# ---------------------------------------------------------------- ssh access
say "installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  openssh-server git cmake build-essential curl ca-certificates \
  libcurl4-openssl-dev python3 python3-pip pciutils >/dev/null

mkdir -p /run/sshd /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
  printf '%s\n' "$PUBLIC_KEY" >>/root/.ssh/authorized_keys
fi
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
# Ubuntu ships `Include /etc/ssh/sshd_config.d/*.conf` at the top of sshd_config
# and OpenSSH keeps the FIRST value it sees, so a drop-in silently overrides the
# edits above. Clearing the drop-in dir is what actually makes root key auth work.
rm -f /etc/ssh/sshd_config.d/*.conf 2>/dev/null || true
/usr/sbin/sshd -D -e &
say "sshd started (authorized_keys lines: $(wc -l </root/.ssh/authorized_keys 2>/dev/null || echo 0))"

# Community-cloud pods frequently have no public IP, so SSH may be unreachable.
# Serve just the log dir over the RunPod HTTP proxy so progress is observable
# either way. Only logs are exposed - not the patch or the rest of /workspace.
python3 -m http.server 8888 --directory "$LOGS" >/dev/null 2>&1 &
say "log server on :8888 (serving $LOGS only)"

# CUDA forward-compat shim breaks on some host drivers (same fix as the repo image)
rm -rf /usr/local/cuda/compat 2>/dev/null || true

# ------------------------------------------------------------ model download
(
  set -uo pipefail
  pip install -q --no-cache-dir --break-system-packages \
    "huggingface_hub[cli,hf_transfer]" 2>/dev/null \
    || pip install -q --no-cache-dir "huggingface_hub[cli,hf_transfer]"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  mkdir -p "$WORK/models"
  echo "pulling $MODEL_REPO $MODEL_GLOB"
  hf download "$MODEL_REPO" --include "$MODEL_GLOB" --local-dir "$WORK/models"
  echo "pulling $DRAFT_REPO $DRAFT_FILE"
  hf download "$DRAFT_REPO" --include "$DRAFT_FILE" --local-dir "$WORK/models"
  du -sh "$WORK/models"
  touch "$WORK/.models-done"
  echo "MODELS DONE"
) >"$LOGS/download.log" 2>&1 &
DL_PID=$!
say "model download started (pid $DL_PID) -> $LOGS/download.log"

# --------------------------------------------------------------- build stage
(
  set -uo pipefail
  cd "$WORK"
  # Full clone: the pinned SHA is orphaned on the fork, so it must be fetched by
  # SHA rather than reached from a branch tip.
  git clone --quiet https://github.com/ggml-org/llama.cpp.git llama.cpp
  cd llama.cpp
  git fetch --quiet "$LLAMA_REMOTE" "$LLAMA_REF"
  git checkout --quiet FETCH_HEAD
  echo "tree: $(git rev-parse --short HEAD)"
  git apply --check "$WORK/qwen4exp-mtp-draft-head.patch"
  git apply "$WORK/qwen4exp-mtp-draft-head.patch"
  echo "patch applied"
  cmake . -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  cmake --build build --config Release -j"$(( $(nproc) > 12 ? 12 : $(nproc) ))" \
    --target llama-server llama-cli llama-gguf-split
  ./build/bin/llama-cli --version 2>&1 | head -2
  touch "$WORK/.build-done"
  echo "BUILD DONE"
) >"$LOGS/build.log" 2>&1 &
BUILD_PID=$!
say "llama.cpp build started (pid $BUILD_PID, sm_$CUDA_ARCH) -> $LOGS/build.log"

wait "$DL_PID" || say "WARN download exited nonzero"
wait "$BUILD_PID" || say "WARN build exited nonzero"

if [ ! -f "$WORK/.build-done" ] || [ ! -f "$WORK/.models-done" ]; then
  say "ERROR bootstrap incomplete: build=$([ -f "$WORK/.build-done" ] && echo ok || echo FAIL) models=$([ -f "$WORK/.models-done" ] && echo ok || echo FAIL)"
  say "leaving pod up for inspection"
  sleep infinity
fi

say "both stages complete; launching llama-server"
bash "$WORK/serve.sh" || say "ERROR llama-server exited"
sleep infinity
