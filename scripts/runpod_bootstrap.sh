#!/usr/bin/env bash
# [TIME] | INFO | RUNPOD_BOOTSTRAP | stand up Qwen3.8-Flash-Next (qwen4exp) on a
# single RTX PRO 6000 Blackwell pod.
#
# Why this exists: the arch needs llama.cpp PR #27742 (unslothai fork, orphaned
# SHA) plus the local MTP draft-head patch, and the repo image is compiled for
# CUDA_ARCH=86 (Ampere). Blackwell is sm_120, so the binary must be rebuilt on
# the pod. Build and the 111 GB model pull run concurrently since both are long.
#
# Memory plan on one 96 GB card (sizes read from the UD-Q4_K_XL tensor table):
#  82.5 GB  GPU weights (77.0 GB MoE experts + 5.5 GB attn/norms/embd/output)
#  28.8 GB  per_layer_token_embd (51B n-gram PLE, IQ4_NL) -> host RAM via mmap
#    <3 GB  q8_0 KV at 192K ctx (only 12/48 layers hold KV: QSA + GDN)
# ~10 GB is left for compute buffers, CUDA context and the MTP draft model.
# The PLE is a gather, not a matmul, so hosting it off-GPU is near free.
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

# serve   : llama-server per GPU, the original purpose of this script
# extract : no servers. Builds libllama.so plus the mtp shim and waits for
#           mtp_extract_h.py to be driven against it, for harvesting trunk
#           hidden states to train the draft head on. One 96 GB card runs the
#           model without a layer split, which is the whole point -- the 4x3090
#           Ti rig serialises on the split and idles three cards out of four.
ROLE="${ROLE:-serve}"
case "$ROLE" in serve|extract) ;; *) echo "ROLE must be serve or extract" >&2; exit 1 ;; esac

say "start on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# ---------------------------------------------------------------- ssh access
say "installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  openssh-server git cmake build-essential curl ca-certificates \
  libcurl4-openssl-dev python3 python3-pip pciutils >/dev/null

# llama.cpp is compiled from source for sm_120, so nvcc is mandatory. RunPod's
# pytorch images carry the CUDA runtime and usually the compiler too, but leave
# it off PATH; only pay for the toolkit download if it is genuinely absent.
export PATH="/usr/local/cuda/bin:$PATH"
if ! command -v nvcc >/dev/null 2>&1; then
  say "nvcc not found, installing cuda toolkit (this is a few GB)"
  apt-get install -y -qq --no-install-recommends cuda-toolkit-12-8 >/dev/null 2>&1 \
    || apt-get install -y -qq --no-install-recommends nvidia-cuda-toolkit >/dev/null 2>&1 \
    || say "WARN could not install a cuda toolkit; the CUDA build will fail"
  export PATH="/usr/local/cuda/bin:$PATH"
fi
say "nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo MISSING)"

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
# edits above.
rm -f /etc/ssh/sshd_config.d/*.conf 2>/dev/null || true
# Root has no password in a base container, so PAM's account stage rejects the
# login *after* sshd has already accepted the key - the tell is a debug log with
# "Server accepts key" immediately followed by "Permission denied (publickey)".
sed -i 's/^#\?UsePAM.*/UsePAM no/' /etc/ssh/sshd_config
grep -q '^UsePAM' /etc/ssh/sshd_config || echo 'UsePAM no' >>/etc/ssh/sshd_config
/usr/sbin/sshd -D -e &
say "sshd started (authorized_keys lines: $(wc -l </root/.ssh/authorized_keys 2>/dev/null || echo 0))"

# Community-cloud pods frequently have no public IP, so SSH may be unreachable.
# Serve just the log dir over the RunPod HTTP proxy so progress is observable
# either way. Only logs are exposed - not the patch or the rest of /workspace.
python3 -m http.server 8888 --directory "$LOGS" >/dev/null 2>&1 &
say "log server on :8888 (serving $LOGS only)"

# NOTE: the repo Dockerfile does `rm -rf /usr/local/cuda/compat` because the shim
# breaks on the local rig's driver. Do NOT copy that here - on a rented host the
# kernel driver is usually OLDER than the image's CUDA runtime, and the compat
# layer is the thing that bridges the gap. Removing it aborts llama-server with
# "CUDA driver version is insufficient for CUDA runtime version".
say "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
say "cuda compat present: $([ -d /usr/local/cuda/compat ] && echo yes || echo no)"

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
  echo "cloning llama.cpp (competes with the model pull for bandwidth)"
  git clone --progress https://github.com/ggml-org/llama.cpp.git llama.cpp 2>&1 | tail -3
  cd llama.cpp
  echo "fetching pinned SHA $LLAMA_REF"
  git fetch "$LLAMA_REMOTE" "$LLAMA_REF" 2>&1 | tail -3
  git checkout FETCH_HEAD 2>&1 | tail -2
  echo "tree: $(git rev-parse --short HEAD)"
  git apply --check "$WORK/qwen4exp-mtp-draft-head.patch"
  git apply "$WORK/qwen4exp-mtp-draft-head.patch"
  echo "patch applied"
  # extract drives libllama over ctypes, so it needs the shared objects that a
  # static build never produces
  SHARED=OFF; [ "$ROLE" = extract ] && SHARED=ON
  cmake . -B build \
    -DBUILD_SHARED_LIBS=$SHARED \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  cmake --build build --config Release -j"$(( $(nproc) > 12 ? 12 : $(nproc) ))" \
    --target llama-server llama-cli llama-gguf-split
  ./build/bin/llama-cli --version 2>&1 | head -2
  if [ "$ROLE" = extract ]; then
    LLAMA_DIR="$WORK/llama.cpp" bash "$WORK/scripts/build_mtp_shim.sh"
  fi
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

# Fail loudly here rather than 100 GB into a model load.
say "CUDA sanity check"
VER_OUT=$("$WORK/llama.cpp/build/bin/llama-cli" --version 2>&1)
echo "$VER_OUT"
if echo "$VER_OUT" | grep -q 'failed to initialize CUDA'; then
  say "ERROR CUDA init failed - driver/runtime mismatch, not launching"
  say "  driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  say "  compat: $([ -d /usr/local/cuda/compat ] && echo present || echo MISSING)"
  sleep infinity
fi
say "CUDA ok"

if [ "$ROLE" = extract ]; then
  # numpy drives the extractor; torch is for training the head on the same pod
  # afterwards, so the 9.7 GB of hidden states never has to cross the network.
  # cu128 wheels to match the image and give sm_120 kernels -- the default
  # PyPI wheel has no Blackwell support and falls back to failing at load.
  pip install -q --no-cache-dir --break-system-packages numpy 2>/dev/null \
    || pip install -q --no-cache-dir numpy
  say "installing torch (cu128, for the on-pod training phase)"
  pip install -q --no-cache-dir --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu128 torch 2>/dev/null \
    || pip install -q --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cu128 torch
  python3 -c "import torch;print('  torch',torch.__version__,'sm',torch.cuda.get_arch_list()[-3:])" \
    2>&1 | sed 's/^/    /' || say "WARN torch import failed"
  say "extract role ready:"
  say "  model  $(ls -d "$WORK"/models/UD-Q4_K_XL 2>/dev/null || echo MISSING)"
  say "  shim   $WORK/llama.cpp/build/bin/libmtpextract.so"
  # The prep data (tokens.bin + index.jsonl) is pushed separately rather than
  # baked in: it is derived from real Cursor traffic, so it moves point to point
  # and is never published. Extraction starts when that lands.
  say "waiting for $WORK/prep/.ready (push tokens.bin + index.jsonl, then touch it)"
  while [ ! -f "$WORK/prep/.ready" ]; do sleep 20; done
  say "prep data present:"; ls -la "$WORK/prep" | sed 's/^/    /'
  say "run: python3 $WORK/scripts/mtp_extract_h.py --split train --per-session 3 \\"
  say "       --prep-dir $WORK/prep --out-dir $WORK/prep \\"
  say "       --model $WORK/models/UD-Q4_K_XL/<first-shard>.gguf \\"
  say "       --lib $WORK/llama.cpp/build/bin/libmtpextract.so \\"
  say "       --tensor-split '' --n-gpu-layers 999"
  sleep infinity
fi

# One llama-server per GPU on ports 8080, 8081, ... Each is --parallel 1 (arch
# limit), so N GPUs means N servers; there is no way to get concurrency out of a
# single server on qwen4exp. Weights are mmapped from one shared copy on disk.
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$NGPU" -lt 1 ] && NGPU=1
say "both stages complete; launching $NGPU llama-server instance(s)"

for i in $(seq 0 $((NGPU - 1))); do
  port=$((8080 + i))
  GPU_INDEX="$i" PORT="$port" NSERVERS="$NGPU" PARALLEL="${PARALLEL:-1}" \
    bash "$WORK/serve.sh" >"$LOGS/server-$port.log" 2>&1 &
  say "  gpu $i -> :$port (log server-$port.log)"
  # Stagger the loads: four processes faulting in the same 104 GB of weights at
  # once thrashes the page cache and slows every one of them down.
  sleep 45
done

# Report readiness so the log endpoint shows when each port starts answering.
for i in $(seq 0 $((NGPU - 1))); do
  port=$((8080 + i))
  for _ in $(seq 1 120); do
    if curl -fsS -m 3 -H "Authorization: Bearer ${LLAMA_API_KEY:-}" \
         "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      say "READY :$port"; break
    fi
    sleep 10
  done
done
say "all servers launched"
wait
