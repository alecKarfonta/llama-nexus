#!/usr/bin/env bash
# Local-side collector: waits for the pod's LoRA to land, brings back the
# adapter and the extracted trunk states, verifies both, and only then deletes
# the pod.
#
# Runs locally because the pod cannot reach in -- the transfer has to be pulled.
#
# Order matters. The adapter is 6 MB and is the actual product of the run, so it
# comes first and is confirmed before anything slower is attempted. The states
# are ~10 GB and cost eight GPU-hours to produce, so they are worth carrying
# home to make any future retrain free, but they are also the part most likely
# to be interrupted -- hence rsync with --append-verify and a retry loop rather
# than scp, which earlier in this project stalled at 180 KB/s and had to be
# abandoned mid-file.
#
# The pod is deleted only after every checksum matches. A failed verify leaves
# it running and says so: an idle pod costs ~$2/hr, but re-extracting costs ~$16
# and eight hours, so the cheap failure is to keep paying rent.
set -u

POD=${POD:-am7nj5yqtpbzo3}
SSHCFG=${SSHCFG:-$HOME/.ssh/pod_config}
HOSTALIAS=${HOSTALIAS:-mtppod}
DEST=${DEST:-/mnt/comfyui-models/llama-nexus/datasets/dataset-v2/mtp-h-pod}
ADAPTER=${ADAPTER:-/workspace/mtp_lora_r32.pt}
DELETE=${DELETE:-1}

SSH="ssh -F $SSHCFG -o ConnectTimeout=20 -n $HOSTALIAS"
say() { echo "[$(date -Is)] $*"; }

# pgrep skips its own process but not its parent, and running it over ssh means
# the remote shell's command line contains the pattern -- so a naive
# `pgrep -f mtp_train_head.py` always matches and any wait on it never ends.
# The bracket makes the pattern still match the real process while no longer
# matching the literal text of the command carrying it.
running() { $SSH "pgrep -f '[${1:0:1}]${1:1}' > /dev/null" 2>/dev/null; }

mkdir -p "$DEST"

say "waiting for $ADAPTER to appear on the pod"
while ! $SSH "test -s $ADAPTER" 2>/dev/null; do
  if ! running pod_train_head.sh && ! running mtp_train_head.py \
     && ! running mtp_extract_h.py; then
    say "ABORT: nothing running on the pod and no adapter present"
    exit 1
  fi
  sleep 180
done
# let the trainer finish writing and settle
while running mtp_train_head.py; do sleep 60; done
say "adapter present and trainer exited"

say "pulling adapter"
scp -F "$SSHCFG" "$HOSTALIAS:$ADAPTER" "$DEST/" || { say "ABORT: adapter pull failed"; exit 1; }
scp -F "$SSHCFG" "$HOSTALIAS:/workspace/train.log" "$DEST/train.log" 2>/dev/null || true
scp -F "$SSHCFG" "$HOSTALIAS:/workspace/extract.log" "$DEST/extract.log" 2>/dev/null || true
say "adapter is home: $(ls -la "$DEST/$(basename "$ADAPTER")" | awk '{print $5" bytes"}')"

# Checksums are computed on the pod so the comparison is end-to-end rather than
# against whatever happened to arrive.
say "checksumming states on the pod (a few minutes for ~10 GB)"
$SSH 'cd /workspace/out && md5sum * > /workspace/out.md5 && cat /workspace/out.md5' \
  > "$DEST/out.md5.remote" || { say "ABORT: remote checksum failed"; exit 1; }
sed 's/^/  /' "$DEST/out.md5.remote"

say "pulling states with rsync (resumable)"
ok=0
for attempt in 1 2 3 4 5; do
  if rsync -avh --partial --append-verify --timeout=120 \
       -e "ssh -F $SSHCFG" "$HOSTALIAS:/workspace/out/" "$DEST/" ; then
    ok=1; break
  fi
  say "rsync attempt $attempt failed or stalled; retrying"
  sleep 30
done
[ "$ok" = 1 ] || { say "ABORT: rsync did not complete; pod left running"; exit 1; }

say "verifying locally against the pod's checksums"
if ( cd "$DEST" && md5sum -c --quiet out.md5.remote ); then
  say "all states verified"
else
  say "ABORT: checksum mismatch after transfer; pod left running for a retry"
  exit 1
fi

du -sh "$DEST" | sed 's/^/  collected: /'

if [ "$DELETE" = 1 ]; then
  say "deleting pod $POD"
  runpodctl pod delete "$POD" && say "pod $POD deleted; billing stopped" \
    || say "WARNING: delete failed -- check the console, the pod may still bill"
else
  say "DELETE=0, leaving pod $POD running"
fi
say "done"
