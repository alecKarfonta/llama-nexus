#!/usr/bin/env bash
# Runs the full extraction on a rented pod: val first, then train.
#
# val goes first on purpose. It is 4.8M tokens against train's 44.8M, so it
# finishes in under an hour and gives a complete evaluation set to score
# against while train is still running -- if something is wrong with the
# extraction we find out one hour in, not eight.
#
# Detached via setsid so an SSH drop does not take the run with it. Both splits
# use --resume, which reconciles h.bf16 against the index and truncates to the
# last row actually backed by data, so re-running after any interruption is
# safe and cheap.
set -u

LLAMA=${LLAMA:-/workspace/llama.cpp}
MODEL=${MODEL:-/workspace/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
PREP=${PREP:-/workspace/prep}
OUT=${OUT:-/workspace/out}
LOG=${LOG:-/workspace/extract.log}

# Mandatory for qwen4exp: the attention path asserts that no k/v rotation
# tensors exist, but a quantized KV cache turns rotation on by default.
export LLAMA_ATTN_ROT_DISABLE=1

run() {
  local split=$1
  echo "=== $split start $(date -Is) ==="
  python3 /workspace/scripts/mtp_extract_h.py \
    --split "$split" --resume \
    --lib "$LLAMA/build/bin/libmtpextract.so" \
    --model "$MODEL" \
    --prep-dir "$PREP" --out "$OUT" \
    --n-gpu-layers 999 --report-every 25
  echo "=== $split done $(date -Is) exit=$? ==="
}

verify() {
  # A prep dir can be the right size and still be the wrong bytes -- an
  # interrupted transfer once refilled val.* with the pre-subset originals,
  # which have identical length because val takes no selection. The offsets
  # then point into a differently ordered stream, so extraction runs happily
  # and produces states paired to the wrong tokens. Cheap to check, expensive
  # to miss, so it gates the GPU.
  if [ ! -s "$PREP/manifest.md5" ]; then
    echo "ABORT: no $PREP/manifest.md5 to verify against"; exit 1
  fi
  if ! ( cd "$PREP" && md5sum -c --quiet manifest.md5 ); then
    echo "ABORT: prep checksums do not match manifest; re-send the payload"; exit 1
  fi
  echo "prep verified against manifest ($(wc -l < "$PREP/manifest.md5") files)"
}

verify
run val
verify   # re-checked: val took ~45 min, long enough for a stray writer to land
run train
echo "=== ALL EXTRACTION DONE $(date -Is) ==="
