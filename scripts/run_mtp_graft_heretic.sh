#!/usr/bin/env bash
# Graft MTP heads from an official 35B-A3B MTP GGUF onto the heretic Q6_K base.
# Runs in an isolated container (never inside llamacpp-api). Uses ionice/nice for I/O.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOLUME="${LLAMACPP_MODELS_VOLUME:-llama-nexus_gpt_oss_models}"
MODELS="/models"
TARGET="${TARGET_GGUF:-Qwen3.6-35B-A3B-uncensored-heretic-Q6_K.gguf}"
SOURCE="${SOURCE_GGUF:-Qwen3.6-35B-A3B-Q8_0.gguf}"
OUTPUT="${OUTPUT_GGUF:-Qwen3.6-35B-A3B-uncensored-heretic-Q6_K-MTP.gguf}"
HF_REPO="${MTP_DONOR_REPO:-unsloth/Qwen3.6-35B-A3B-MTP-GGUF}"

echo "=== MTP graft (safe offline job) ==="
echo "Volume:  $VOLUME"
echo "Target:  $TARGET"
echo "Donor:   $SOURCE (from $HF_REPO)"
echo "Output:  $OUTPUT"
echo "Inference container is NOT touched."
echo

docker run --rm \
  --name "mtp-graft-$$" \
  -e "HF_REPO=${HF_REPO}" \
  -e "TARGET=${TARGET}" \
  -e "SOURCE=${SOURCE}" \
  -e "OUTPUT=${OUTPUT}" \
  -e "MODELS=${MODELS}" \
  -v "${VOLUME}:${MODELS}" \
  -v "${REPO_ROOT}/scripts/mtp_graft_convert.py:/opt/mtp_graft_convert.py:ro" \
  python:3.12-slim \
  bash -c "
    set -euo pipefail
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export HF_HUB_ENABLE_HF_TRANSFER=0
    apt-get update -qq >/dev/null
    apt-get install -y -qq util-linux procps >/dev/null
    pip install -q 'gguf>=0.10.0' 'huggingface_hub>=0.23.0'

    run_nice() {
      if command -v ionice >/dev/null 2>&1; then
        ionice -c3 nice -n19 \"\$@\"
      else
        nice -n19 \"\$@\"
      fi
    }

    if [ ! -f '${MODELS}/${TARGET}' ]; then
      echo 'ERROR: target not found: ${MODELS}/${TARGET}' >&2
      exit 1
    fi

    if [ ! -f '${MODELS}/${SOURCE}' ]; then
      echo 'Downloading MTP donor (one-time, ~38GB)...'
      run_nice python3 - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
repo = os.environ['HF_REPO']
fname = os.environ['SOURCE']
models = os.environ['MODELS']
path = hf_hub_download(repo_id=repo, filename=fname, local_dir=models + '/.mtp-donor-cache')
dest = models + '/' + fname
if os.path.abspath(path) != os.path.abspath(dest):
    shutil.copy2(path, dest)
print('Donor ready:', dest, os.path.getsize(dest))
PY
    fi

    if [ -f '${MODELS}/${OUTPUT}' ]; then
      echo 'Output already exists — validating only: ${MODELS}/${OUTPUT}'
    else
      echo 'Preflight header check...'
      run_nice python3 - <<'PY'
from gguf import GGUFReader
import os
models = os.environ['MODELS']
target = os.environ['TARGET']
source = os.environ['SOURCE']
tr = GGUFReader(models + '/' + target, 'r')
sr = GGUFReader(models + '/' + source, 'r')
arch = tr.fields['general.architecture'].contents()
tb = tr.fields[f'{arch}.block_count'].contents()
sb = sr.fields[f'{arch}.block_count'].contents()
sn = sr.fields.get(f'{arch}.nextn_predict_layers')
sn = sn.contents() if sn else None
print(f'arch={arch} target_blocks={tb} source_blocks={sb} source_nextn={sn}')
if tb != sb - (sn or 0):
    print('WARN: block_count mismatch — graft may fail; check donor file')
extra = [t.name for t in sr.tensors if t.name.startswith(f'blk.{tb}.')]
print(f'extra tensors to graft: {len(extra)}')
if not extra:
    raise SystemExit('ERROR: no blk.{tb}.* tensors in donor')
PY
      echo 'Running graft (streaming copy, several minutes)...'
      run_nice python3 /opt/mtp_graft_convert.py \
        '${MODELS}/${TARGET}' '${MODELS}/${SOURCE}' '${MODELS}/${OUTPUT}'
    fi

    echo 'Post-graft metadata:'
    run_nice python3 - <<'PY'
from gguf import GGUFReader
import os
path = os.environ['MODELS'] + '/' + os.environ['OUTPUT']
r = GGUFReader(path, 'r')
arch = r.fields['general.architecture'].contents()
bc = r.fields[f'{arch}.block_count'].contents()
nn = r.fields[f'{arch}.nextn_predict_layers'].contents()
extra = sum(1 for t in r.tensors if t.name.startswith(f'blk.{bc - nn}.'))
print(f'OK: {os.path.basename(path)} block_count={bc} nextn={nn} mtp_tensors={extra} total_tensors={len(r.tensors)}')
PY
  "

echo
echo "Done. Deploy with model file: ${OUTPUT}"
echo "Then restart llamacpp-api and enable MTP in Deploy UI."
