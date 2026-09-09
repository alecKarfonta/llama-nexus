#!/usr/bin/env bash
# [TIME] | INFO | BUILD_MTP_SHIM | compile libmtpextract.so against a patched
# llama.cpp tree.
#
# The shim is the C ABI that mtp_extract_h.py drives over ctypes; see
# mtp_extract_shim.cpp for why it exists rather than binding libllama directly.
#
# Two include paths are load-bearing and easy to miss:
#   src/     llama-ext.h is a staging header (llama_set_embeddings_nextn and
#            llama_get_embeddings_nextn_ith), not part of the public include/
#   ggml/include  ggml-backend.h
#
# The tree must be built with -DBUILD_SHARED_LIBS=ON. runpod_bootstrap.sh builds
# static because it only needs llama-server, so an extraction pod has to
# override that or there is no libllama.so to link against.
set -euo pipefail

LLAMA="${LLAMA_DIR:-/mnt/comfyui-models/llama-nexus/build/llama.cpp}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mtp_extract_shim.cpp"
OUT="$LLAMA/build/bin/libmtpextract.so"

for f in "$LLAMA/include/llama.h" "$LLAMA/src/llama-ext.h" "$LLAMA/build/bin/libllama.so"; do
  [ -e "$f" ] || { echo "missing $f" >&2
    [ "$f" = "$LLAMA/build/bin/libllama.so" ] && echo "  (rebuild with -DBUILD_SHARED_LIBS=ON)" >&2
    exit 1; }
done

echo "building $OUT"
g++ -O2 -std=c++17 -shared -fPIC \
  -o "$OUT" "$SRC" \
  -I"$LLAMA/include" -I"$LLAMA/src" -I"$LLAMA/ggml/include" \
  -L"$LLAMA/build/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$LLAMA/build/bin"

echo "ok: $(ls -la "$OUT" | awk '{print $5}') bytes"
ldd "$OUT" | grep -E 'llama|ggml' | sed 's/^/  /'
