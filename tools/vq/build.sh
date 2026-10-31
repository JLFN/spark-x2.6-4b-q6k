#!/usr/bin/env bash
# Build the vendored dense-matrix VQ encoder.
#
# The encoder is a copy of /data/embeddinggemma2/tools/eg2_vq.c with its two
# absolute includes rewritten to tools/vq/vendor. The headers are vendored from
# /data/YoungAi (MIT, commit a4e6d31) and /data/ds4-dfm-rs (ds41_vq_fmt.h, which
# is itself vendored from the same YoungAi tree). See THIRD_PARTY.md.
#
# The reference builds this with plain cc and no CUDA: the GPU Lloyd path assumes
# unified memory, and this machine reports pageableMemoryAccess = 0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CC=${CC:-cc}

mkdir -p "$ROOT/bin"
"$CC" -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L \
  -I "$ROOT/tools/vq/vendor" \
  -o "$ROOT/bin/eg2_vq" "$ROOT/tools/vq/eg2_vq.c" -lm
echo "built $ROOT/bin/eg2_vq"

# Q6_K reference error, using llama.cpp's own quantizer. Needs a built llama.cpp.
LLAMACPP=${LLAMACPP:-/data/llama.cpp}
if [ -f "$LLAMACPP/build/bin/libggml-base.so" ]; then
  "$CC" -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L \
    -I "$ROOT/tools/vq/vendor" \
    -I "$LLAMACPP/ggml/include" \
    -I "$LLAMACPP/ggml/src" \
    -o "$ROOT/bin/quant_err" "$ROOT/tools/vq/quant_err.c" \
    -L "$LLAMACPP/build/bin" -lggml-base -lggml -lm \
    -Wl,-rpath,"$LLAMACPP/build/bin"
  echo "built $ROOT/bin/quant_err"
else
  echo "skipped quant_err: $LLAMACPP/build/bin/libggml-base.so not found"
fi
