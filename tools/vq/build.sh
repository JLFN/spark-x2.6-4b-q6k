#!/usr/bin/env bash
# Build every helper in this repository from sources that live inside it.
#
# No path outside the repository is used:
#   - the VQ encoder and its YoungAi headers are in tools/vq
#   - the k-quant reference encoders are vendored from llama.cpp in tools/kquant
#   - the four ggml entry points they need are implemented in tools/kquant/stubs.c
# The runtimes (llama.cpp and the XHToken fork) are separate: tools/runtime/
# build-runtime.sh fetches the pinned commits into vendor/ and builds them there.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CC=${CC:-cc}
CFLAGS=${CFLAGS:--O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L}

VQ_INC="-I $ROOT/tools/vq/vendor"
KQ_INC="-I $ROOT/tools/kquant/vendor/include -I $ROOT/tools/kquant/vendor/src -I $ROOT/tools/kquant/vendor/src/ggml-cpu"

mkdir -p "$ROOT/bin" "$ROOT/.build"
OBJ="$ROOT/.build"

# The vendored ggml quantizer, compiled once.
$CC $CFLAGS -c -o "$OBJ/ggml-quants.o" "$ROOT/tools/kquant/vendor/src/ggml-quants.c" $KQ_INC
echo "built $OBJ/ggml-quants.o"

$CC $CFLAGS $VQ_INC -o "$ROOT/bin/eg2_vq" "$ROOT/tools/vq/eg2_vq.c" -lm
echo "built $ROOT/bin/eg2_vq"

$CC $CFLAGS $VQ_INC -o "$ROOT/bin/rvq" "$ROOT/tools/vq/rvq.c" -lm
echo "built $ROOT/bin/rvq"

$CC $CFLAGS $VQ_INC -o "$ROOT/bin/plan" "$ROOT/tools/vq/plan.c"
echo "built $ROOT/bin/plan"

# Needs both the safetensors reader and the k-quant encoders, nothing else.
$CC $CFLAGS $VQ_INC $KQ_INC -o "$ROOT/bin/quant_err" \
  "$ROOT/tools/vq/quant_err.c" "$ROOT/tools/kquant/stubs.c" "$OBJ/ggml-quants.o" -lm
echo "built $ROOT/bin/quant_err"
