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

$CC $CFLAGS $VQ_INC -o "$ROOT/bin/build-vq" "$ROOT/tools/vq/build-vq.c" -lm
echo "built $ROOT/bin/build-vq"

# Reference decoder: reads an RQVQ container and the HF checkpoint, nothing else.
# -ffp-contract=off keeps the accumulate in two roundings, which is what the Rust
# port does by default; without it GCC fuses into FMA and the two decoders differ
# in the last bit on about one value in a thousand.
$CC $CFLAGS -ffp-contract=off $VQ_INC -o "$ROOT/bin/decode" "$ROOT/tools/vq/decode.c" -lm
echo "built $ROOT/bin/decode"

# Rust tools: one cargo workspace, as ds4-dfm-rs is laid out. The canonical
# toolchain is a prerequisite, not an option: a build without cargo is an error,
# never a silently missing binary.
CARGO=${CARGO:-cargo}
if ! command -v "$CARGO" >/dev/null 2>&1; then
  echo "error: cargo not found; install the Rust toolchain from https://rustup.rs" >&2
  exit 1
fi
"$CARGO" build --release --manifest-path "$ROOT/tools/Cargo.toml"
cp "$ROOT/tools/target/release/dqvq" "$ROOT/bin/dqvq"
cp "$ROOT/tools/target/release/vq-bench" "$ROOT/bin/vq-bench"
cp "$ROOT/tools/target/release/vq-runtime" "$ROOT/bin/vq-runtime"
cp "$ROOT/tools/target/release/logits-diff" "$ROOT/bin/logits-diff"
cp "$ROOT/tools/target/release/vq-tune" "$ROOT/bin/vq-tune"
cp "$ROOT/tools/target/release/dspark" "$ROOT/bin/dspark"
echo "built $ROOT/bin/dqvq, $ROOT/bin/vq-bench, $ROOT/bin/vq-runtime, $ROOT/bin/logits-diff, $ROOT/bin/vq-tune and $ROOT/bin/dspark"

# Needs both the safetensors reader and the k-quant encoders, nothing else.
$CC $CFLAGS $VQ_INC $KQ_INC -o "$ROOT/bin/quant_err" \
  "$ROOT/tools/vq/quant_err.c" "$ROOT/tools/kquant/stubs.c" "$OBJ/ggml-quants.o" -lm
echo "built $ROOT/bin/quant_err"

# GPU path: device-resident Lloyd and assignment. Needs nvcc.
# Resolve the toolkit without pinning a version: an explicit NVCC wins, then
# PATH, then the unversioned /usr/local/cuda symlink. A hardcoded
# /usr/local/cuda-13.3 breaks the build the day the toolkit is upgraded.
if [ -z "${NVCC:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC=$(command -v nvcc)
  else
    NVCC=/usr/local/cuda/bin/nvcc
  fi
fi
ARCH=${ARCH:-89}
if [ -x "$NVCC" ]; then
  $NVCC -O3 -std=c++17 -arch=sm_$ARCH -c -o "$OBJ/vq_gpu.o" "$ROOT/tools/vq/gpu/vq_gpu.cu"
  $CC $CFLAGS $VQ_INC $KQ_INC -o "$ROOT/bin/build-vq-gpu" \
    "$ROOT/tools/vq/gpu/build-vq-gpu.c" "$OBJ/vq_gpu.o" -lm -lstdc++ -L/usr/local/cuda/lib64 -lcudart \
    -Wl,-rpath,/usr/local/cuda/lib64
  echo "built $ROOT/bin/build-vq-gpu"

  # CUDA decoder and decode-speed bench: one container, fused GEMV and dequant.
  $NVCC -O3 -std=c++17 -arch=sm_$ARCH -c -o "$OBJ/decode_gpu.o" "$ROOT/tools/vq/gpu/decode-gpu.cu"
  $CC $CFLAGS $VQ_INC -I/usr/local/cuda/include -o "$ROOT/bin/decode-gpu" \
    "$ROOT/tools/vq/gpu/decode-gpu.c" "$OBJ/decode_gpu.o" -lm -lstdc++ -L/usr/local/cuda/lib64 -lcudart \
    -Wl,-rpath,/usr/local/cuda/lib64
  echo "built $ROOT/bin/decode-gpu"

  # Whole-model GPU runtime: the same object's fused GEMV, plus the model's
  # norms, rope, attention, gate and MLP kernels.
  $NVCC -O3 -std=c++17 -arch=sm_$ARCH -o "$ROOT/bin/vq-model-gpu" \
    "$ROOT/tools/vq/gpu/vq-model-gpu.cu" "$OBJ/decode_gpu.o" -lm
  echo "built $ROOT/bin/vq-model-gpu"
else
  echo "skipped build-vq-gpu and decode-gpu: $NVCC not found"
fi
