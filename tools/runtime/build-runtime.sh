#!/usr/bin/env bash
# Build the two llama.cpp runtimes this repository pins, inside the repository.
#
# Nothing is read from /data or any other checkout: both trees are cloned from
# their upstream URLs at the recorded commits into vendor/ (gitignored) and built
# there. Re-running is safe; an existing checkout is fetched and checked out.
#
#   ./tools/runtime/build-runtime.sh            # both trees, CUDA if nvcc is found
#   CUDA=0 ./tools/runtime/build-runtime.sh     # CPU only
#   ONLY=upstream ./tools/runtime/build-runtime.sh
#
# Pins (README "Pins and artefacts"):
#   upstream ggml-org/llama.cpp at fa3645feb5f47cbb30df5e6d031c22543d43d240
#   XHToken/llama.cpp           at 22c0ae1871b60123c5afc008a78fe9c6fffa2858
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR="$ROOT/vendor"
CUDA=${CUDA:-auto}
ONLY=${ONLY:-both}
ARCH=${ARCH:-89}

UPSTREAM_URL="https://github.com/ggml-org/llama.cpp.git"
UPSTREAM_PIN="fa3645feb5f47cbb30df5e6d031c22543d43d240"
FORK_URL="https://github.com/XHToken/llama.cpp.git"
FORK_PIN="22c0ae1871b60123c5afc008a78fe9c6fffa2858"

mkdir -p "$VENDOR"

nvcc_path=""
if [ "$CUDA" != "0" ]; then
  for c in "$(command -v nvcc 2>/dev/null || true)" /usr/local/cuda/bin/nvcc /usr/local/cuda-13.3/bin/nvcc; do
    if [ -n "$c" ] && [ -x "$c" ]; then nvcc_path="$c"; break; fi
  done
fi

configure_and_build() {
  local dir="$1"
  if [ -n "$nvcc_path" ]; then
    cmake -B "$dir/build" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$nvcc_path"
  else
    echo "nvcc not found: building $dir for CPU only" >&2
    cmake -B "$dir/build" -G Ninja -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
  fi
  cmake --build "$dir/build" -j "${JOBS:-16}"
}

sync_tree() {
  local url="$1" pin="$2" dir="$3"
  if [ ! -d "$dir/.git" ]; then
    git clone "$url" "$dir"
  fi
  git -C "$dir" fetch --all --quiet
  git -C "$dir" checkout --quiet "$pin"
  echo "$dir at $(git -C "$dir" rev-parse --short HEAD)"
}

if [ "$ONLY" = "both" ] || [ "$ONLY" = "upstream" ]; then
  sync_tree "$UPSTREAM_URL" "$UPSTREAM_PIN" "$VENDOR/llama.cpp"
  configure_and_build "$VENDOR/llama.cpp"
fi

if [ "$ONLY" = "both" ] || [ "$ONLY" = "fork" ]; then
  sync_tree "$FORK_URL" "$FORK_PIN" "$VENDOR/llama.cpp-xhtoken"
  # The fork's llama-context.cpp calls std::all_of without including <algorithm>
  # and fails under gcc 14. Apply the one-line fix only if it is missing.
  f="$VENDOR/llama.cpp-xhtoken/src/llama-context.cpp"
  if ! grep -q '#include <algorithm>' "$f"; then
    sed -i 's|#include "llama.h"|#include "llama.h"\n\n#include <algorithm>|' "$f"
    echo "patched $f with <algorithm>"
  fi
  configure_and_build "$VENDOR/llama.cpp-xhtoken"
fi

echo "runtimes ready:"
echo "  $VENDOR/llama.cpp/build/bin/llama-server"
echo "  $VENDOR/llama.cpp-xhtoken/build/bin/llama-server"
