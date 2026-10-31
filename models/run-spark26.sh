#!/usr/bin/env bash
# Spark-X2.6-4B-Base, Q6_K trunk + Q8_0 head, served by llama.cpp on this box.
#
#   ./run-spark26.sh                 # 131072 context, f16 KV
#   KV=q8_0 ./run-spark26.sh         # halve the KV if VRAM is tight
#   CTX=32768 ./run-spark26.sh       # smaller window
#   PORT=8091 ./run-spark26.sh --verbose-prompt
#
# This is a BASE model: it continues text. open-grok talks chat_completions to it,
# so --jinja applies the checkpoint's own chat template; expect completion-style
# behaviour, not an instruct model's.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL=${MODEL:-$ROOT/models/spark26-4b-Q6_K.gguf}
# Built by tools/runtime/build-runtime.sh inside this repository.
BIN=${BIN:-$ROOT/vendor/llama.cpp/build/bin/llama-server}
if [ ! -x "$BIN" ] && [ -x /data/llama.cpp/build/bin/llama-server ]; then
  BIN=/data/llama.cpp/build/bin/llama-server
  echo "note: using $BIN; run tools/runtime/build-runtime.sh for an in-repo build" >&2
fi
CTX=${CTX:-131072}
PORT=${PORT:-8090}
KV=${KV:-f16}
ALIAS=${ALIAS:-spark2.6-4b-q6k}

[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 1; }
[[ -x "$BIN" ]]   || { echo "missing llama-server: $BIN" >&2; exit 1; }

kv_args=()
case "$KV" in
  f16)  ;;
  q8_0) kv_args+=(-ctk q8_0 -ctv q8_0) ;;
  *)    echo "KV must be f16 or q8_0 (got: $KV)" >&2; exit 2 ;;
esac

# VRAM budget before we start: weights + KV. 9 of 36 layers keep full attention
# (the other 27 slide a 512-token window, so their cache stays small); per token
# and per full layer the KV is 4 kv heads x 256 dims x 2 (K,V) x bytes-per-value.
python3 - "$MODEL" "$CTX" "$KV" <<'PY'
import os, sys
model, ctx, kv = sys.argv[1], int(sys.argv[2]), sys.argv[3]
w = os.path.getsize(model)
b = 2 if kv == "f16" else 1
full = 9 * 4 * 256 * 2 * b * ctx
win  = 27 * 4 * 256 * 2 * b * 512
total = w + full + win
print(f"weights {w/2**30:5.2f} GiB | KV {full/2**30:5.2f} GiB (f16-equivalent scaled by {b}) "
      f"| about {total/2**30:5.2f} GiB of 11.99 GiB before compute buffers", file=sys.stderr)
PY

exec "$BIN" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" \
  -ngl 999 \
  -fa on \
  -np 1 \
  --jinja \
  "${kv_args[@]}" \
  "$@"
