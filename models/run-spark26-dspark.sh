#!/usr/bin/env bash
# Serve the Q6_K target with the Spark-X2.6-4B-DSpark draft for speculative decoding.
#
# The draft needs the XHToken llama.cpp fork: the released Spark-X2.6-4B-DSpark.gguf
# stores a per-head attn_gate instead of attn_q_norm/attn_k_norm, and the only loader
# that accepts both layouts is the fork's has_attn_gate branch (src/models/dflash.cpp
# in XHToken/llama.cpp at 22c0ae1871b6). Upstream fa3645feb fails with
# "tensor 'blk.0.attn_q_norm.weight' not found" (measured 2026-10-31).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN=${BIN:-/data/llama.cpp-xhtoken/build/bin/llama-server}
MODEL=${MODEL:-$ROOT/models/spark26-4b-Q6_K.gguf}
DRAFT=${DRAFT:-$ROOT/models/dspark/Spark-X2.6-4B-DSpark.gguf}
ALIAS=${ALIAS:-spark-dspark}
CTX=${CTX:-131072}
PORT=${PORT:-8090}
KV=${KV:-q8_0}          # f16 also fits: 10,876 MiB of 12,282, only 1,018 MiB free

exec "$BIN" \
  -m "$MODEL" \
  -md "$DRAFT" \
  --alias "$ALIAS" \
  --spec-type draft-dspark \
  --spec-draft-sampling greedy \
  --spec-draft-n-max 7 \
  --spec-draft-p-min 0 \
  -ngl 999 -ngld 999 \
  -ctk "$KV" -ctv "$KV" -ctkd f16 -ctvd f16 \
  -fa on \
  -c "$CTX" -np 1 -b 2048 -ub 512 -t 8 \
  --jinja \
  --host 127.0.0.1 --port "$PORT" \
  "$@"
