# Spark-X2.6-4B-Base — local Q6_K build

A quantized, runnable build of
[XHToken/Spark-X2.6-4B-Base](https://huggingface.co/XHToken/Spark-X2.6-4B-Base)
for this machine: RTX 4070 SUPER 12 GB, sm_89, CUDA 13.3, 28 cores.

The model is a dense 4.11 B parameter, 36-layer trunk: hidden 2560, FFN 10240,
GQA 16/4 at head_dim 256, 27 sliding-512 and 9 full-attention layers, a per-head
sigmoid output gate, and a tied embedding that is also the output head. 8.22 GB in
BF16.

## The build, measured

| | |
|---|---|
| Artifact | `models/spark26-4b-Q6_K.gguf`, 3,460,437,312 B, 6.72 BPW |
| Type plan | Q6_K trunk, Q8_0 token embedding and output |
| Context | 131,072 tokens, one slot |
| Decode | 84.9 tok/s (11.78 ms/token) at a 128K allocation |
| VRAM | 9,465 MiB of 12,282 at load |
| Quality | not yet gated; see `RECIPE.md` section 7 |
| Published | no |

## Run it

    ./models/run-spark26.sh                 # 128K context, f16 KV, port 8090
    KV=q8_0 ./models/run-spark26.sh         # halve the KV if VRAM is tight
    CTX=32768 PORT=8091 ./models/run-spark26.sh

`llama-server` must be the one built here, not the stale binaries in
`~/.local/bin`:

    /data/llama.cpp/build/bin/llama-server

open-grok already knows the model as `spark2.6-4b-q6k`
(`http://127.0.0.1:8090/v1`, context 131072). Start the script first, then pick the
model in open-grok. This is a BASE checkpoint: it continues text rather than
following instructions.

## Speculative decoding with the DSpark draft

The upstream draft [`XHToken/Spark-X2.6-4B-DSpark-GGUF`](https://huggingface.co/XHToken/Spark-X2.6-4B-DSpark-GGUF)
(624 M parameters, 5 SWA layers, BF16, 1.25 GB) is the supported multi-token
accelerator for Spark: it is not a weight merge, it proposes up to 7 tokens per
pass and the target verifies them. It runs against two different targets here.

The default target is the official post-trained Q4_K_M, the pair the draft was
trained against and the only one that also behaves as an instruct model:

    ./models/run-spark26-dspark.sh
    MODEL="$PWD/models/spark26-4b-Q6_K.gguf" ./models/run-spark26-dspark.sh
    CTX=32768 ./models/run-spark26-dspark.sh

Both targets need the XHToken llama.cpp fork, not `/data/llama.cpp`: the released
draft stores a per-head `attn_gate` where the 2026-10-25 upstream `dflash` loader
expects `attn_q_norm`/`attn_k_norm`, so upstream fails with
`tensor 'blk.0.attn_q_norm.weight' not found`. Build the fork at
`/data/llama.cpp-xhtoken` (commit `22c0ae1`, plus an `#include <algorithm>` fix that
its `llama-context.cpp` is missing under gcc 14).

Measured on this machine, 128K context, q8_0 KV, chat requests at temperature 1.0
and top_p 0.95, 400 tokens, same fork binary:

| Target | Decode, no draft | Decode, with draft | Draft acceptance | Mean accept len | VRAM with draft |
|---|---:|---:|---:|---:|---:|
| Q4_K_M instruct (3.54 GB) | 107.1 / 106.7 / 107.7 tok/s | 122.7 / 120.8 / 149.8 tok/s | 0.22 / 0.21 / 0.31 | 2.55 / 2.46 / 3.17 | 8,837 MiB |
| Q6_K base (3.46 GB) | 98.1 / 94.7 / 94.6 tok/s | 123.0 / 125.4 tok/s | 0.20 / 0.23 | — | 10,876 MiB (f16 KV) |

So the pairing is a 1.13-1.40x gain of real decode speed here, not the roughly 2x
the draft's card reports on a 4090, and the instruct target does not raise
acceptance as much as its provenance suggested. Treat 1.1-1.4x as the honest
expectation on a 12 GB RTX 4070 SUPER. Quality is unchanged by the draft itself:
speculative decoding only changes which tokens are proposed, the target still
samples every accepted token.

One trap with the BASE target, measured today: under greedy decoding the chat
template drives it into an endless `\n\n\n<think>\n</think>` loop, and the loop
does not stop at EOS, so without a `max_tokens` it fills the whole context
(72,000 tokens observed). The unquantized BF16 build loops identically, so this is
model behavior, not quantization damage. Send temperature 0.8-1.0 with a small
repeat penalty, and always send `max_tokens`. The instruct target does not
reproduce the loop.

## Layout

    RECIPE.md         the recipe and the record: pins, type choice, commands, gate, release plan
    models/           the artifacts, the source checkpoint, logs, checksums, run script
      spark26-4b-Q6_K.gguf    3.46 GB, the artifact
      spark26-4b-bf16.gguf    8.22 GB, the gate reference (not for publishing)
      SHA256SUMS              both hashes
      run-spark26.sh          the server launcher for the local Q6_K build
      run-spark26-dspark.sh   the launcher with the DSpark draft (default target: official Q4_K_M)
      dspark/                 the DSpark draft GGUF and its run logs (not in git)
      instruct/               the official post-trained Q4_K_M target (not in git)
      hf/                     the source checkpoint at the pinned revision
      *.log                   download, convert, quantize, server logs
    docs/             gate receipts, once section 7 runs
    out/              build logs
    tools/            project-side tools, if any are needed

## Pins

- Weights: `XHToken/Spark-X2.6-4B-Base` rev `e1bd659ae2b1c2263b2fc2544485976c1b516115`, Apache-2.0.
- Toolchain: `/data/llama.cpp` @ `fa3645feb5f47cbb30df5e6d031c22543d43d240`, which
  implements `spark2_5` (converter, loader, tokenizer, quantizer).
- DSpark toolchain: `/data/llama.cpp-xhtoken` @ `22c0ae1871b60123c5afc008a78fe9c6fffa2858`
  (XHToken fork, CUDA sm_89) plus the `<algorithm>` include fix; required for the draft.
- Draft: `XHToken/Spark-X2.6-4B-DSpark-GGUF`, `Spark-X2.6-4B-DSpark.gguf`,
  1,254,293,824 B, sha256 `2ee073c18870c66d7fbc8e124fd8d6c2364c8a83e4bbbc3ebaeaa9660eedeac8`.
- Instruct target: `XHToken/Spark-X2.6-4B-GGUF`, `Spark-X2.6-4B-Q4_K_M.gguf`,
  3,543,441,312 B, sha256 `91cb2811aebb47ac916aa619d27d4cdd7f4143106b7917accb8ead4cd7f5cbd9`,
  Apache-2.0, post-trained, 290 tensors with the same names as the Base checkpoint.

## Before publishing

`RECIPE.md` section 7: perplexity and KL divergence against the BF16 GGUF, plus a
128K smoke test. Section 8 has the file list and the upload commands.
