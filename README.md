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

## Layout

    RECIPE.md         the recipe and the record: pins, type choice, commands, gate, release plan
    models/           the artifacts, the source checkpoint, logs, checksums, run script
      spark26-4b-Q6_K.gguf    3.46 GB, the artifact
      spark26-4b-bf16.gguf    8.22 GB, the gate reference (not for publishing)
      SHA256SUMS              both hashes
      run-spark26.sh          the server launcher
      hf/                     the source checkpoint at the pinned revision
      *.log                   download, convert, quantize, server logs
    docs/             gate receipts, once section 7 runs
    out/              build logs
    tools/            project-side tools, if any are needed

## Pins

- Weights: `XHToken/Spark-X2.6-4B-Base` rev `e1bd659ae2b1c2263b2fc2544485976c1b516115`, Apache-2.0.
- Toolchain: `/data/llama.cpp` @ `fa3645feb5f47cbb30df5e6d031c22543d43d240`, which
  implements `spark2_5` (converter, loader, tokenizer, quantizer).

## Before publishing

`RECIPE.md` section 7: perplexity and KL divergence against the BF16 GGUF, plus a
128K smoke test. Section 8 has the file list and the upload commands.
