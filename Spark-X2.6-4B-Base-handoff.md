# Spark-X2.6-4B-Base — session handoff

Date: 2026-10-31. Repo: /data/Spark-X2.6-4B-Base, remote
https://github.com/JLFN/spark-x2.6-4b-q6k at aa10dcc, working tree clean.

This session took the repository from a Q6_K build record to a working VQ
toolchain, measured it, and wrote it down. Everything below was run and measured
here; nothing is a plan dressed as a result.

## 1. Where things stand

- Q6_K GGUF build: done, runs in llama.cpp, 3.46 GB, about 95-98 tok/s at 128K.
- DSpark speculative decoding: done, pairs with the official Q4_K_M instruct
  target, 1.13-1.40x decode, requires the XHToken fork.
- VQ toolchain: built, CPU and GPU, converts the checkpoint in 22 s on the GPU.
- VQ artifact: exists at models/vq-build-gpu (dim 4, 1024 codewords, 1 stage,
  1.29 GB, 2.43 bpw, per-tensor relL2 19-25%).
- VQ runtime: does not exist. Nothing decodes these containers yet.

## 2. What was measured (do not re-derive these)

Per-tensor, model.layers.0.mlp.down_proj.weight (26.2 M weights):

| Encoding | bpw | relL2 |
|---|---:|---:|
| Q8_0 | 8.50 | 0.54% |
| Q6_K | 6.56 | 1.79% |
| Q5_K | 5.50 | 3.63% |
| Q4_K | 4.50 | 7.17% |
| VQ 4x4096 x2 | 6.02 | 3.93% |
| VQ 4x2048 x2 | 5.51 | 4.88% |
| Q3_K | 3.44 | 15.17% |
| VQ 4x1024 x1 | 2.50 | 23.10% |
| Q2_K | 2.63 | 29.76% |

Conversion speed: GPU 0.2 s vs CPU 22.0 s for that matrix (about 110x); whole
checkpoint 22.4 s on GPU against 858.5 s on 12 CPU workers. Quality identical to
0.003 percentage points; the GPU is not byte-deterministic (FP64 atomics).

VRAM, 128K, q8_0 KV, measured on the card: Q4_K_M 7,127 MiB alone, 8,817 with the
draft; Q6_K 9,396 MiB at f16 KV. Planner predicts within 1.5-2.3%.

Context: config.json declares 1,048,576, and the planner defaults to it. At 1M
with a GQA cache the KV alone is 19,613 MiB at q8_0 (36,918 at f16), so 1M does
not fit this card; the repo serves 131,072. DeepSeek V4.1 reaches 0.93 GB at 1M
because its KV is a compressed latent carried by four source layers, which Spark
does not have. VQ does not change any of this.

## 3. Decisions taken, with reasons

- VQ is not a size win here. At Q5_K-class error it saves 335 MiB against Q6_K,
  and Q6_K is twice as accurate at nearly the same memory. Recorded in
  RECIPE-VQ section 1; do not relitigate without new measurements.
- The repo serves 131,072 because that is what fits, not because the model stops
  there. Earlier text in this session got that wrong and was corrected.
- The VQ path reads the HF checkpoint, never a GGUF. The tools read
  models/hf directly through the vendored safetensors reader.
- Everything needed to build is in-repo: YoungAi VQ headers, llama.cpp's
  ggml-quants.c, the CUDA encoder. The only external inputs are the checkpoint,
  the CUDA toolkit and (for runtimes) tool/runtime/build-runtime.sh fetching two
  pinned llama.cpp commits into vendor/.

## 4. Open work, in priority order

1. Calibration. The encoder already supports X-weighted assignment
   (1/diag(H^-1), GPTQ-consistent) but build-vq-gpu passes NULL. Needed: a Spark
   forward over a corpus that dumps per-tensor input activations, then a --stats
   path. Reference pipeline: /data/ds4-dfm-rs/tools/vq_calib (collect_stats.py,
   reduce_stats.py); Spark-relevant parts mapped in RECIPE-VQ section 6.
2. A runtime that decodes the containers. Until then the only quality signal is
   per-tensor relL2. Options: a loader that applies codebooks at load, or
   converting to a format an existing runtime understands.
3. Re-run the conversion at a rate that matters (4x4096, 2 stages) now that it
   costs seconds rather than an hour, and decide with data.
4. Optional: verification that decoding an RQVQ file reproduces the encoder's
   reconstruction (eg2_vq proves this for its own payloads; build-vq does not
   ship a checker yet).

## 5. How to resume in one screen

    ./tools/vq/build.sh                     # builds all tools, CPU + GPU
    ./bin/plan models/hf --vq 4 4096 2      # size and VRAM before converting
    ./bin/build-vq-gpu models/hf models/vq-build-gpu 4 1024 1
    ./bin/eg2_vq tensor models/hf model.layers.0.mlp.down_proj.weight 4 1024
    ./bin/quant_err models/hf model.layers.0.mlp.down_proj.weight q6_K
    ./models/run-spark26.sh                 # the GGUF server, port 8090
    ./models/run-spark26-dspark.sh          # the DSpark pairing, port 8090

AGENTS.md is the agent-facing version of this file: it has the commands, the
verification snippet, the pitfall list and the same honesty about VQ's place.

## 6. Traps that already cost time

- Two llama-server processes do not fit: 9-11 GB each. Kill by the pid on port
  8090, and confirm with nvidia-smi before starting another.
- `kill` did not stop one server; `kill -9` did. A stuck generation filled 72,012
  tokens because the client sent no max_tokens.
- Upstream llama.cpp cannot load the DSpark draft (attn_gate layout); only the
  XHToken fork at 22c0ae1 can, plus an <algorithm> include fix under gcc 14.
- The GPU encoder needs device pointers: pageableMemoryAccess is 0 on this card.
- quantize into Q5_K/Q4_K artifacts exists locally (models/spark26-4b-Q{5,4}_K.gguf)
  and is gitignored.
