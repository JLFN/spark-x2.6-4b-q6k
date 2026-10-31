# AGENTS.md — how to build a VQ Spark, measure its quality, and know its size

This file is for an agent working in this repository. It assumes nothing beyond
this file and the checkout. Every command here was run on this machine on
2026-10-31; the numbers quoted are measured, not estimated.

## 1. What this repository does

It converts the HF checkpoint in `models/hf` (Spark-X2.6-4B, 5 safetensors
shards, BF16) into dense VQ weights with the tools in `tools/vq`, measures the
reconstruction quality and predicts the memory the result needs.

Two artifact families live here and they are not the same thing:

- GGUF k-quants (`models/spark26-4b-Q6_K.gguf`, plus Q5_K and Q4_K builds). These
  run in llama.cpp today and are what the server scripts use.
- VQ weights (per-tensor containers under `models/vq-build*`). Nothing serves
  these yet; there is no runtime that decodes them.

Know which one a request is about before acting.

## 2. Build everything

    ./tools/vq/build.sh

Builds `bin/eg2_vq`, `bin/rvq`, `bin/plan`, `bin/quant_err`, `bin/build-vq`
(CPU) and, when `nvcc` is present at /usr/local/cuda-13.3/bin/nvcc,
`bin/build-vq-gpu`. Nothing is read from outside the repository except the CUDA
toolkit and the checkpoint itself.

Runtimes are separate: `./tools/runtime/build-runtime.sh` clones and builds the
two pinned llama.cpp trees into `vendor/` (gitignored).

## 3. How big will a VQ build be? (size, before converting)

    ./bin/plan models/hf --vq 4 4096 2 --vq-embed
    ./bin/plan models/hf --ctx 1048576 --kv q8_0 --budget 9000
    ./bin/plan models/hf --vq 4 2048 2 --draft models/dspark/Spark-X2.6-4B-DSpark.gguf

`plan` reads `config.json` for the context (1,048,576 on this checkpoint), the
tensor inventory for the weights, and prints artifact GB, bits per weight, KV,
buffers, total and headroom against a 12,282 MiB card. `--kv-bytes N` replaces the
computed KV with a measured whole-model bytes-per-token figure.

Byte accounting is exact for VQ (the same payload formula the encoder writes) and
validated against real artifacts: Q6_K predicts 3,456 MB against 3,460 MB shipped,
and total VRAM is within 1.5-2.3% of the measured runs at 128K and 32K.

## 4. Convert (fast path: GPU)

    ./bin/build-vq-gpu models/hf models/vq-build-gpu 4 2048 2

22 s for the whole checkpoint at dim 4 / 1024 codewords on this card; one stage.
Each tensor is encoded with `vq_encode_full`'s algorithm (strided subsample of at
most 200k vectors, 8 Lloyd iterations, f16 codebook round-trip), with the loop and
assignment on the device. The CPU path (`./bin/build-vq ... --jobs 12`) is the
byte-deterministic reference and took 858 s for the same output.

Output: one `.bin` per tensor plus `manifest.txt` (name, rows, cols, stages,
file_bytes, relL2). Containers are `[RQVQ header][stage payloads]`, and each stage
is exactly the bytes `vq_pack` writes: `[DQVQ][codebook nc*dim f16][row gains
f16][index stream]`.

## 5. How to know the quality

Three levels, cheapest first:

1. Per tensor, one matrix: `./bin/eg2_vq tensor models/hf <name> 4 4096` prints
   cos, relL2, maxabs, bpw and verifies decode-vs-encode is exact (maxabs=0).
   Takes about 2 min per 26 M-weight matrix.
2. Whole build: the `manifest.txt` relL2 column from `build-vq-gpu`.
3. Against k-quants on the same matrix: `./bin/quant_err models/hf <name> q6_K`
   (q8_0/q6_K/q5_K/q4_K/q3_K/q2_K).

Measured reference points on `model.layers.0.mlp.down_proj.weight` (26.2 M
weights), so any new setting can be placed:

| Encoding | bpw | relL2 |
|---|---:|---:|
| Q8_0 | 8.50 | 0.54% |
| Q6_K | 6.56 | 1.79% |
| Q5_K | 5.50 | 3.63% |
| Q4_K | 4.50 | 7.17% |
| VQ 4x4096 x2 stages | 6.02 | 3.93% |
| VQ 4x2048 x2 stages | 5.51 | 4.88% |
| Q3_K | 3.44 | 15.17% |
| VQ 4x1024 x1 (this artifact) | 2.50 | 23.10% |
| Q2_K | 2.63 | 29.76% |

Rule of thumb from those numbers: VQ loses to k-quants per bit on this model.
Q6_K beats the two-stage VQ at nearly the same memory. Say so when asked; do not
present VQ as a size win at Q5_K quality.

To raise quality further: more codebook entries (`nc` 4096), more dimensions
(dim 8 lowers rate), more stages (residual, `bin/rvq`), and calibration weighting.
All trade rate for error; check with `eg2_vq` before claiming a gain.

## 6. Calibration (the remaining quality lever)

`vq_encode_full` accepts `X`, an `[n_act x cols]` block of input activations, and
weights codebook assignment by `1/diag(H^-1)` (GPTQ-consistent). `build-vq-gpu`
currently passes NULL, so gains are 1.0 and assignment is unweighted.

To calibrate: run the checkpoint with its own `modeling_spark.py` over a corpus,
capture per-tensor input activations, save them, then extend `build-vq-gpu` to
load that block and call the weighted path. The ds4 pipeline to copy is
`/data/ds4-dfm-rs/tools/vq_calib` (collect_stats.py, reduce_stats.py,
axk2_weighting.py); its family-specific parts are reviewed in RECIPE-VQ.md
section 6. Until that exists, per-tensor relL2 is the only quality signal.

## 7. Verifying an artifact

    python3 - <<'EOF'
    import os
    d = {}
    for line in open('models/vq-build-gpu/manifest.txt'):
        if line.startswith('#'): continue
        p = line.split()
        d[p[0]] = int(p[4])
    bad = [k for k, v in d.items()
           if os.path.getsize(os.path.join('models/vq-build-gpu', k + '.bin')) != v]
    print(len(d), 'tensors,', len(bad), 'mismatches')
    EOF

Expected: 290 tensors, 0 mismatches, 1,290,051,957 bytes at dim 4 / 1024 / 1 stage.

## 8. Pitfalls that cost time here

- Do not trust a hardcoded context. `config.json` says 1,048,576; the repo serves
  131,072 because a GQA cache for 1M is 19,613 MiB at q8_0 on a 12,282 MiB card.
- The 48 KiB shared-memory gate: dim 4 with 4096 codewords is 81,920 bytes per
  codebook and does not fit one tile. The local kernels multi-pass the codebook,
  so they are not affected; older reference kernels are.
- `pageableMemoryAccess = 0` on this card. Kernels that take host pointers will
  fault; keep pointers device-side (as vq_gpu.cu does).
- The GPU path is not byte-deterministic (FP64 atomics). Use the CPU path when
  bytes must be identical.
- 1-D tensors are copied as f32 with a 12-byte `RQVF` header; the manifest counts
  that header now.
- The Q6_K Base artifact loops forever under greedy chat decoding. That is the
  BASE checkpoint, not damage: BF16 loops identically. Send temperature 0.8-1.0
  and always send max_tokens.

## 9. File map

    RECIPE.md          the Q6_K build record and the release plan
    RECIPE-VQ.md       the VQ investigation: measurements, GPU path, calibration map
    RECIPE-DOMAIN.md   the ds4 domain-sidecar mechanism and what Spark would need
    tools/vq/          encoder, residuals, planner, builders (CPU and GPU)
    tools/kquant/      vendored llama.cpp quantizer for the comparison numbers
    tools/runtime/     fetch and build the pinned llama.cpp trees
    models/hf/         the checkpoint the VQ path reads (not in git)
    models/*.gguf      k-quant artifacts (not in git)
    models/vq-build*/  VQ artifacts and manifests (not in git)
