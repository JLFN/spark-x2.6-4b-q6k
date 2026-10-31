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

Measured on the RTX 4070 SUPER: 22.4 s for the whole checkpoint at 4x1024 x1 (the
artifact in `models/vq-build-gpu`), 88.5 s at 4x4096 x2 (3.104 GB) and 129.5 s at
4x8192 x2 (3.375 GB). Each tensor
is encoded with `vq_encode_full`'s algorithm (strided subsample of at most 200k
vectors, 8 Lloyd iterations, f16 codebook round-trip), with the loop and
assignment on the device. Each stage after the first trains on the residual left
by the previous one. Commit f5fb725 fixed a bug here: `stages` was accepted,
written into the header and ignored, so a 2-stage run stored a one-stage payload
and measured 16.90% instead of 3.94% on layer 0. The CPU path
(`./bin/build-vq ... --jobs 12`) is the byte-deterministic reference; a whole
checkpoint at one stage took 858 s there.

Output: one `.bin` per tensor plus `manifest.txt` (name, rows, cols, stages,
file_bytes, relL2). Containers are `[RQVQ header][stage payloads]`, and each stage
is exactly the bytes `vq_pack` writes: `[DQVQ][codebook nc*dim f16][row gains
f16][index stream]`.

## 5. How to know the quality

Quality is measured in four steps. Each step answers a different question and
costs a different amount of time, so run them in this order.

    decide rate  ->  sweep one matrix  ->  build the checkpoint  ->  compare to k-quants

### 5a. One matrix, one setting

    ./bin/eg2_vq tensor models/hf model.layers.0.mlp.down_proj.weight 4 4096
    ./bin/rvq    models/hf model.layers.0.mlp.down_proj.weight 4 4096 2

`eg2_vq` prints cos, relL2, maxabs, the exact payload bits per weight, and verifies
decode-vs-encode is exact (maxabs=0). `rvq` stacks residual stages and prints the
residual relL2 after each stage, which is how you see whether a second stage still
buys anything. On the CPU a 26 M-weight matrix takes about 2 min per stage at dim 4
/ 4096; on the GPU (`build-vq-gpu ... --match <name>`) the same matrix takes 0.6 s.

### 5b. The sweep: what a setting is worth

The sweep below is the one that produced the table. Ten settings, one matrix, a
few seconds each on the GPU.

    for cfg in "4 2048 2" "4 4096 2" "4 8192 2" "4 16384 2" \
               "4 4096 3" "4 8192 3" "4 4096 4" "8 4096 2" "8 8192 2" "8 4096 3"; do
      set -- $cfg
      ./bin/build-vq-gpu models/hf /tmp/sweep/$1-$2-$3 $1 $2 $3 \
          --match "layers.0.mlp.down_proj" >/dev/null
      awk -v c="$cfg" '!/^#/ {printf "%s  bpw %.3f  relL2 %.3f%%\n",
                               c, 8*$5/($2*$3), $6}' /tmp/sweep/$1-$2-$3/manifest.txt
    done

Measured on `model.layers.0.mlp.down_proj.weight` (26,214,400 weights):

| Setting | bpw | relL2 |
|---|---:|---:|
| Q8_0 | 8.50 | 0.54% |
| Q6_K | 6.56 | 1.79% |
| Q5_K | 5.50 | 3.63% |
| Q4_K | 4.50 | 7.17% |
| VQ 4x2048 x2 | 5.51 | 4.91% |
| VQ 4x4096 x2 | 6.02 | 3.94% |
| VQ 4x8192 x2 | 6.54 | 3.28% |
| VQ 4x16384 x2 | 7.08 | 2.81% |
| VQ 4x4096 x3 | 9.03 | 1.86% |
| VQ 4x8192 x3 | 9.81 | 1.75% |
| VQ 4x4096 x4 | 12.05 | 1.49% |
| VQ 8x4096 x2 | 3.04 | 17.88% |
| VQ 8x8192 x2 | 3.33 | 15.51% |
| VQ 8x4096 x3 | 4.56 | 8.15% |
| Q3_K | 3.44 | 15.17% |
| Q2_K | 2.63 | 29.76% |

Two things the table settles. Dim 8 is a bad trade on this model: it halves the
rate per stage but lands near Q3_K error at Q3_K size. Stages do keep paying, but
only at a rate nobody wants: 4x4096 x3 reaches Q6_K-class error at 9.03 bpw, which
is larger than the 8.50 bpw Q8_0 and 3.4x its error.

### 5c. The whole checkpoint

One matrix can mislead, because the last layers are harder than the first and the
tied embedding is a third of the weights. Build the checkpoint and aggregate the
manifest by weight count:

    ./bin/build-vq-gpu models/hf /tmp/vq-full 4 8192 2
    awk '!/^#/ && $6>=0 {w=$2*$3; tot+=w; se+=$6*$6*w; bytes+=$5}
         END {printf "%.3f GB, whole-model relL2 %.3f%%\n",
                      bytes/1e9, sqrt(se/tot)}' /tmp/vq-full/manifest.txt

Measured whole checkpoint (217 matrices, 4,111,892,480 weights, 128K rows are
resident size at q8_0 KV from `bin/plan`):

| Artifact | size | bpw | whole-model relL2 | resident at 128K |
|---|---:|---:|---:|---:|
| Q4_K | 2.68 GB | 4.83 | not measured | 6,143 MiB |
| Q5_K | 3.06 GB | 5.75 | 3.626% | 6,593 MiB |
| VQ 4x4096 x2 | 3.10 GB | 6.04 | 4.244% | 6,737 MiB |
| VQ 4x8192 x2 | 3.38 GB | 6.57 | 3.569% | 6,996 MiB |
| Q6_K | 3.46 GB | 6.72 | 1.784% | 7,072 MiB |
| VQ 4x4096 x3 | 4.66 GB | 9.06 | not measured (1.86% on layer 0) | 8,217 MiB |

The worst VQ tensor is the last layer, not the first: at 4x4096 x2 layer 35's
down_proj is 6.24% against layer 0's 3.93%. That is why the whole-model number
sits above the early-layer matrices.

### 5d. Against the k-quants, whole model

`bin/quant_err` takes one tensor at a time, so loop over the build's manifest and
weight the result the same way:

    while read name w; do
      r=$(./bin/quant_err models/hf "$name" q6_K | sed -n 's/.*relL2= *\([0-9.]*\)%.*/\1/p')
      echo "$name $w $r"
    done < <(awk '!/^#/ && $4>0 {print $1, $2*$3}' /tmp/vq-full/manifest.txt) \
    | awk '{tot+=$2; se+=$3*$3*$2} END {printf "%.3f%%\n", sqrt(se/tot)}'

That is the apples-to-apples number: same tensors, same weighting, one quantizer at
a time. Measured here: Q6_K 1.784%, Q5_K 3.626% over the same 4.11 B weights.

### 5e. Reading the result

At equal memory VQ loses on this model, and the loss is not small:

- At 3.38 GB, VQ 4x8192 x2 gives 3.57% against Q6_K's 1.78% at 3.46 GB.
- At 3.10 GB, VQ 4x4096 x2 gives 4.24% against Q5_K's 3.63% at 3.06 GB.
- VQ 4x8192 x2 is the one setting that edges Q5_K on the whole-model number
  (3.57% against 3.63%), but it pays 0.82 bpw and 0.32 GB for 0.06 points, and on
  the deep layers it is far worse: layer 35 down_proj 5.77% against Q5_K's 3.74%.
- The only VQ point that reaches Q6_K-class error is 4x4096 x3, at 4.66 GB and
  9.06 bpw, larger than Q8_0 and 3.4x its error.

So the recommendation is Q6_K for anything that has to run and be good, 4x8192 x2
if a VQ artifact is wanted anyway, and 4x4096 x2 only when 0.27 GB matters more
than 0.7 points of whole-model error. Do not present VQ as a size win at Q5_K
quality; it is not one on this checkpoint.

To raise quality further, in the order worth trying: calibration weighting (the
one lever that changes the curve, section 6), then more codebook entries, then more
stages. More dimensions is the wrong direction here.

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

Why this is the next lever: the encoder already contains the code that makes
calibration matter (`vq_qc.h` weights each 128-column group by `1/diag(H^-1)`,
the GPTQ-consistent cost, and records that plain `E[x^2]` weighting measured
worse). What is missing is the activation data, not the mathematics. Nothing is
measured on Spark yet, so treat any expected gain as unknown.

### 6b. `--domain` is not a VQ switch, and it does not exist here

Nothing in `tools/vq` accepts `--domain`; the flag belongs to the ds4 sidecar
mechanism, and the two are different things:

- In ds4, `/data/ds4-dfm-rs/install.sh:89` parses `--domain NAME`, line 105 maps
  it to a sidecar directory, and line 173 passes that directory to the server as
  `--zchain`. The directories are listed in `docs/deepseek41-flash.md:72`: five
  trained `…-engine` dirs, each `gr_Lnn.bin` (per-channel gains), `rb_Lnn.bin`
  (router bias), optional `amp_Lnn.bin` (low-rank `y += x*B*A`), a `manifest.txt`
  and a `base.fnv` binding it to one base model.
- Those files are produced by post-training and calibration losses. They are a
  behavior correction on top of a model, not a weight encoding, so "VQ with
  --domain" is not a combination that exists even in ds4.
- For Spark there is no sidecar, no runtime that could load one (llama.cpp has no
  sidecar concept; ds4-rs has no spark2_5 family), and no published training for
  it. RECIPE-DOMAIN.md records the file contract and the work it would take.

What the domain idea maps onto in this pipeline is calibration text: encode the
codebooks with activation statistics collected from the domain you care about.
That changes the one shared artifact; it does not give you switchable per-domain
codebooks, and nothing here implements a codebook swap. If a request asks for
`--domain`, answer that it is not a flag in this tool and offer calibration
instead.

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

Expected for the shipped dim 4 / 1024 / 1-stage artifact: 290 tensors, 0
mismatches, 1,290,051,957 bytes. For the 4x8192 x2 checkpoint: 290 tensors, 0
mismatches, 3,375,234,666 bytes.

## 8. Measure the decode rate (VQ only)

Three decoders read the same container, and the GPU one answers the speed
question in one command:

    ./bin/decode-gpu --manifest models/vq-build-8192-2 --iters 5

It sums the fused gather+dot over every matrix in the manifest and prints the
container rate and the modeled tokens per second for matrix work at batch 1.
Measured on the RTX 4070 SUPER, best of five:

| Artifact | container | bpw | fused ms/token | GB/s | tok/s |
|---|---:|---:|---:|---:|---:|
| 4x1024 x1 | 1.289 GB | 2.51 | 5.064 | 254.6 | 197.5 |
| 4x8192 x2 | 3.375 GB | 6.57 | 10.266 | 328.7 | 97.4 |
| 4x4096 x3 | 4.655 GB | 9.06 | 16.196 | 287.4 | 61.7 |

Decode time tracks container bytes almost exactly, so quality is bought with
decode time in direct proportion. The kernel matters: the first warp-per-row
version ran at 107 GB/s and 36.7 tok/s, and the current one reaches 329 GB/s by
loading one 8-byte `uint2` codebook entry per index, converting with
`__half22float2`, and unrolling four indices deep. A plain f32 GEMV on the same
card reaches 470 GB/s, so the gather still costs about 30 percent of bandwidth.

Two paths that are deliberately slower, both in the same binary:

- dequant to f32 (`dequant` line): 136 ms for the whole artifact at 24.8 GB/s.
  This is the expand-once loader shape, and it is 13x slower than fusing.
- shared-memory codebook (variant 0): 38 GB/s; staging 64 KB per block costs
  more than the L1 misses it saves.

Container checks, independent of the encoder's own bookkeeping:

    ./bin/decode info   <file.bin>                      # header, layout, consistency
    ./bin/decode check  <file.bin> models/hf <tensor>   # relL2 against the checkpoint
    ./bin/decode dump   <file.bin> <out.f32>            # raw f32, for the Rust gate
    ./bin/dqvq check    <file.bin> <out.f32>            # Rust must match the C dump
    ./bin/decode-gpu    <file.bin> --check              # CUDA must match a host f64 matvec

The C and Rust decoders are bit-exact against each other (26,214,400 of
26,214,400 values on layer 0). The CUDA gate matches the host f64 matvec at
maxabs 3e-6.

What these numbers are not: the matrix cost of one token with no attention, no
KV read and no sampling. They are the decode rate of the VQ path, not a server
throughput claim.

## 9. Pitfalls that cost time here

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

## 10. File map

    RECIPE.md          the Q6_K build record and the release plan
    RECIPE-VQ.md       the VQ investigation: measurements, GPU path, calibration map
    RECIPE-DOMAIN.md   the ds4 domain-sidecar mechanism and what Spark would need
    tools/vq/          encoder, residuals, planner, builders (CPU and GPU)
    tools/vq/decode.c  C reference decoder: info, check, dump, matvec
    tools/vq/decode-rs/ Rust decoder and fused GEMV, no crates (bin/dqvq)
    tools/vq/gpu/decode-gpu.cu  CUDA decode and GEMV kernels
    tools/vq/gpu/decode-gpu.c   driver: per-tensor bench and `--manifest` bench
    tools/kquant/      vendored llama.cpp quantizer for the comparison numbers
    tools/runtime/     fetch and build the pinned llama.cpp trees
    models/hf/         the checkpoint the VQ path reads (not in git)
    models/*.gguf      k-quant artifacts (not in git)
    models/vq-build*/  VQ artifacts and manifests (not in git)
