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
- VQ weights (per-tensor containers under `models/vq-build*`). Two runtimes
  read and run them: the CPU f32 reference (tools/vq-runtime) and the
  whole-model GPU runtime (tools/vq/gpu/vq-model-gpu.cu). Both are in section
  8b; the GPU one is gated against the CPU reference, decodes at 71 tok/s
  (24x the reference at the 4x8192 x2 point) and ingests a prompt at about
  850 tok/s through the tiled prefill. Its kernel shapes are measured per
  artifact by bin/vq-tune (section 8c), which writes the profile the runtime
  reads.

Know which one a request is about before acting.

## 2. Build everything

    ./tools/vq/build.sh

Builds `bin/eg2_vq`, `bin/rvq`, `bin/plan`, `bin/quant_err`, `bin/build-vq`
(CPU) and, when `nvcc` is present at /usr/local/cuda-13.3/bin/nvcc,
`bin/build-vq-gpu`. Nothing is read from outside the repository except the CUDA
toolkit and the checkpoint itself. The cargo workspace adds `bin/dqvq`,
`bin/vq-bench`, `bin/vq-runtime` and `bin/logits-diff`; the GPU block adds
`bin/decode-gpu` and `bin/vq-model-gpu`.

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

    ./bin/decode-gpu --manifest models/vq-build-8192-2-v3 --iters 10
    ./bin/decode-gpu --manifest models/vq-build-8192-2-v3 --iters 5 --prefill 512

It sums the fused gather+dot over every matrix in the manifest and prints the
container rate and the modeled tokens per second for matrix work at batch 1;
`--prefill N` adds an N-token GEMM through the same weight path. The Rust
harness prints the same numbers and names the kernel and prefill path it chose:

    ./bin/vq-bench models/vq-build-8192-2-v3 --iters 10 [--prefill 512]

Measured on the RTX 4070 SUPER, every artifact in the DQV3 row-local payload,
one run on 2026-10-31:

| Artifact | container | bpw | fused ms/token | GB/s | tok/s |
|---|---:|---:|---:|---:|---:|
| 4x1024 x1 | 1.289 GB | 2.51 | 4.080 | 316.0 | 245.1 |
| 4x8192 x2 | 3.375 GB | 6.57 | 10.844 | 311.2 | 92.2 |
| 4x4096 x3 | 4.655 GB | 9.06 | 13.322 | 349.4 | 75.1 |

Repeats of the 4x8192 x2 point in the same session landed between 10.18 and
10.84 ms (98 to 92 tok/s). The older DQVQ build of the same 4x8192 x2 weights
measured 14.629 ms (230.7 GB/s, 68.4 tok/s) in that run, and 13.25-14.57 ms in
the run before it. Two cautions follow from that:

- Compare kernels and artifacts only inside one run. The card's throughput
  moved by up to 30 percent between sessions on the same binary and artifact
  (a 2026-10-30 run put the DQVQ build at 10.27 ms), so an absolute number
  from another day is not a baseline.
- The kernel, not the payload, is where the 20-25 percent lives. On one matrix
  in one run: DQV3 block kernel 0.0717 ms, per-index dim-4 kernel 0.0881 ms.

The four fused variants, all in the same binary (`vqg_gemv_ready` says which a
geometry can run, and the report's `kernels` line names the one used):

- variant 3, DQV3 block decode. The row's 12-bit stream is read as 96-word
  blocks (three 128-byte lines): lane j holds words j, j+32, j+64, and a
  round's index word arrives by one shuffle plus a funnel shift; the 13th bit
  comes from that block's plane word by one more shuffle. This is the only
  kernel that uses the v3 payload's structure (ds4-dfm-rs
  `cuda/ds41_vq_row.cuh:122-137`, `:200-220`). It needs `nidx_row % 32 == 0`
  and a 4-byte-aligned row stream.
- variant 2, dim-4 fast path: one index at a time, one 8-byte `uint2` codebook
  entry, `__half22float2`, four indices deep. The DQVQ payload can only use
  this one, because its indices are not row-local.
- variant 1, generic warp: any dim, one index at a time, about 100 GB/s. This
  is what a non-dim-4 artifact falls back to.
- variant 0, shared codebook: only when `nc*dim*2` fits the device's opt-in
  shared window (dim 8 with 8192 codewords is 131,072 bytes and does not).

A plain f32 GEMV on the same card reaches 470 GB/s, so the best VQ rate is
about 70 percent of what this card demonstrably streams: the index gather and
the plane shuffles are issue- and latency-bound before DRAM is saturated.

## 8a. Measure the prefill (N tokens through the same weights)

Prefill is a real GEMM over the decoded matrix, so it needs tiles: one block
takes 128 weight rows x 128 tokens over all K, stepping 64 columns a round,
16 warps, f16 tensor cores with f32 accumulate (ds4-dfm-rs
`cuda/ds41_vq_prefill_mma.cuh:52-58`, `:82-94`, `:167-201`). The codebook is
copied into the A tile as f16 with no conversion because the payload's
codebook is already f16; the one named deviation from that reference is f16
instead of bf16 products, since bf16 would round a 10-mantissa-bit codeword by
2^-8 and ds4 only chose bf16 because its codebook is E4M3.

Measured on the RTX 4070 SUPER, 4x8192 x2, one session on 2026-10-31:

| tokens | prefill path | time | tok/s |
|---:|---|---:|---:|
| 128 | tiled f16, codebook in shared | 107.2 ms | 1193.8 |
| 128 | tiled f16, codebook through L1 | 121.8 ms | 1051.2 |
| 128 | legacy one-thread-per-(row,token) | 1569.96 ms | 81.5 |
| 512 | tiled f16, codebook in shared | 293.8 ms | 1742.8 |
| 512 | same, Rust harness | 269.9 ms | 1896.7 |

4x4096 x3 at 512 tokens: 424.2 ms, 1207.0 tok/s. 4x1024 x1 falls back to the
legacy path (its 10-bit stream is below the tiled kernel's 12-bit round map)
and reaches 168.8 tok/s at 128 tokens. The tiled path is 14.6x the legacy one
at 128 tokens, and token tiles run in parallel, so 512 tokens cost 2.7x the
128-token time rather than 4x.

`--prefill-variant` picks the path (0 auto, 1 tiled L1, 2 tiled shared,
3 legacy); the report's `prefill k` line names what ran. Gate: a host f64
matvec of the dequantised matrix at tokens 0, 1, 63, 64, 127, 128 and 511 (the
tile boundary and the second block), which every shape above passes at
relL2 0.0159 percent and maxabs 1.3e-3:

    ./bin/decode-gpu <file.bin> --iters 3 --prefill 128 --check

Two paths that are deliberately slower, both in the same binary:

- dequant to f32 (`dequant` line): 136 ms for the whole artifact at 24.8 GB/s.
  This is the expand-once loader shape, and it is about 13x slower than fusing.
- shared-memory codebook (variant 0): 38 GB/s; staging 64 KB per block costs
  more than the L1 misses it saves at batch 1. In prefill the same table in
  shared is the faster choice (107 ms against 122 for 128 tokens), because a
  block reuses every entry many times instead of once per row.

Container checks, independent of the encoder's own bookkeeping:

    ./bin/decode info   <file.bin>                      # header, layout, consistency
    ./bin/decode check  <file.bin> models/hf <tensor>   # relL2 against the checkpoint
    ./bin/decode dump   <file.bin> <out.f32>            # raw f32, for the Rust gate
    ./bin/dqvq check    <file.bin> <out.f32>            # Rust must match the C dump
    ./bin/decode-gpu    <file.bin> --check              # CUDA must match a host f64 matvec
    ./bin/decode-gpu    <file.bin> --prefill 128 --check # the same for a token batch

The C and Rust decoders are bit-exact against each other (26,214,400 of
26,214,400 values on layer 0), checked on five payload shapes: dim 4 with 1024,
4096 and 8192 codewords, dim 8 with 8192, and the DQVQ (v2) build of 8192. The
CUDA gate matches the host f64 matvec at maxabs 2e-6 on the same shapes, and
both CUDA prefill paths (tiled and legacy) match it at relL2 0.0159 percent.

Two widths are enforced, not just documented, because getting either wrong
returns different weights with no error:

- 13 bits per index. DQV3 carries 12 in the row's main stream plus one in the
  plane, so nc above 8192 cannot be encoded. `build-vq-gpu` refuses such an
  encoding and all three decoders refuse such a container, instead of decoding
  the low 13 bits of a 14-bit index.
- The main stream is min(bit width, 12) bits wide, not always 12. A 4096
  codebook packs 12-bit slots, a 1024 one packs 10-bit slots and shorter rows.
  All three decoders read a fixed 12 until 2026-10-31 and therefore decoded
  every payload below nc 4096 at the wrong offsets (the CUDA shared-codebook
  kernel even ran off the codebook and faulted). The width now comes from the
  header. The block and tiled kernels are 12-bit by construction, so
  `v3_blk_ok` and `pre_ok` refuse narrower streams and the generic kernels,
  which read `rd_idx_fmt`, serve them.

What these numbers are not: the matrix cost of one token with no attention, no
KV read and no sampling. They are the decode rate of the VQ path, not a server
throughput claim.

## 8b. Run the reference runtime, and gate it against bf16 and llama.cpp

`bin/vq-runtime` is the CPU f32 reference: it loads either the bf16 checkpoint
or an RQVQ artifact, runs the full Spark2_5 forward pass (36 layers, GQA, the
sliding windows, the per-head gate, exact erf gelu) and prints the next-token
distribution or a greedy continuation. It takes token ids, never text:

    IDS=$(curl -s http://127.0.0.1:8091/tokenize -H 'Content-Type: application/json' \
          -d '{"content":"The capital of France is","add_special":false}' \
          | sed -n 's/.*"tokens":\[\([^]]*\)\].*/\1/p')
    ./bin/vq-runtime --weights hf --dir models/hf --ids "$IDS" --gen 8 --logits /tmp/hf.f32
    ./bin/vq-runtime --weights vq --dir models/vq-build-8192-2-v3 --ids "$IDS" --gen 8 --logits /tmp/vq.f32
    ./bin/logits-diff /tmp/hf.f32 /tmp/vq.f32 --top 20

`logits-diff` prints the argmax agreement, logit relL2, cosine, both KL
directions, top-k overlap and the top ids with raw logit and log-softmax value.
It exits 0 when the argmax agrees, 1 when it does not, 2 on an error.

Measured 2026-10-31, prompt [1046, 11862, 314, 17349, 390] ("The capital of
France is", 5 tokens; the id list llama.cpp tokenizes with add_special false),
greedy:

| comparison | result |
|---|---|
| forward vs llama.cpp bf16 GGUF | top-20 ids identical, same order; log-softmax within 0.011 (19500: -0.8606 against -0.8592) |
| 4x8192 x2 vs bf16 | argmax 19500 both; logit relL2 11.52%, cosine 0.99896, KL 0.42 nats, top-10 5/10 |
| 4x4096 x3 vs bf16 | argmax 19500 both; logit relL2 9.51%, cosine 0.99915, KL 0.38 nats, top-10 6/10 |
| load | 28.9 s bf16 (7.7 GB), 20.6 s 4x8192 x2 (3.2 GB, decoded to f32), 16.1 GB resident |
| steady state, 32 generated tokens | 2.72 tok/s bf16, 2.78 tok/s VQ |

The forward-pass row is the important one: two independent implementations
(llama.cpp's bf16 kernels and this f32 pass) pick the same next tokens from the
same weights, so the loader, the rope, the sliding-window mask, the gate and
the tied head are all right. The VQ rows are then a measurement of the
quantization, not of a possible bug: both points keep the argmax, and the
higher-bitrate point is closer on every metric.

What these numbers are not: a perplexity or a full-text KL. They are one
prompt, one position, and greedy continuation. The 2.7-2.9 tok/s is the f32
reference streaming 16.45 GB per token.

`bin/vq-model-gpu` is the model-level GPU runtime. It loads the same artifact,
keeps every container on the device, and runs the whole forward pass there:
the fused DQV3 block GEMV (section 8) for every matrix, with rmsnorm (f64
accumulation like the reference), rope, GQA attention with the sliding window,
the per-head sigmoid gate, the exact erf gelu and the tied head as kernels in
tools/vq/gpu/vq-model-gpu.cu. Same ids, same --logits:

    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 --ids "$IDS" --gen 64 --ctx 512 --logits /tmp/gpu.f32
    ./bin/logits-diff /tmp/vq.f32 /tmp/gpu.f32 --top 20

The CPU reference is its acceptance test. Measured 2026-10-31:

| comparison | result |
|---|---|
| 4x8192 x2, GPU vs CPU reference, 5 tokens | argmax 19500 both; logit relL2 0.000027%, maxabs 1.9e-5, top-100 100/100 |
| 4x4096 x3, GPU vs CPU reference, 5 tokens | argmax 19500 both; logit relL2 0.000028%, maxabs 2.1e-5, top-25 25/25 |
| 4x8192 x2, GPU vs CPU, 600 tokens | argmax both; relL2 0.000863%, maxabs 9.2e-4, top-25 25/25 |
| generated ids, 8 tokens | identical on both: 19500, 27, 614, 11862, 314, 17349, 390, 19500 |
| load | 0.7 s for 3.37 GB (page-cached); 7.42 GB free of 12.47 GB after load |
| decode rate, ctx 512, 64 generated tokens | 70.8-71.8 tok/s (13.9-14.1 ms/token), three runs |

The 600-token row is the one that exercises the sliding mask (positions past
512) and rope far from 0; the 5-token rows only see the first few cache
entries. 71 tok/s is 24x the CPU reference on the same artifact and about 74
percent of the artifact-only block-kernel rate measured in the same session
(10.32 ms/token, 96.9 tok/s matrix-only; the section 8 figure drifts between
sessions); the rest is 218 kernel launches, the attention, the norms and one
device sync per token.

The tiled prefill is now wired into the model runtime's prompt path. The
prompt runs in chunks of up to 512 tokens through the tiled f16 mma kernel,
with batched causal attention over each token's own window, rope at each
token's absolute position and the same f32 residual stream. Gated against the
token-by-token path on the same binary with bin/logits-diff:

| comparison | result |
|---|---|
| 4x8192 x2, 5 tokens | argmax 19500 both; relL2 0.005455%, maxabs 4.9e-3, top-100 100/100, KL 0 both |
| 4x8192 x2, 600 tokens | argmax 1046 both; relL2 0.1501%, maxabs 0.169, top-25 25/25, top-100 99/100, KL 1e-6 |
| 4x4096 x3, 5 tokens | argmax 19500 both; relL2 0.009183%, maxabs 5.5e-3, top-25 25/25 |
| chunking | 600 tokens as 512+88 and as one chunk are bit-identical (relL2 0) |
| generated ids, 8 | identical to the token-by-token path on both artifacts |
| prompt rate | 128 tokens 851 tok/s (token path 66), 512 tokens 868 (62), 600 tokens 798-834 (62) |

The 0.0055-0.15% spread is the f16 activation conversion inside the tiled mma
kernel, not a logic difference: the token path itself matches the CPU f32
reference to 0.000027% at 5 tokens and 0.000863% at 600, and tiled against the
CPU at 5 tokens is 0.005458%, the same number as tiled against token. Prompt
ingestion is about half the artifact-only tiled rate (1743 tok/s at 512); the
rest is the batched attention, four transposes per layer, the norms and 512
embedding row dequeues, which is the micro-tuning target rather than a
correctness gap. An artifact with any matrix whose geometry the tiled kernel
cannot take (4x1024 x1 has a 10-bit stream) runs the prompt token by token and
prints that it does. `--no-prefill` forces the token-by-token path and
`--chunk N` sets the chunk size.

## 8c. Tune the VQ kernels per model (bin/vq-tune)

The kernel shapes are not build constants. The tiled prefill is a template over
its tile (BN token tile, BK column round) and its codebook placement, and the
GEMV takes a launch width. `bin/vq-tune` measures the legal choices on one
artifact's real payloads and writes `<artifact>/vq-tune.profile`.
`bin/vq-model-gpu` reads that file when it matches the artifact and the device.
`bin/vq-bench` does not read it: the harness keeps the readiness heuristic, so
its rates describe the default shapes (see the note at the end). Whenever a
profile is missing or refused, the runtime falls back to the heuristic and
prints why, on one line, before it loads weights.

Quick start:

    ./bin/vq-tune models/vq-build-8192-2-v3 --tokens 512 --iters 7
    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 --ids 1046,11862,314,17349,390 --gen 8 --ctx 512
    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 --ids <same> --gen 8 --ctx 512 --no-tune

The first command writes `models/vq-build-8192-2-v3/vq-tune.profile`; the
second run prints `profile ... matches artifact <hex> on <gpu> (12 entries,
driver 13.3)` and uses it; the third ignores it, which is the A/B baseline.

What the sweep does. It reads manifest.txt, groups the matrices into their
distinct geometries (six in a full model, not 216), asks the library which
GEMV variants, launch widths and prefill tiles each geometry can legally run,
times each on the real containers with CUDA events (`--iters` launches, best
of them), rejects any candidate whose output moves past relL2 1e-4 against the
default, and keeps the fastest. A 3.2 GB artifact sweeps in about four
seconds. Nothing is guessed: a variant or tile the kernel library reports as
illegal is never launched, and a rejected candidate is printed with its relL2.

Reading the output. One block per geometry, then two lines:

    2560x10240 dim 4 nc 8192 DQV3 (36 matrices, 2 stages)
      gemv    variant 3 threads 256   328.1 GB/s   (runner-up variant 3 threads 512: 324.9)
      prefill variant 2 tile 64x64   321246.4 tok/s   (runner-up variant 2 128x64: 272349.7)

The header is rows x cols, the payload width, the payload format, and how many
matrices share that geometry. The gemv rate is container bytes over the summed
best time of the matrix's stages (the same accumulation the runtime does). The
prefill rate is tokens over the same summed time at `--tokens`, and "legacy"
means no tiled tile fits that geometry. A tiny matrix (g_proj is 16 rows) shows
a low GB/s because the launch, not the bandwidth, is the cost; that is expected
and the tuner still records its winner.

The profile file is plain text, one directive per line, and is safe to read:

    VQPROF 1
    artifact 173b2f06d538deb8
    gpu NVIDIA GeForce RTX 4070 SUPER
    driver 13.3
    gemv 4 8192 2560 10240 1 3 256 328.1
    prefill 4 8192 2560 10240 1 2 64 64 321246.4

The fields after the tag are dim, nc, rows, cols, v3, then for `gemv` the
variant (0 shared codebook, 1 generic warp, 2 dim-4 warp, 3 DQV3 block), the
launch width, and the measured GB/s; for `prefill` the variant (1 tiled with
the codebook in L1, 2 tiled with it in shared, 3 legacy), the tile (BN, BK),
and the measured tok/s. The trailing rate is informational; the reader
ignores it. The file is written through a temporary and renamed, so a reader
never sees a partial profile, and it lives under `models/`, which git ignores.

When to re-tune: after building a new artifact (the fingerprint changes and the
old profile is refused), on a different GPU (the device name changes), and
after a toolkit or driver change if the margins matter (the driver is recorded
in the file but not matched, so a re-tune is recommended rather than enforced).
Do not copy a profile between artifacts to save four seconds.

Refusals are explicit and never silent. The runtime prints one of:

    profile     PATH is absent or not version 1; using heuristics
    profile     PATH was tuned on artifact <a>, this one is <b>; ignored
    profile     PATH was tuned on <gpu>, this device is <gpu>; ignored
    profile     none; DIR/manifest.txt could not be fingerprinted

and per entry, if a hand-edited or stale profile names something illegal:

    profile: gemv entry for RxC is not runnable here; heuristic kept

The fingerprint is FNV-1a over manifest.txt and the first and last 4 KB of
every matrix container, computed identically by the Rust writer and the C
reader. `--no-tune` skips the file entirely; `--profile PATH` overrides the
lookup path.

Useful variants of the tuner:

    ./bin/vq-tune DIR --dry-run                  # sweep and report, write nothing
    ./bin/vq-tune DIR --force                    # replace a profile from another artifact
    ./bin/vq-tune DIR --profile /tmp/p          # write elsewhere
    ./bin/vq-tune DIR --tokens 128 --iters 5    # judge a 128-token prompt instead

What is not tuned: the attention, norm, rope and transpose kernels, the
runtime's chunk size (tune with `--tokens` matching `--chunk` if you change
it), and anything per layer (matrices that share a geometry share an entry,
and the sweep shows the winner depends on the shape, not the layer). BK 128 is
also absent by construction (see the rules below).

Measured 2026-10-31, 4x8192 x2 (DQV3), 12 entries:

| geometry | matrices | decode GB/s | prefill tile | prefill tok/s |
|---|---:|---:|---|---:|
| 131072x2560 embedding | 1 | 378.0 | v1 256x64 | 22,083 |
| 2560x10240 down | 36 | 328.1 | v2 64x64 | 321,246 |
| 10240x2560 gate/up | 72 | 389.2 | v2 128x64 | 354,108 |
| 16x2560 g_proj | 36 | 16.1 | v1 64x64 | 4,273,504 |
| 2560x4096 out | 36 | 274.8 | v2 64x64 | 788,644 |
| 6144x2560 qkv | 36 | 337.7 | v2 64x64 | 545,275 |

The profile is numerically neutral, not merely inside the f16 gate: a tile
changes which tokens share a block, never the per-token K order, so logits
against `--no-tune` are relL2 0.000000%, maxabs 0, top-100 100/100, KL 0 both
directions. The model-level effect is the prompt rate: 500 tokens 917.8 tok/s
against 892.3 on the heuristic (+2.9%) on 4x8192 x2, and 749.4 against 734.8
(+2.0%) on 4x4096 x3. Decode is unchanged within run-to-run noise (64.3
against 62.4 tok/s in one pair of runs). Most of the prompt time is still the
batched attention, the transposes and the norms, not the matrix kernels.

DQVQ artifacts sweep too, but their payload has no tiled prefill path at all
(the round map reads the row-local DQV3 stream), so the profile records the
legacy prefill for every geometry and the runtime keeps the token-by-token
prompt and prints why. Only the GEMV choice differs there.

Two rules a change must preserve:

- A tiled tile is chosen whenever one fits; legacy is recorded only when none
  does. One legacy matrix switches the whole prompt off the chunked path.
- BK 128 is not offered. The A-tile extracts four code words from one 64-bit
  window; BK 128 needs eight (96 bits) and would truncate silently. Adding it
  means a three-word 96-bit window first, the shape v3_rounds already uses.

For the bench: `vq-bench` picks its own variants from readiness (block, then
dim-4, then generic; tiled with the shared codebook, then L1, then legacy) and
always uses the default 128x64 tile, so its artifact-only numbers are the
heuristic's ceiling. Compare the runtime against the bench only with that in
mind, and use `--no-tune` to reproduce the bench's matrix path in the runtime.

## 8d. The DSpark draft (speculative decoding for the VQ target)

The draft is a separate dflash checkpoint, models/dspark/Spark-X2.6-4B-DSpark.gguf
(1.25 GB, BF16 weights, F32 norms). It is not MTP inside the checkpoint: it has
no token_embd and no output weight, and it borrows the target's embedding and
LM head, which the VQ runtime has. bin/dspark dump prints the file; the reader
is tools/dspark/src/gguf.rs.

Architecture, read from the GGUF metadata (bin/dspark dump):

    dflash.block_count                5 blocks, all sliding-window
    dflash.embedding_length           2560
    dflash.feed_forward_length        10240, gelu (ggml's tanh form)
    dflash.attention.head_count       16 q heads / 4 kv, head_dim 256
    dflash.rope.dimension_count       64 full, 256 swa; theta 5e6 / 1e4
    dflash.attention.sliding_window   512, pattern [true x 5]
    dflash.attention.causal           false (block is attended non-causally)
    dflash.block_size                 7
    dflash.sample_from_anchor         true
    dflash.target_layers              [3, 11, 19, 26, 34]

Tensors: per block attn_norm, attn_q/k/v/out, attn_gate [2560x16], ffn_norm,
ffn_gate/up/down; plus fc [12800x2560] (five hidden states concatenated),
fc_norm [2560x5], enc.output_norm, output_norm, markov_w1 [256x131072],
markov_w2 [256x131072], conf_proj [2816x1]. 58 tensors, 1.249 GB.

The reference implementation is the XHToken llama.cpp fork at 22c0ae1,
src/models/dflash.cpp. It has two graph modes:

- Embd batch (feature injection): the target's hidden states at the five
  target_layers for the committed positions are normed per layer with fc_norm,
  projected by fc, normed by enc.output_norm, and their K/V are written into
  the draft's cache.
- Token batch (block diffusion): the block [anchor, MASK x 6] is embedded with
  the target's token_embd, run through the five SWA blocks with attn_gate
  (sigmoid per head before out_proj), and turned into logits by the target's
  LM head.

Then the DSpark Markov head rewrites those logits: for each position i of the
block, logits(i) += markov_w2 @ markov_w1[prev], where prev starts as the
anchor token and advances by argmax of the biased logits of position i. The
optional confidence head is sigmoid(conf_proj . [hidden(i); markov_w1[prev]]).
With sample_from_anchor true, position 0 is a real draft slot.

The support is metadata-driven by design: the converter emits a role manifest
(dimensions, rope, activation, window, block size, target layers and which
optional heads exist) and the executor takes all of it from that manifest, with
one tensor-name table and no per-checkpoint constants. The plan is handoff
section 4; the same rule as the kernel tuner applies, describe and measure the
model rather than specialize the tool to it.

Convert and check (phase A, done). The tool path is:

    ./bin/dspark dump models/dspark/Spark-X2.6-4B-DSpark.gguf
    ./bin/dspark convert models/dspark/Spark-X2.6-4B-DSpark.gguf out/draft [--force] [--verify]
    ./bin/dspark check out/draft

convert copies the tensor bytes into out/draft/weights.bin (64-byte aligned,
one region per tensor) and writes out/draft/manifest.txt, the role manifest.
It never decodes a weight; --verify re-reads the output and compares every
region with the source, which is what backs that claim. It refuses, by tensor
name, a quantized tensor, a name no role maps, a known role this build does not
implement (dflash2 conv, selector, lilicorr, attn_sinks, rope_freqs, a draft
that ships its own embedding or head), and a shape the metadata contradicts.
check validates the directory alone: every shape is re-derived from the
manifest's own header, the regions must tile weights.bin with only alignment
padding between them, and those padding bytes are read to prove they are zero.
A truncated tensor, an edited offset, a missing required role or a foreign
architecture is refused with the tensor or field named.

Measured 2026-10-31 on the released draft: 58 tensors, 1,248,919,044 bytes of
source weights, the same 1,248,919,044 bytes in weights.bin (this file needs no
alignment padding at all), conversion 3.7 s with --verify, and two runs produce
a bit-identical weights.bin and manifest. A copy with one tensor's byte count
reduced by 4 is refused as "tensor attn_q 0: 20971516 bytes recorded, but BF16
2560x4096 is 20971520 bytes"; a weights.bin truncated by 4 bytes is refused as
"the last tensor conf_bias ends at byte 1248919044, but weights.bin is
1248919040 bytes".

The manifest's tensor line is role, layer (0-based, -1 global), dtype code
(0 F32, 1 F16, 2 BF16), ne0, ne1, offset, bytes. Shapes are the GGUF shapes:
ne0 is the fastest-varying dimension, so a weight that turns a 2560-vector into
4096 values is 2560x4096, and fc is 12800x2560 (five target layers x hidden in,
hidden out). Three fields go beyond the handoff's first sketch: vocab, so the
Markov head validates without a target; mask_token, read from
tokenizer.ggml.mask_token_id and checked against vocab, because the block's
noise slots need it and the draft ships no vocabulary; and the conf_bias role,
because conf_proj ships a bias in the reference.

Status (2026-11-01, branch feature/dspark-vq): phases A, B, C and D are
implemented. The reader, the dump, the converter, the checker, the CPU draft
forward and the CPU propose/verify loop are phase A/B. Phase C is the GPU
executor: vq-model-gpu captures the target layer-input rows at layers 3, 11,
19, 26 and 34 into a position-major device buffer, verify_logits returns the
tied-head logits for every row of a verification batch, and dspark-gpu.cuh
runs the feature encoder, the block decoder (S independent blocks), the
Markov chain and the confidence head on the same device. The draft weights
are loaded as f32 from the manifest and multiplied with cuBLAS, so the GPU
follows the CPU gate; the target keeps its VQ kernels. The draft's K/V cache
holds the injected target features for committed positions, the block's own
K/V stay in the per-round row buffer, and no host round-trip is needed
between the target verify and the next draft block. Phase D is `--bench`,
which runs the spec path and the plain greedy baseline in one process and
prints the break-even acceptance.

Run one block from a capture (the first-block gate):

    ./bin/vq-runtime --weights hf --dir models/hf --ids 1046,11862,314,17349 \
        --capture-layers 3,11,19,26,34 --capture /tmp/hid.bin
    ./bin/dspark forward --draft out/draft --hidden /tmp/hid.bin \
        --target models/hf --anchor 390 --pos0 4

Run the full propose/verify loop (the 64-token gate):

    ./bin/dspark spec --draft out/draft --target models/hf \
        --ids 1046,11862,314,17349,390 --gen 64 --rounds --out /tmp/ids.txt
    ./bin/dspark spec --draft out/draft --target models/vq-build-8192-2-v3 \
        --weights vq --ids 1046,11862,314,17349,390 --gen 64

The GPU path runs the same loop inside vq-model-gpu, with the target and the
draft on one device and no host round-trip between a verify and the next
draft block:

    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 \
        --ids 1046,11862,314,17349,390 --gen 64 --ctx 4096 \
        --draft out/draft --out /tmp/gpu-ids.txt
    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 \
        --ids 1046,11862,314,17349,390 --ctx 4096 --draft out/draft --gate1
    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 \
        --ids 1046,11862,314,17349,390 --gen 64 --ctx 4096 \
        --draft out/draft --bench

The prompt ids are "The capital of France is" from llama.cpp /tokenize with
add_special false; the reference holds the last prompt token (390) back as the
first anchor, which is what both commands above do.

Measured gates, 2026-10-31:

- First block, BF16 and VQ captures: proposals [220 30 35 29 27 198 198],
  exactly the ids the instrumented reference fork logs for the same state
  (llama-speculative-simple, --spec-type draft-dspark, --spec-draft-n-max 7,
  temp 0, top-k 0). Per-row logits differ slightly between CPU f32 and the
  CUDA BF16 reference; every argmax is identical.
- 64-token greedy, BF16 target: generated ids identical to the reference
  fork's committed ids, 18 rounds, 126 drafted, 51 accepted (40.476%). 16 of
  18 full draft blocks are identical. Round 7's second proposal is a near-tie
  (ours 34:25.559708 against 614:25.553513; the reference 614:25.562506
  against 34:25.534792) and both are rejected, so the committed ids do not
  move. Round 18 differs only because the reference clamps the final block to
  the remaining generation budget while this loop keeps the full block; the
  first 64 ids still match.
- 64-token greedy, VQ 4x8192 x2 target: the speculative ids are identical to
  that VQ target's own greedy continuation (vq-runtime --weights vq --gen 64),
  11 rounds, 77 drafted, 53 accepted (68.831%). The VQ continuation is not the
  BF16 one: after the first tokens the 64-token run settles into the cycle
  19500, 390, 275, 11862, 314, 17349, 27. That is the quantization's effect
  on the target, not a chain error: the chain is exact for the target it
  runs.
- CPU speed is reference speed, not a product number: the 64-token BF16 chain
  takes 61.4 s of decode after a 26 s weight load, about 1.1 committed
  tok/s; the VQ chain takes 38.3 s of decode after a 20 s load.
- GPU phase C, 2026-11-01, VQ 4x8192 x2 target: the GPU spec ids are
  byte-identical to the CPU spec ids (11 rounds, 77 drafted, 53 accepted,
  68.831%). The verify tail uses per-row GEMV for the tied head; routing
  those logits through prefill_mat changed the first token (220 against the
  correct 19500), so verify_logits stays on the same GEMV as tail_logits.
- Gate 1, batch permutation invariance: four blocks with different anchors
  run in one batch and then in a permuted batch; maxabs 0 over 28 rows of
  131072 logits, no argmax or proposal mismatch.
- Phase D bench, same prompt and device: spec 89.25 tok/s against a 70.81
  tok/s greedy baseline (1.29x), round 63.8 ms (draft 13.9 ms, verify 49.6
  ms), break-even acceptance 50.3% per drafted token against the measured
  68.8%. The handoff's earlier ~0.3 estimate was optimistic: the measured
  draft cost puts break-even at 50.3%, and 68.8% still clears it.

Gate 1 is met. The GPU block decoder runs four independent blocks in one
batch, then the same blocks in a permuted order; the logits are bit-identical
(maxabs 0) and every argmax and proposal matches.

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
    tools/vq-format/   the one container decoder, linked by dqvq and the runtime
    tools/dqvq/        CLI over vq-format: info, check, gemv (bin/dqvq)
    tools/vq-runtime/  CPU f32 reference runtime and logits-diff (sections 8b)
    tools/vq/gpu/decode-gpu.cu  CUDA decode, GEMV, tiled prefill kernels
    tools/vq/gpu/decode-gpu.c   driver: per-tensor bench and `--manifest` bench
    tools/vq/gpu/vq-model-gpu.cu  whole-model GPU runtime (bin/vq-model-gpu)
    tools/vq/gpu/dspark-gpu.cuh  GPU draft executor, gates and bench (phases C/D)
    tools/vq/gpu/vq-profile.h   tuning profile format, fingerprint and reader
    tools/vq-tune/      per-model, per-kernel tuner (bin/vq-tune, section 8c)
    tools/dspark/       DSpark draft reader, converter and checker (bin/dspark, section 8d)
    tools/kquant/      vendored llama.cpp quantizer for the comparison numbers
    tools/runtime/     fetch and build the pinned llama.cpp trees
    models/hf/         the checkpoint the VQ path reads (not in git)
    models/*.gguf      k-quant artifacts (not in git)
    models/vq-build*/  VQ artifacts and manifests (not in git)
