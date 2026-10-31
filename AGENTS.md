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
- VQ weights (per-tensor containers under `models/vq-build*`). A CPU f32
  reference runtime reads and runs them (tools/vq-runtime, section 8b). It is
  correct but reference-grade speed, and no fast VQ runtime exists yet.

Know which one a request is about before acting.

## 2. Build everything

    ./tools/vq/build.sh

Builds `bin/eg2_vq`, `bin/rvq`, `bin/plan`, `bin/quant_err`, `bin/build-vq`
(CPU) and, when `nvcc` is present at /usr/local/cuda-13.3/bin/nvcc,
`bin/build-vq-gpu`. Nothing is read from outside the repository except the CUDA
toolkit and the checkpoint itself. The cargo workspace adds `bin/dqvq`,
`bin/vq-bench`, `bin/vq-runtime` and `bin/logits-diff`.

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
reference streaming 16.45 GB per token; the block GEMV and the tiled prefill in
section 8 are what a fast runtime would use instead.

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
    tools/vq/gpu/decode-gpu.cu  CUDA decode and GEMV kernels
    tools/vq/gpu/decode-gpu.c   driver: per-tensor bench and `--manifest` bench
    tools/kquant/      vendored llama.cpp quantizer for the comparison numbers
    tools/runtime/     fetch and build the pinned llama.cpp trees
    models/hf/         the checkpoint the VQ path reads (not in git)
    models/*.gguf      k-quant artifacts (not in git)
    models/vq-build*/  VQ artifacts and manifests (not in git)
