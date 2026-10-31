# Recipe: a VQ build of Spark-X2.6-4B

Status: encoder in-repo and working; no runnable VQ Spark yet. The measurements
below are the reason the next step is a decision, not a build order.

## 1. Verdict first

Dense VQ, as implemented in the ds4 toolchain, is not competitive with the
k-quants on Spark's matrices at equal bit rate. Measured on the same tensors with
the tools in this repo:

| Encoding | bpw | relL2 | Note |
|---|---:|---:|---|
| Q8_0 | 8.50 | 0.54% | |
| Q6_K | 6.56 | 1.79% | the artifact this repo ships |
| Q5_K | 5.50 | 3.63% | |
| Q4_K | 4.50 | 7.17% | |
| Q3_K | 3.44 | 15.17% | |
| VQ 4x4096 x2 stages | 6.02 | 3.93% | Q5_K-class error, 0.52 bpw more expensive |
| VQ 4x2048 x2 stages | 5.51 | 4.88% | same rate as Q5_K, 34% worse error |
| VQ 4x16384 | 3.54 | 12.48% | beats Q3_K slightly |
| VQ 4x4096 | 3.01 | 16.89% | loses to Q3_K at 3.44 |
| Q2_K | 2.63 | 29.76% | |

Matrix: model.layers.0.mlp.down_proj.weight, 2560x10240 BF16, 26.2 M weights.
A second matrix (layer 18, same shape) reproduces the ordering: Q6_K 1.81%,
Q4_K 7.24%, Q3_K 15.31%, VQ 4x4096 17.15%.

Residual VQ (tools/vq/rvq.c, new) stacks codebook stages on the previous
residual. That is the tool built to answer the Q5_K question, and the answer is
measured: two stages at dim 4, 4096 codewords reach 3.93% at 6.02 bpw, so VQ
does reach Q5_K-class quality, but it needs about 0.5 bpw more than Q5_K to get
there. At Q5_K's own 5.50 bpw, two VQ stages sit at 4.88%, a third worse.

At equal memory, Q6_K still wins outright: 6.56 bpw for 1.79%, against 6.02 bpw
for 3.93%. VQ is therefore dominated on this model at every budget, which is the
reason the runtime port is not worth starting.

So a VQ Spark lands in Q3_K class at about 3.5 bpw, roughly a 1.8-2.1 GB artifact
against the 3.46 GB Q6_K, and that estimate is unmeasured. The EmbeddingGemma VQ
result does not transfer: an embedding model tolerates 3 bpw because its cosine
stays at 0.99, while a 4B causal LM at Q3_K class loses real quality.

Proceed only if minimum size at Q3-class quality is the actual goal. For
everything else the existing Q6_K and the official Q4_K_M are better artifacts.

## 2. What was moved into this repo

The encoder used to live in /data/embeddinggemma2 and read headers from
/data/YoungAi. It is now self-contained here.

    tools/vq/eg2_vq.c        the encoder (copy, only include paths changed)
    tools/vq/vendor/*.h      YoungAi VQ headers, MIT, pinned to a4e6d31
    tools/vq/vendor/ds41_vq_fmt.h   the on-disk format contract
    tools/vq/quant_err.c     new: k-quant error on one HF matrix
    tools/vq/rvq.c           new: residual (multi-stage) VQ encoder
    tools/vq/plan.c          new: artifact size and VRAM planner with --budget
    tools/vq/build.sh        builds bin/eg2_vq, rvq, quant_err and plan
    tools/kquant/vendor/     llama.cpp's ggml-quants.c and its headers, MIT
    tools/kquant/stubs.c     the four ggml entry points that file needs
    tools/runtime/build-runtime.sh  clones and builds the pinned llama.cpp trees
    THIRD_PARTY.md           provenance and licenses

Build and verify:

    ./tools/vq/build.sh
    ./bin/eg2_vq selftest

Expected selftest on this machine, byte-identical to the reference tool:

    selftest/hot-4x512    dim=4 nc=512  cos=0.9946  relL2=10.4161%  2.4068 bpw
    selftest/cold-8x256   dim=8 nc=256  cos=0.9685  relL2=24.8898%  1.1567 bpw

## 3. Measure one tensor

    ./bin/eg2_vq tensor models/hf model.layers.0.mlp.down_proj.weight 4 4096
    ./bin/quant_err models/hf model.layers.0.mlp.down_proj.weight q6_K

eg2_vq reports cos, relL2, maxabs and the exact payload bits per weight. quant_err
calls llama.cpp's reference quantizers, so the comparison is like for like.
Timing on this CPU: 109 s for dim4/nc4096 on a 26 M-weight matrix, 445 s for
dim4/nc16384.

## 4. Choosing a rate from a RAM budget: bin/plan

plan answers the question directly: if the model is converted this way, how much
disk and how much VRAM will it take. It reads the checkpoint inventory and prints,
per encoding, the exact stored bytes, bits per weight, KV at the chosen context,
the measured compute buffers and the total against the card.

    ./bin/plan models/hf                            # standard table, 128K, f16 KV
    ./bin/plan models/hf --vq 4 4096 2 --vq-embed   # a VQ build, embedding included
    ./bin/plan models/hf --ctx 65536 --kv q8_0 --budget 9000
    ./bin/plan models/hf --vq 4 4096 2 --draft models/dspark/Spark-X2.6-4B-DSpark.gguf
    ./bin/plan models/hf --vq 4 4096 2 --json       # one JSON object per row

Measured output for the two-stage VQ at 128K with q8_0 KV:

    encoding           artifact      bpw   weights        KV   buffers      total  headroom
                             GB                MiB       MiB       MiB        MiB       MiB
    VQ 4x4096 x2           3.10    6.039      2960      2477      1300       6737      5545   relL2 3.93%
    Q8_0                   4.37    8.501      4167      2477      1300       7944      4338   relL2 0.54%
    Q6_K                   3.46    6.722      3295      2477      1300       7072      5210   relL2 1.79%
    Q5_K                   2.95    5.746      2817      2477      1300       6593      5689   relL2 3.63%
    Q4_K                   2.48    4.828      2367      2477      1300       6143      6139   relL2 7.17%

So a VQ build costs 3.10 GB on disk and 6,737 MiB resident at 128K with q8_0 KV,
against 7,072 MiB for the Q6_K that measures twice as accurate. The draft adds
1,710 MiB to every row when --draft is passed.

Byte accounting is exact: VQ rows use the same payload formula the encoder writes,
k-quant rows use the block sizes (Q8_0 34/32, Q6_K 210/256, Q5_K 176/256,
Q4_K 144/256), and 1-D norms stay f32. Validation:

| Item | Predicted | Actual | Error |
|---|---:|---:|---:|
| Q6_K artifact | 3,456 MB | 3,460 MB | 0.1% |
| Q6_K at 128K f16 KV | 9,258 MiB | 9,396 MiB | 1.5% |
| Q6_K at 32K f16 KV | 5,455 MiB | 5,585 MiB | 2.3% |

The Q4_K row is a plain Q4_K, not the official Q4_K_M mix, so it under-predicts
that file by about 1 GiB; use the Q4_K_M row's real size (3,543,441,312 B) when
budgeting that artifact.

## 5. KV cost: the runtime's, not the weight encoding's

In ds4 the KV is much smaller than in llama.cpp, and that is worth separating from
the VQ question because it is a property of the runtime, not of the weights.

Measured and recorded in the ds4 tree:

| Family | KV layout | Cost |
|---|---|---|
| DeepSeek-V4.1 | packed FP4: main KV 512 dims = 256 B nibbles + 32 B E4M3 scales = 288 B per group; index K = 64 B nibbles + 4 B E8M0, padded to 72 B (ds41_kvfmt.h:7-11) | one KV head, 512-dim latent |
| A.X-K2 (VQ) | 770 bytes per token: a 641-byte latent row plus a 129-byte index key (docs/axk2-vq.md) | 11.47 GiB per 262,144-token bank |
| Beam-VQ | sliding layers keep f16 rings of 512; full layers keep FP8 rows, 2,080 bytes per token and layer (docs/beam-vq.md:48) | 18.28 GiB at 1M |

That compactness comes from those families' latent or compressed KV designs and
their kernels. It is not produced by the weight VQ: the same packed KV would serve
a Q6_K build of the same model.

Spark has 4 KV heads of 256 dims, K and V, on 9 full-attention layers, so a
token costs 2,048 elements per full layer whatever the weights are. Packed like
ds4's FP4 that is about 1,152 B per token per layer, 10,368 B per token for the
whole model, about 1.30 GiB at 128K. A 770-B/token latent like A.X-K2 would need
a trained compressor that Spark does not have.

plan models this directly: --kv-bytes N replaces the computed KV with a measured
per-token figure.

    ./bin/plan models/hf --vq 4 4096 2 --vq-embed --kv-bytes 770 --ctx 131072
    # VQ 4x4096 x2   3.10 GB   2,960 weights  +  96 KV  + 1,300 buffers = 4,356 MiB

The model's own limit is 1,048,576 tokens (config.json
max_position_embeddings), so plan defaults to 1M and only serves a smaller window
when told to. At 1M the GQA KV is the binding constraint on this card: 19,613 MiB
at q8_0 and 36,919 MiB at f16, against 12,282 MiB total, so 1M does not fit at
all with a GQA cache; 131,072 is what this repository actually serves.

So in a runtime that packs its KV, the VQ build would sit at 4,356 MiB rather than
6,737 MiB, and the KV term stops mattering. Two things follow. The weight saving
from VQ stays at 335 MiB either way, so it does not become more attractive. And
the KV saving is available to any Spark runtime that implements a packed KV,
including one serving the existing Q6_K.

At 1M the same packing changes the verdict on context: 10,368 B/token at a plain
4-bit packing is about 10.4 GiB, still too large beside 2,960 MiB of weights, so
only a latent scheme in A.X-K2's class (770 B/token, 807 MiB at 1M) would make the
full window reachable, and Spark has no such compressor.

## 6. What a runnable VQ Spark still needs

1. Encoder pass over the full model. eg2_vq handles one tensor at a time and reads
   HF safetensors. A driver must walk the 290 tensors, keep 1-D tensors
   (norms) verbatim, decide per-matrix rate, and write a container. The reference
   container is DQVD with a recipe string (see
   /data/embeddinggemma2/RECIPE.md section 4d); the GGUF wrapping must name a
   family the runtime knows.

2. A runtime that decodes VQ. This is the real gap.
   - llama.cpp, which runs Spark today, has no codebook type at all: its ftypes
     are F16, Q4_0, Q4_1, Q8_0, Q5_0, Q5_1, Q2_K..Q6_K (include/llama.h:120).
     A VQ Spark on llama.cpp means a new ggml type plus a CUDA GEMV/GEMM path for
     spark2_5, which is the bulk of the work.
   - ds4-rs decodes DQVQ, but it has no Spark family: searching its crates for
     spark2_5 returns nothing, and the gemma_vq path is an embedding-only tower.
     This route means a full Spark2_5 family port into ds4.

3. MTP / DSpark on top. The draft model reads target hidden states at layers
   2, 10, 18, 25 and 33. Weight format does not matter to the draft; what matters
   is that those activations match the BF16 target. VQ weights shift them, so
   acceptance will fall against the 0.21-0.31 measured for Q4_K_M, and the drop
   is unmeasured. A VQ build must therefore implement the draft path in the same
   runtime that decodes VQ, and re-measure acceptance. Nothing in llama.cpp or
   ds4-rs does both today.

4. The quality gate. RECIPE.md section 7 applies unchanged: perplexity and KL
   against models/spark26-4b-bf16.gguf, plus a decode-speed and acceptance run of
   the paired draft. A Q3_K-class artifact must clear the same bar as the Q6_K
   before it is publishable.

## 7. Decision rule

Do not start the runtime work until the encoder shows VQ beating Q4_K per bit on
at least the FFN matrices, because that is the only outcome that would justify a
2.0 GB artifact over the existing files. Today it does not: 12.48% at 3.54 bpw
against 7.17% at 4.50 bpw. If the goal is only smaller files at Q3-class quality,
a plain Q3_K is the cheaper and better path, and it already runs on llama.cpp
with the DSpark draft.
