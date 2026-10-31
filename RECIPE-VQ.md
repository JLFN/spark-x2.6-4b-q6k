# Recipe: a VQ build of Spark-X2.6-4B

Status: encoder in-repo and working; two runtimes load and run the artifacts.
The CPU f32 reference is gated against the bf16 checkpoint and against
llama.cpp, and the GPU model runtime is gated against that reference; it
decodes at 71 tok/s on 4x8192 x2, 24x the reference, and ingests a prompt at
about 850 tok/s on the tiled prefill (section 8, item 2, and AGENTS.md
section 8b). The measurements below are the reason the next step is
a decision, not a build order, and the runtime did not change the verdict on
quality. AGENTS.md section 5 carries the fuller sweep, the whole-checkpoint
builds and the whole-model k-quant comparison measured after this file was
written.

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
for 3.93%. VQ is therefore dominated on this model at every budget, which is
why the runtime port was not started on that evidence. It was started later, on
the operator's instruction, as a reference implementation (section 8, item 2);
it does not change this table.

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
    tools/vq/build-vq.c      new: whole-checkpoint builder (per-tensor containers + manifest)
    tools/vq/build.sh        builds bin/eg2_vq, rvq, quant_err, plan and build-vq
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

## 5. Building the artifact

build-vq walks the checkpoint index and writes one container per tensor plus a
manifest, in parallel:

    ./bin/build-vq models/hf models/vq-build 4 1024 1 --jobs 12
    # encoding: dim=4 nc=1024 stages=1
    # 290 tensors: 288 matrices encoded, 2 norms copied as f32

Per tensor the container is [RQVQ header][stage 1][stage 2]..., where each stage
is exactly the payload vq_pack writes ([DQVQ header][codebook nc*dim f16][row
gains f16][index stream]). The manifest records name, rows, cols, stages and the
file size, so the artifact is reproducible and checkable without the encoder.

A 26.2 M-weight matrix at dim 4 / 1024 codewords takes about a minute on one
core, so the whole checkpoint needs a parallel run; --jobs 12 is the setting used
here. The embedding is encoded too unless --no-embed is passed, since on this
model it is a tied 131,072 x 2,560 matrix and a large share of the file.

## 5b. GPU conversion

The GPU path is built and measured. build-vq-gpu runs Lloyd and the assignment on
the card with device pointers, so nothing crosses PCIe inside the loop, and the
codebook is staged in 1024-codeword shared-memory tiles instead of the reference's
single tile.

| Work | CPU (1 core) | GPU | Speedup |
|---|---:|---:|---:|
| one 2560 x 10240 matrix, dim 4, nc 1024 | 22.0 s | 0.2 s | about 110x |
| whole checkpoint, 217 matrices + 73 copied norms | about 11 min (12 workers, unfinished) | 22.4 s | about 30x against the parallel CPU build |

Quality is unchanged: the same matrix reconstructs at relL2 23.0985% on the CPU and
23.1014% on the GPU, with the same 8,205,329-byte payload. The small difference is
FP64 atomic summation order in the centroid update, which also means the GPU path
is not byte-deterministic run to run; the CPU path remains the reference when
identical bytes matter.

    ./bin/build-vq-gpu models/hf models/vq-build-gpu 4 1024 1
    # 217 tensors in 22.4 s, 1.3 GB, manifest with per-tensor relL2

Why the reference kernels could not be used as-is, measured here: this card
reports pageableMemoryAccess = 0 (probe, RTX 4070 SUPER sm_89, 56 SMs), so the
ds4 kernels that take host pointers fault unless driven through pinned host
memory, and the embeddinggemma run measured that path 1.25-3x slower than CPU per
tensor on this card because of PCIe. Its single-tile shared-memory gate (48 KiB)
also refuses dim 4 with 4096 codewords (81,920 bytes). The local kernels avoid
both problems: device pointers, and tiled codewords.

## 6. Calibrated VQ: what the ds4 toolchain has

/data/ds4-dfm-rs/tools/vq_calib is the calibration pipeline that produced a
calibrated VQ release. Its README splits the pieces into family-specific and
family-independent, which is the map to follow for Spark:

| Piece | In vq_calib | Spark |
|---|---|---|
| source reader | axk2_source.py, FP8 with 128x128 block scales | not needed: Spark ships BF16 safetensors |
| reference forward + stats hooks | axk2_ref.py, selftest_ref.py | must be written against modeling_spark.py |
| calibration corpus | build_calib.py (seeded, pinned revisions, think and non-think) | adapt to Spark's tokenizer and chat template |
| statistics | collect_stats.py, run_collect.sh, reduce_stats.py | same shape, one rank per GPU, FP64 sums |
| weighting | axk2_weighting.py (none/row/imatrix/imatrix_freq/output) | the MoE modes do not apply; see below |
| weighted Lloyd and pack | cuda/axk2_vq_convert.cu, axk2_vq_gpu.py | already present in tools/vq/vendor |
| container and release | axk2_vq_format.py, assemble_release.py, publish_release.py | a Spark writer has to be written |
| evaluation | measure_layer.py, eval_e2e.py, router_mask_ab.py | held-out KL and teacher-forced agreement |

The important discovery is that the vendored encoder already implements
calibration-weighted assignment: vq_encode_full takes X, an [n_act x cols] block
of input activations, and computes per 128-column group H = X^T X, inverts it, and
weights each column by 1/diag(H^-1), the GPTQ-consistent cost. The comment in
vq_qc.h records why: plain E[x^2] weighting measured worse (KLD 0.46186 ->
0.48138) because it over-protects columns whose error later columns can
compensate. So the encoder side needs no new code; what is missing is X.

What Spark needs, in order:

1. An activation collector. Load models/hf with transformers and the checkpoint's
   own modeling_spark.py (trust_remote_code), run a calibration corpus, and dump
   per-tensor input activations for the matrices being quantized. On 12 GB this
   has to stream layer by layer and write chunks, the way collect_stats.py does
   for its family; a full 128-batch capture does not fit.
2. A --stats path in build-vq that feeds those chunks to vq_encode_full instead
   of NULL.
3. A held-out evaluation. Per-tensor relL2 (bin/quant_err, eg2_vq) is a proxy;
   the real gate is KL on held-out text against the BF16 model, which needs a
   forward that can run the VQ weights. Until Spark has a runtime that decodes
   VQ, the gate stays at per-tensor and per-layer error.
4. A Spark container writer and loader, for the same reason.

Not needed for Spark: the MoE-specific weighting modes, the router-bias A/B, the
FP8 source reader and the non-expert Q4_K path. Spark's 36 layers are dense and
its matrices are plain BF16.

## 7. The VQ path reads the checkpoint, not a GGUF

The conversion starts from the HF checkpoint in models/hf: 5 safetensors shards,
model.safetensors.index.json, config.json and the tokenizer files. Every tool in
tools/vq reads that directly through the vendored ds4_st41 index reader, so no
GGUF is involved at any point of the VQ pipeline. The tools validate the shards
by name and shape; the local copy matches the Hub file for file on all five
weight shards and on every tokenizer/config file.

The context comes from the same config. config.json declares
max_position_embeddings = 1048576, and plan reads that field, so its default
window is the model's own 1M rather than a number typed into a script.

The attention geometry, also from config.json:

| Field | Value |
|---|---|
| num_hidden_layers | 36 |
| num_attention_heads / num_key_value_heads | 16 / 4 |
| head_dim | 256 |
| layer_types | 27 sliding_attention, 9 full_attention |
| sliding_window | 512 |
| max_position_embeddings | 1,048,576 |

That decides the KV: nine full layers hold the whole window, so at 1,048,576
tokens the cache is 9 x 1,048,576 x 2 x 4 x 256 values, which is 38.7 GB at f16
(36.0 GiB) and 19,613 MiB at q8_0. A plain 4-bit packing is about 10.4 GiB.

For contrast, and not as a template: DeepSeek-V4.1 in ds4 keeps one KV head with a
512-dim latent in FP4 with E4M3 scales per 16-token group, carried by four source
layers, about 890 bytes per token, 0.93 GB at 1M. That is an architectural
compressor. Spark stores real K and V for nine layers, and no weight encoding
changes that.

plan models any of this directly with --kv-bytes N (a whole-model bytes-per-token
figure):

    ./bin/plan models/hf --ctx 1048576 --kv-bytes 890
    # Q6_K  3.46 GB  weights 3,295  KV 890  buffers 1,300  total 5,485 MiB

The 1M table from the checkpoint itself, with an ordinary GQA cache:

    ./bin/plan models/hf --ctx 1048576 --kv q8_0 \
        --vq 4 2048 1 --vq 4 2048 2 --vq 4 4096 1 --vq 4 4096 2 --vq 8 4096 2 --vq 4 8192 2

    encoding           artifact    weights       KV   buffers     total   headroom
    VQ 4x2048 x1        1.66 GB     1,584    19,613    1,300    22,497    -10,215
    VQ 4x2048 x2        2.97 GB     2,828    19,613    1,300    23,740    -11,458
    VQ 4x4096 x1        1.78 GB     1,700    19,613    1,300    22,613    -10,331
    VQ 4x4096 x2        3.21 GB     3,060    19,613    1,300    23,972    -11,690
    VQ 8x4096 x2        1.81 GB     1,723    19,613    1,300    22,635    -10,353
    VQ 4x8192 x2        3.46 GB     3,298    19,613    1,300    24,211    -11,929
    Q8_0                4.37 GB     4,167    19,613    1,300    25,080    -12,798
    Q6_K                3.46 GB     3,295    19,613    1,300    24,208    -11,926
    Q5_K                2.95 GB     2,817    19,613    1,300    23,729    -11,447
    Q4_K                2.48 GB     2,367    19,613    1,300    23,279    -10,997

Nothing fits at 1M on this card with a GQA cache: the KV alone is 19,613 MiB.
With a 4-bit packed KV (--kv-bytes 10368) the VQ build drops to 14,628 MiB, still
2,346 MiB over. Only a latent cache in V4.1's class (--kv-bytes 890) fits, and
that cache does not exist for Spark.

Caveat on the buffer term: 1,300 MiB is measured at 32K and 128K, where it ranged
1,079-1,434 MiB. It is not measured at 1M and may grow with context, so treat any
1M row as a lower bound of a few hundred MiB.

## 8. What a runnable VQ Spark still needs

1. Encoder pass over the full model. eg2_vq handles one tensor at a time and reads
   HF safetensors. A driver must walk the 290 tensors, keep 1-D tensors
   (norms) verbatim, decide per-matrix rate, and write a container. The reference
   container is DQVD with a recipe string (see
   /data/embeddinggemma2/RECIPE.md section 4d); the GGUF wrapping must name a
   family the runtime knows.

2. A runtime that decodes VQ. A CPU f32 reference now exists
   (tools/vq-runtime): it loads the bf16 checkpoint or an RQVQ artifact, runs
   the full Spark2_5 forward pass over token ids, and writes the logits so a
   second binary (bin/logits-diff) can price the quantization. Gated
   2026-10-31 on the prompt "The capital of France is": against llama.cpp's
   bf16 GGUF the top 20 next tokens are the same ids in the same order with
   log-softmax within 0.011, so the forward pass itself is verified; against
   bf16, 4x8192 x2 keeps the argmax with logit relL2 11.52% and KL 0.42 nats,
   and 4x4096 x3 is closer at 9.51% and 0.38. It is a reference, not a speed
   path: 16.45 GB of f32 weights per token, 2.7-2.9 tok/s, 16.1 GB resident.
   A second runtime, tools/vq/gpu/vq-model-gpu.cu, runs the whole forward
   pass on the GPU over the same artifact and is gated against the reference:
   at the same prompt the logits agree to relL2 0.000027% (maxabs 1.9e-5,
   top-100 100/100) and the 8 generated ids are identical; over a 600-token
   prompt, which exercises the sliding window past position 512, relL2 is
   0.000863% with the same argmax. It decodes at 70.8-71.8 tok/s (13.9-14.1
   ms/token) at ctx 512 on 4x8192 x2, 24x the CPU reference and about 74
   percent of the artifact-only block-kernel rate measured in the same session
   (10.32 ms/token). The prompt path is wired to the tiled f16 prefill:
   against the token-by-token path the logits agree to relL2 0.0055% at 5
   tokens (0.15% over 600 tokens, KL 1e-6, top-25 25/25), the generated ids
   are identical, and prompt ingestion is 851-868 tok/s at 128-512 tokens
   (14x the token path). The residual error is the f16 activation conversion
   in the mma kernel; the token path itself is three orders of magnitude
   closer to the CPU reference. A 600-token prompt split across two chunks is
   bit-identical to one chunk. The speed gap against llama.cpp is unchanged:
   - llama.cpp, which runs Spark today, has no codebook type at all: its ftypes
     are F16, Q4_0, Q4_1, Q8_0, Q5_0, Q5_1, Q2_K..Q6_K (include/llama.h:120).
     A VQ Spark on llama.cpp means a new ggml type plus a CUDA GEMV/GEMM path for
     spark2_5, which is the bulk of the work.
   - ds4-rs decodes DQVQ, but it has no Spark family: searching its crates for
     spark2_5 returns nothing, and the gemma_vq path is an embedding-only tower.
     This route means a full Spark2_5 family port into ds4.
   What the reference runtime did settle: the container, the decoder and a
   correct forward pass are no longer open questions. The block GEMV and the
   tiled prefill are both wired into the GPU model runtime now (299 GB/s and
   71 tok/s decode end to end; 851-868 tok/s prompt ingestion); the remaining
   speed work is closing the gap between the model-level prompt rate and the
   artifact-only 1743 tok/s, which is attention and layout overhead rather
   than matrix work.

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

## 9. Decision rule

Do not start the runtime work until the encoder shows VQ beating Q4_K per bit on
at least the FFN matrices, because that is the only outcome that would justify a
2.0 GB artifact over the existing files. Today it does not: 12.48% at 3.54 bpw
against 7.17% at 4.50 bpw. If the goal is only smaller files at Q3-class quality,
a plain Q3_K is the cheaper and better path, and it already runs on llama.cpp
with the DSpark draft.
