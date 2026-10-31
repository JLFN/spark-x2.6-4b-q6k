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
| VQ 4x16384 | 3.54 | 12.48% | beats Q3_K slightly, loses to Q4_K badly |
| VQ 4x4096 | 3.01 | 16.89% | loses to Q3_K |
| Q2_K | 2.63 | 29.76% | |

Matrix: model.layers.0.mlp.down_proj.weight, 2560x10240 BF16, 26.2 M weights.
A second matrix (layer 18, same shape) reproduces the ordering: Q6_K 1.81%,
Q4_K 7.24%, Q3_K 15.31%, VQ 4x4096 17.15%.

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
    tools/vq/quant_err.c     new: llama.cpp k-quant error on one HF matrix
    tools/vq/build.sh        builds bin/eg2_vq and bin/quant_err
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

## 4. What a runnable VQ Spark still needs

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

## 5. Decision rule

Do not start the runtime work until the encoder shows VQ beating Q4_K per bit on
at least the FFN matrices, because that is the only outcome that would justify a
2.0 GB artifact over the existing files. Today it does not: 12.48% at 3.54 bpw
against 7.17% at 4.50 bpw. If the goal is only smaller files at Q3-class quality,
a plain Q3_K is the cheaper and better path, and it already runs on llama.cpp
with the DSpark draft.
