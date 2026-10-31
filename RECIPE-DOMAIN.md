# Recipe: domain sidecars for Spark (the ds4 "domain" mechanism)

Status: nothing exists for Spark yet. This document records how DeepSeek-V4.1
does it in /data/ds4-dfm-rs, what of that applies to a dense Spark model, and the
concrete steps to build the same capability here.

## 1. What "loading a domain" means in ds4

On DeepSeek-V4.1-Flash the server is started with a domain sidecar directory:

    ./ds4-server --cuda -m <base>.gguf --zchain <domain sidecar> ...

    bash install.sh --domain code        # domains: finance, code, law, medicine, science

The artifact ships five directories,
`...-grrb-{vqfin41_vqhalf_a_n8192,code_fit_n15360,law_fit_n15360,med_fit_n15360,sci_fit_n15360}-engine`
(docs/deepseek41-flash.md:72). Each holds per-layer files plus a fingerprint:

| File | Size | Meaning |
|---|---:|---|
| `gr_Lnn.bin` | 1,044,492 B | per-expert per-channel gain factors |
| `rb_Lnn.bin` | n_expert * 4 + 8 B | router-bias delta added to the selection score |
| `amp_Lnn.bin` | varies | low-rank correction `y += x * (B * A)` at the forward tail |
| `manifest.txt` | text | the directory's own record |
| `base.fnv` | 20 B | fingerprint binding the sidecar to one base model |

A sidecar is produced by post-training (zfinetune plus the four calibration
losses), not by quantization. It changes behavior without touching the base.

## 2. The exact file contract

Read from the ported reader, crates/ds4-core/src/sidecar.rs, which cites the C
engine (/data/YoungAi core_v41_amp.c):

- gr (sidecar.rs:133): header is 3 little-endian i32 [n_expert, d, type] where d
  is n_embd and type is 0 = f32, 1 = f16, 2 = fp4x32 (stored as s-1, restored as
  1+raw). Body is n_expert * d factors. Merge rule (zchain.rs:4-10): every
  present gr file multiplies into one table, so ② and ③ compose.
- rb (sidecar.rs:207): header 2 i32 [n_expert, 1], body is n_expert f32. Comes
  from the ② directory only; the device table is process-wide and unloaded
  before each load.
- amp (sidecar.rs:241): header 3 i32 [d, k, type], then A (k rows of d) and B
  (d rows of k). Apply is `y += x*B*A`, scaled by `--zchain-scale` (default 1.0,
  non-positive clamped to 1.0); ② and ③ concatenate by rank, ② rows first, max
  rank 8192 (zchain.rs:24-27).
- base.fnv: checked before anything loads; a mismatch between the sidecar and
  the base is a refusal, not a warning (zchain.rs:18-19).

Serving state of the port itself, which matters for any plan here: the reader
and merge are ported to Rust, but the amp apply is not. v41_run.rs:1714 refuses
a directory carrying `amp_Lnn.bin` by name ("the low-rank amplifier apply is not
ported"); only the C engine applies it. gr and rb with no amp file are the legal
plugin shape (zchain.rs:29).

## 3. What maps onto Spark, and what does not

Spark-X2.6-4B is dense: 36 layers, 2560 embedding, 290 tensors, no experts. That
decides each file:

| Sidecar part | Spark | Why |
|---|---|---|
| rb (router bias) | not applicable | no router and no experts; the file's header would be [0, 1], there is nothing to bias |
| gr (gains) | would apply with n_expert = 1 | an interpretation, not a tested path: the vector becomes a per-channel gain over d = 2560, a diagonal correction on the hidden stream |
| amp (low rank) | applies unchanged | `y += x*B*A` is a hidden-state correction with d = 2560 and any rank up to 8192; it is architecture-agnostic |
| base.fnv | applies unchanged | same fingerprint idea, need the Spark base hash |

So a Spark domain sidecar is a per-layer pair: one 2560-wide gain vector and one
low-rank A/B pair. That is small, on the order of a few MB per layer, and exactly
the shape of the ds4 artifact.

## 4. Why nothing can load it today

1. llama.cpp, which runs Spark, has no sidecar concept at all. There is no
   `--zchain`, no gain table, no forward-tail hook. It would need a patch: a
   reader for the three files, a per-layer gain multiply, and an add of `x*B*A`
   in the graph plus its CUDA kernel.
2. ds4-rs cannot run Spark: no spark2_5 family (searching crates for spark2_5
   returns nothing), and its V4.1 amp path is refused even for its own family.
3. No Spark sidecar exists. The five published directories were trained on
   DeepSeek-V4.1 with its post-training pipeline; nothing equivalent has been
   trained for Spark, and it needs a corpus plus the zfinetune/loss machinery.

## 5. Steps to build it

1. Define the Spark sidecar layout. Keep the ds4 byte format for gr and amp so
   existing tooling and reviewers can read it: 3 i32 headers, f32 bodies,
   n_expert = 1, d = 2560, rank per layer. Add a Spark base.fnv computed over
   models/spark26-4b-bf16.gguf.

2. Write the apply in llama.cpp first, before any training. Put the gain
   multiply after each layer's residual add and the `x*B*A` add at the same
   point the engine uses (forward tail), behind a flag such as `--sidecar DIR`.
   Acceptance test: a sidecar whose A is all zeros must be bit-identical to the
   no-sidecar run. That test needs no trained weights and proves the plumbing.

3. Produce a first sidecar by fitting, not by post-training. The cheapest useful
   experiment is a closed-form fit on held-out text: pick one layer, collect its
   hidden states, and solve a rank-16 B/A pair that reduces next-token NLL. This
   measures whether the mechanism has any headroom on Spark before investing in
   the full zfinetune pipeline.

4. Only if step 3 shows a real gain, port the post-training path: corpus, the
   four calibration losses, and per-domain fitting, with the same discipline ds4
   records (held-out split, base fingerprint, refusal on mismatch).

5. Gates before release, same shape as RECIPE.md section 7: held-out KL and
   next-token agreement on the no-sidecar base, per domain, plus a decode-speed
   check that the amp add does not cost more than a few percent.

## 6. Honest expectation

The ds4 sidecars were trained for a 113 GB MoE model with a post-training
pipeline and five domain corpora. Spark is a 4B dense model; a low-rank
correction of a few MB per layer can move behavior, but there is no published
evidence for Spark and none was measured here. Treat step 3 as the decision
point: if a rank-16 fit on one layer does not move held-out NLL, the full
pipeline will not pay for itself.
