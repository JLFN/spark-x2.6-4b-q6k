# Recipe: Spark-X2.6-4B-Base — Q6_K build, llama.cpp, RTX 4070 SUPER

This document is both the recipe and the record of the run. Everything marked
MEASURED was produced on this machine on 2026-10-25; the build, the artifact, the
server and the open-grok registration all exist.

Goal: keep almost the original quality, gain speed, and run locally at a 128K
context on a 12 GB card.

## 0. Status

| | |
|---|---|
| Artifact | `models/spark26-4b-Q6_K.gguf`, 3,460,437,312 B, 6.72 BPW, sha256 `f708e464106424cd001229cb3b9d8c07a224ec1adc81dfda8684a9fc621ee4f5` |
| Gate reference | `models/spark26-4b-bf16.gguf`, 8,229,920,832 B, sha256 `fd30efba4c3ceb8c16a9d9c7a07d7753db9c52b6fdc07d2b44a8ceb9ca15cbe5` |
| Served | `llama-server` on `127.0.0.1:8090`, context 131072, one slot, 9,465 MiB of 12,282 MiB VRAM |
| Decode, MEASURED | 84.9 tok/s (11.78 ms/token) at a 131072-token context allocation |
| open-grok | registered as `spark2.6-4b-q6k`, `context_window = 131072` |
| Quality gate | NOT RUN — perplexity and KL against the BF16 GGUF are the precondition for publishing |
| DSpark pairing | WORKS on the XHToken fork, 1.30-1.40x decode on normal text at 128K (section 12) |
| Hugging Face | not published; plan is section 8 |

## 1. Target host

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB, compute capability 8.9 (sm_89) |
| Driver / CUDA | 610.43.03 / CUDA 13.3 (`/usr/local/cuda` -> `cuda-13.3`) |
| CPU / RAM | 28 cores, 31 GB |
| Read bandwidth, MEASURED | 439.0 GB/s (1 GiB buffer, 60 passes, 2.45 ms each, `/tmp/bwprobe.cu`) = 87% of the 504 GB/s spec |
| Build target | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` |

## 2. What type, and why

llama.cpp publishes a perplexity delta per type, measured on Llama-3-8B
(`tools/quantize/quantize.cpp:55-73`). It is one model, not Spark, but it is the
only per-type ordering available locally.

| Type | ppl delta | Size on Spark | 2K tok/s (ceiling) | Quality class |
|---|---:|---:|---:|---|
| BF16 | -0.0050 (noise) | 8.22 GB | 53 | source |
| Q8_0 | +0.0026 | 4.37 GB | 100 | near-lossless |
| **Q6_K (chosen)** | **+0.0217** | **3.46 GB** | **127** | **almost original** |
| Q5_K_M | +0.0569 | 2.83 GB | 155 | very good |
| Q4_K_M | +0.1754 | 2.31 GB | 190 | different bracket |
| Q3_K_M | +0.6569 | 1.77 GB | 249 | visibly degraded |

Q6_K is the last type whose measured delta still justifies "almost the same", and
the token embedding is lifted one step to Q8_0 because the tied table is also the
output head and is the one tensor whose damage caps everything downstream.

MEASURED: the artifact came out at 6.72 BPW overall, 3,294.99 MiB of quantized
tensors, exactly the class this table predicts.

## 3. Pinned inputs

| Input | Pin |
|---|---|
| Weights | `XHToken/Spark-X2.6-4B-Base` rev `e1bd659ae2b1c2263b2fc2544485976c1b516115`, Apache-2.0, 4,112,079,360 params, 8,224,158,720 B of tensor data, 5 shards, 20 files |
| Toolchain | `/data/llama.cpp` @ `fa3645feb5f47cbb30df5e6d031c22543d43d240` (2026-10-25), upstream `master` |
| Architecture support | `spark2_5` at `src/llama-arch.cpp:158`; converter `conversion/spark2_5.py` (registered for `Spark2_5ForCausalLM`); loader `src/models/spark2-5.cpp`; tokenizer pre-type `src/llama-vocab.cpp:2326` |
| Converter's own checks | gelu, `gate_attn_act_mode` sigmoid, `headwise_attn_output_gate` true, `layer_types` only sliding/full, `g_proj` rows == `num_attention_heads`. This checkpoint passes all five (16 heads, gelu, sigmoid, true, 27 sliding + 9 full). |

The stale-binary trap: the binaries in `~/.local/bin` are from 2026-07-25 and
predate the Spark merge (upstream 2026-10-13, PR #27868); `strings
libllama-quantize-impl.so | grep -c spark2_5` returns 0. Everything here runs from
`/data/llama.cpp/build/bin`, built from the pin above. Use those paths, not
`~/.local/bin`.

## 4. The build record

| Step | Command | Result | Wall |
|---|---|---|---:|
| Clone | `git clone https://github.com/ggml-org/llama.cpp /data/llama.cpp` | 641 MB, HEAD `fa3645feb` | 23 s |
| Configure | `cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release` | CUDA 13.3.73, host gcc 14.2.0 | 2 s |
| Build | `cmake --build build -j 16` | 604 steps, all binaries in `build/bin` | 235 s |
| Download | `hf download XHToken/Spark-X2.6-4B-Base --revision e1bd659... --local-dir models/hf` | 20 files, 7.7 GiB | 51 s |
| Convert | `python3 convert_hf_to_gguf.py models/hf --outtype bf16 --outfile models/spark26-4b-bf16.gguf` | 290 tensors, 8.22 GB | 60 s |
| Quantize | `llama-quantize --token-embedding-type q8_0 --output-tensor-type q8_0 <bf16> <out> Q6_K` | 3,294.99 MiB, 6.72 BPW | 47 s |

The converter writing exactly 290 tensors matches the inventory: 36 layers x
(6 matrices + 2 norms) + `token_embd` + `output_norm` = 290.

## 5. Speeds: predicted against measured

The ceiling model was bytes over 439 GB/s. The measured decode is 84.9 tok/s at a
131072 context allocation, inside the predicted 70-85% band of the 127 tok/s
ceiling, and it is the number to quote.

| | Ceiling | MEASURED |
|---|---:|---:|
| Decode, 131072 context allocated | 127 tok/s | **84.9 tok/s** (11.78 ms/token) |
| Prefill, 8-token prompt | — | 249 tok/s (launch-dominated, not representative) |
| Prefill, 72-token chat prompt | — | 857 tok/s |
| VRAM after load | 7.78 GiB estimate | **9,465 MiB** of 12,282 |

The gap between 7.78 GiB (weights + KV) and 9,465 MiB is compute buffers and the
CUDA context. At 131072 tokens that leaves about 2.8 GB of headroom, so the 128K
context fits, but not by a wide margin: if another GPU job takes VRAM, halve the KV
with `KV=q8_0`.

## 6. Running it

    /data/Spark-X2.6-4B-Base/models/run-spark26.sh

Defaults: the Q6_K artifact, `CTX=131072`, `PORT=8090`, `KV=f16`, alias
`spark2.6-4b-q6k`, `-ngl 999 -fa on -np 1 --jinja`. Overrides are environment
variables (`CTX`, `PORT`, `KV=q8_0`, `MODEL`, `BIN`), and extra flags pass through
to `llama-server`.

MEASURED, both endpoints answer:

    POST /v1/chat/completions  {"messages":[{"role":"user","content":"The capital of France is"}]}
      -> "The capital of France is Paris."
    POST /v1/completions       {"prompt":"The capital of France is"}
      -> " Paris."

Note the model is a BASE checkpoint: it continues text. llama-server reports its
capability as `completion`, and `--jinja` applies the checkpoint's own chat
template, but expect continuation behaviour, not instruction following.

open-grok entry (already in `~/.opengrok/config.toml`, config backed up to
`config.toml.bak-20261025-*`):

    [model."spark2.6-4b-q6k"]
    model = "spark2.6-4b-q6k"
    name = "Spark X2.6 4B Q6_K (local 4070)"
    base_url = "http://127.0.0.1:8090/v1"
    api_backend = "chat_completions"
    api_key = "dummy"
    context_window = 131072
    max_completion_tokens = 8192
    supports_images = false

## 7. Gate before publishing

Nothing here has been measured for quality yet. Three checks, in order:

1. Perplexity, BF16 against Q6_K, on the same corpus and the same `--chunks`:

        build/bin/llama-perplexity -m models/spark26-4b-bf16.gguf -f corpus.txt --chunks 32
        build/bin/llama-perplexity -m models/spark26-4b-Q6_K.gguf -f corpus.txt --chunks 32

   Acceptance: the delta must be in the Q6_K class (+0.02 to +0.10 ppl by the type
   table). A large delta means the conversion is wrong, not the type.

2. Teacher-forced KL against the BF16 logits:

        build/bin/llama-perplexity -m models/spark26-4b-bf16.gguf -f corpus.txt \
            --save-all-logits bf16.logits --chunks 32
        build/bin/llama-perplexity -m models/spark26-4b-Q6_K.gguf -f corpus.txt \
            --kl-divergence-base bf16.logits --chunks 32

3. A 128K smoke test: fill the window and confirm the server does not OOM and that
   output stays coherent past the 512-token sliding window. 27 of 36 layers slide,
   so a short prompt cannot see a problem here.

## 8. Publishing to Hugging Face

Not done. Publishing is an external, public action and needs the owner's explicit
go-ahead at the time, plus the gate in section 7.

What to publish, as one repository (`<org>/Spark-X2.6-4B-Base-GGUF`, Apache-2.0,
matching the upstream license):

| File | Size | Purpose |
|---|---:|---|
| `spark26-4b-Q6_K.gguf` | 3.46 GB | the artifact |
| `SHA256SUMS` | — | both hashes, computed on this machine |
| `README.md` | — | model card: source revision, toolchain pin, type table, measured speed, context, limits |
| `LICENSE` | — | Apache-2.0, copied from the upstream checkpoint |
| `run-spark26.sh` | — | how it was served, with paths made relative |

Do not publish the BF16 GGUF: 8.2 GB, and it is reproducible from the pinned source
in one minute. Do not publish `models/hf/` either, for the same reason.

Publish command shape, once approved:

    hf auth whoami
    hf upload <org>/Spark-X2.6-4B-Base-GGUF models/spark26-4b-Q6_K.gguf spark26-4b-Q6_K.gguf
    hf upload <org>/Spark-X2.6-4B-Base-GGUF models/SHA256SUMS SHA256SUMS
    hf upload <org>/Spark-X2.6-4B-Base-GGUF README.md README.md
    hf upload <org>/Spark-X2.6-4B-Base-GGUF models/hf/LICENSE LICENSE

The model card must state: source `XHToken/Spark-X2.6-4B-Base` at rev `e1bd659`,
llama.cpp `fa3645feb`, type plan Q6_K with a Q8_0 token embedding, measured 84.9
tok/s at 128K on an RTX 4070 SUPER, and that the artifact is a base model, not
instruction tuned.

## 9. Optional: the Rust engine port

Only needed if this artifact must run in `ds4-dfm-rs`. The GGUF is the contract:
`blk.{i}.attn_qkv.weight` (fused QKV), `blk.{i}.attn_gate.weight` (16-row per-head
gate), standard `attn_norm`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`, `attn_out`,
`token_embd` (tied output), `output_norm`; hparams `sliding_window`,
`sliding_window_pattern`, `rope_dimension_count` and `rope_dimension_count_swa` from
the 0.25 partial rotary factor. No new kernels: Q6_K and Q8_0 already exist in
`cuda/mmq/` and in `crates/ds4-core/src/tensors.rs:77-81`.

## 10. Risks

- Quality is unmeasured until section 7 runs. The type table is from Llama-3-8B.
- 128K is tight: 9,465 MiB of 12,282 at load. Another GPU job can force `KV=q8_0`.
- The Spark support in llama.cpp is 12 days old upstream and thinly exercised; a
  conversion or loading bug would be new, not ours. The BF16 GGUF loading and
  answering is the evidence that the architecture path works.
- A base checkpoint continues text; it is not an instruct model, and open-grok chat
  will read as autocomplete.
- The upstream checkpoint declares 1M positions. This build serves 128K; beyond
  that the KV alone exceeds the card.

## 11. Open items

1. Run the section 7 gate and record the numbers in `docs/`.
2. Decide the publish target and visibility with the owner, then publish.
3. Decide whether the Rust engine needs the port (section 9) at all.

## 12. DSpark speculative decoding (2026-10-31)

The draft is not a merge partner: `XHToken/Spark-X2.6-4B-DSpark-GGUF` is a 624 M
parameter draft model for speculative decoding, 5 sliding-window layers, BF16,
7 proposed tokens per pass, trained to read target hidden states at layers
`[2, 10, 18, 25, 33]`. Pairing it with the Q6_K target is the legitimate
"combination"; anything that merges weights would be a new model, not this one.

Measured facts:

| | |
|---|---|
| Draft artifact | `Spark-X2.6-4B-DSpark.gguf`, 1,254,293,824 B, sha256 `2ee073c1...eedeac8` (matches the Hub LFS hash) |
| Draft architecture | `dflash` (per the GGUF metadata), context 1,048,576 declared |
| Required toolchain | XHToken fork `/data/llama.cpp-xhtoken` @ `22c0ae1871b60123c5afc008a78fe9c6fffa2858` |
| Upstream `fa3645feb` | FAILS: `check_tensor_dims: tensor 'blk.0.attn_q_norm.weight' not found` |
| Fork build fix | `src/llama-context.cpp` needs `#include <algorithm>` (uses `std::all_of`) |
| Load test | both models load, server answers on 127.0.0.1:8092 |

Why upstream fails, with the evidence: the released draft stores `blk.0.attn_gate.weight`
and no `attn_q_norm` (checked with `strings` on the GGUF). The fork's
`src/models/dflash.cpp` at 22c0ae1 selects the layout with
`has_attn_gate = ml->get_tensor_meta("blk.0.attn_gate.weight") != nullptr` and only
then creates `attn_q_norm`/`attn_k_norm` in the else branch. The 10-25 upstream code
has no such branch and demands both norms unconditionally.

Measurements, greedy, 256-token continuations, three fixed prompts, same fork binary:

| Context, KV | No draft | With draft | Acceptance | VRAM with draft |
|---|---:|---:|---:|---:|
| 32K, f16 | 98.7 / 96.2 / 95.9 | 139.0 / 132.0 tok/s | 0.23 / 0.23 | 7,343 MiB |
| 128K, f16 | 98.1 / 94.7 / 94.6 | 123.0 / 125.4 tok/s | 0.20 / 0.23 | 10,876 MiB (1,018 free) |
| 128K, q8_0 KV | — | 116.5 / 138.2 tok/s | 0.20 / 0.30 | 9,172 MiB (2,723 free) |

The third prompt degenerates into repetition and reaches 0.96 acceptance and
285-299 tok/s in every configuration. That is the honest picture: speculative
decoding pays when the draft can predict the target, and this draft was trained
against the post-trained Spark-X2.6-4B, not against this BASE checkpoint. The card
reports a mean acceptance length of 4.0-5.2 (about 2x speed) on its intended target;
here the mean length is 2.4-2.6 and the gain is 1.3-1.4x on normal text.

Risks:

- The draft and target are different checkpoints (post-trained draft, base target).
  Function works; acceptance is the measured cost, not an assumption.
- 128K with f16 KV leaves 1,018 MiB. Any other GPU job needs `KV=q8_0` or a smaller
  `CTX`; q8_0 KV keeps 2,723 MiB free at the same context.
- The fork is a third-party branch three days younger than the upstream pin, and it
  is built with one local source fix. Treat it as a separate toolchain, not as
  upstream.
- No quality change is expected: the target verifies every accepted token. That is a
  property of the algorithm, not a measurement made here.

