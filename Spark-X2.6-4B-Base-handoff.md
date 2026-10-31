# Spark-X2.6-4B-Base — session handoff

Date: 2026-10-31. Repo: /data/Spark-X2.6-4B-Base. HEAD is the commit that
carries this handoff (docs(agents): how to use the kernel tuner), and
origin/master is behind it until the push is approved. The working tree is
clean once that commit exists.

Everything below was run in this repository; nothing is a plan dressed as a
result. Where a number is missing it says so.

## 1. Where things stand

- Q6_K GGUF: runs in llama.cpp at 128K, 3.46 GB, 107 tok/s without the draft,
  122-150 tok/s with DSpark.
- VQ artifacts are built in the DQV3 row-local payload:
  4x8192 x2 (3.375 GB, 6.5653 bpw, whole-model relL2 3.569%) is the pick;
  4x4096 x3 (4.655 GB, 9.0564 bpw, 2.174%) and 4x1024 x1 (1.289 GB,
  2.5084 bpw, 23.548%, a throughput probe and not a usable model) are built
  too. The DQVQ builds of the same points are kept for comparison.
- Decoders: C (bin/decode), Rust (bin/dqvq) and CUDA (bin/decode-gpu) read the
  container. C and Rust are bit-exact; CUDA matches a host f64 matvec.
- Decode rate for 4x8192 x2: about 10-11 ms/token, 91-100 tok/s, 308-337 GB/s
  of container, the DQV3 block kernel.
- Prefill is MEASURED now: 512 tokens in 293.8 ms (1742.8 tok/s) in the C
  driver and 269.9 ms (1896.7 tok/s) in the Rust harness, on the tiled f16
  kernel. The legacy kernel is 14.6x slower on the same bytes.
- VQ runtime: tools/vq-runtime is a CPU f32 reference that loads either the
  bf16 checkpoint or an RQVQ artifact and runs the full Spark2_5 forward pass
  over token ids. It is gated (numbers in section 2): on bf16 it reproduces
  llama.cpp's top-20 next tokens in the same order, with log-softmax values
  within 0.011, and every VQ point loads and runs with argmax agreement. It is
  reference speed, not a fast runtime: 2.7-2.9 tok/s with 16.1 GB resident.
- GPU VQ runtime: bin/vq-model-gpu keeps the containers on the device and runs
  the same forward pass there, every matrix through the fused DQV3 block GEMV.
  It is gated against the CPU reference (section 2): identical generated ids,
  logit relL2 0.000027% at 5 tokens and 0.000863% over 600 tokens (which
  crosses the sliding window), argmax agreement everywhere. Decode is 70.8-71.8
  tok/s at ctx 512 on 4x8192 x2, 24x the reference. The prompt path now uses
  the tiled f16 prefill in chunks of up to 512 tokens (section 2): 851-868
  tok/s at 128-512 tokens, 14x the token-by-token path, with the same
  generated ids and a relL2 of 0.0055% at 5 tokens and 0.15% at 600 against
  that path.
- Kernel tuner: bin/vq-tune sweeps the legal GEMV variants, launch widths and
  prefill tiles on one artifact's real payloads and writes
  <artifact>/vq-tune.profile; the runtime reads it when the artifact
  fingerprint and the device name match. On 4x8192 x2 the profile is
  bit-identical to the heuristic and moves a 500-token prompt from 892.3 to
  917.8 tok/s (section 2).

## 2. Measurements (do not re-derive)

Quality, whole checkpoint, 4,111,892,480 weights, weighted relL2. The DQV3
build of a point reproduces its DQVQ quality to the last thousandth, as it
must: the payload changes, the trained indices do not.

    Q5_K            3.06 GB   5.75 bpw   3.626%
    VQ 4x4096 x2    3.10 GB   6.04 bpw   4.244%
    VQ 4x8192 x2    3.38 GB   6.57 bpw   3.569%   (DQVQ build: 3.568%)
    Q6_K            3.46 GB   6.72 bpw   1.784%
    VQ 4x4096 x3    4.66 GB   9.06 bpw   2.174%   (DQVQ build: 2.175%)

Decode rate, batch 1, matrix work only. One run on 2026-10-31, all DQV3:

    4x1024 x1    1.289 GB    4.080 ms/token   316.0 GB/s   245.1 tok/s
    4x8192 x2    3.375 GB   10.844 ms/token   311.2 GB/s    92.2 tok/s
    4x4096 x3    4.655 GB   13.322 ms/token   349.4 GB/s    75.1 tok/s

The rate drifts between sessions on the same binary and artifact (10.18 to
10.95 ms for 4x8192 x2 across sessions, and a 2026-10-30 run of the DQVQ build
recorded 10.266 ms where it now measures 13.2-14.6 ms). Compare kernels and
artifacts only inside one run; the ratios are stable, the absolute number is
not.

Prefill, 4x8192 x2, one session on 2026-10-31, whole artifact:

    128 tokens, tiled f16, codebook in shared    107.2 ms   1193.8 tok/s
    128 tokens, tiled f16, codebook through L1   121.8 ms   1051.2 tok/s
    128 tokens, legacy per-(row, token)         1569.96 ms    81.5 tok/s
    512 tokens, tiled f16, codebook in shared    293.8 ms   1742.8 tok/s
    512 tokens, same, Rust harness               269.9 ms   1896.7 tok/s

    4x4096 x3, 512 tokens, tiled shared          424.2 ms   1207.0 tok/s
    4x1024 x1, 128 tokens, legacy                758.1 ms    168.8 tok/s

The 4x1024 x1 artifact cannot take the tiled path: its stream is 10-bit and
the round map is 12-bit, so vqg_prefill_ready refuses it and the legacy kernel
runs. That is the correct outcome, not a regression.

Runtime gate, 2026-10-31, prompt [1046, 11862, 314, 17349, 390] ("The capital
of France is", 5 tokens, ids from llama.cpp /tokenize with add_special false),
greedy:

    forward vs llama.cpp bf16   top-20 ids identical and in the same order,
                                log-softmax within 0.011 (19500: -0.8606
                                against -0.8592)
    4x8192 x2 vs bf16            argmax 19500 both, logit relL2 11.52%,
                                 cosine 0.99896, KL 0.42 nats, top-10 5/10
    4x4096 x3 vs bf16            argmax 19500 both, logit relL2 9.51%,
                                 cosine 0.99915, KL 0.38 nats, top-10 6/10
    load                        28.9 s bf16, 20.6 s 4x8192 x2, 16.1 GB RSS
    steady state, 32 tokens     2.72 tok/s bf16, 2.78 tok/s VQ

The distribution numbers are one prompt and one position. They prove the
loader and the forward pass are right and show the order of magnitude; they
are not a perplexity or KL benchmark over text.

GPU model runtime gate, 2026-10-31, same prompt, GPU logits against the CPU
reference's logits (both artifacts, bin/logits-diff):

    4x8192 x2, 5 tokens     argmax 19500 both, relL2 0.000027%, maxabs 1.9e-5,
                            cosine 1.0, top-100 100/100, KL 0 both directions
    4x4096 x3, 5 tokens     argmax 19500 both, relL2 0.000028%, maxabs 2.1e-5,
                            top-25 25/25
    4x8192 x2, 600 tokens   argmax 1046 both, relL2 0.000863%, maxabs 9.2e-4,
                            top-25 25/25
    generated ids, 8        19500, 27, 614, 11862, 314, 17349, 390, 19500
                            on both runtimes
    load                    0.7 s for 3.37 GB (page-cached); 7.42 GB free of
                            12.47 GB after load
    decode, ctx 512, gen 64 70.84 / 71.25 / 71.80 tok/s (three runs) against
                            2.97 tok/s for the CPU reference

The 600-token prompt matters: it is the only row where positions exceed the
512-token sliding window, so the start index, the mask and rope far from 0 are
exercised. The small drift (0.0009% against 0.00003%) is accumulation order,
three orders of magnitude below the quantization effect the gate exists to
see. 71 tok/s is about 74 percent of the artifact-only block-kernel rate
measured in the same session (10.321 ms/token, 326.96 GB/s, 96.9 tok/s
matrix-only; section 2's 10.84 ms is an earlier session and the rate drifts);
the remainder is 218 launches per token, attention, the norms and one device
sync per token.

Tiled-prefill gate, 2026-10-31, the model runtime's prompt path against its
own token-by-token path (--no-prefill) with bin/logits-diff, plus the CPU
anchor where noted:

    4x8192 x2, 5 tokens     argmax 19500 both, relL2 0.005455%, maxabs 4.9e-3,
                            top-100 100/100, KL 0 both; tiled against the CPU
                            reference is 0.005458%, the same number
    4x8192 x2, 600 tokens   argmax 1046 both, relL2 0.1501%, maxabs 0.169,
                            top-25 25/25, top-100 99/100, KL 1e-6
    4x4096 x3, 5 tokens     argmax 19500 both, relL2 0.009183%, maxabs 5.5e-3,
                            top-25 25/25
    generated ids, 8        19500, 27, 614, 11862, 314, 17349, 390, 19500
                            identical on both prompt paths and both artifacts
    chunking                600 tokens as 512+88 against one 600-token chunk:
                            bit-identical logits (relL2 0)
    prompt rate             128 tokens 851.5 tok/s (token path 66.5)
                            512 tokens 868.1 tok/s (61.8)
                            600 tokens 797.7-833.8 tok/s (62.3)
    4x1024 x1               its 10-bit stream cannot take the tiled kernel;
                            the prompt falls back to token by token and says so

The 0.0055-0.15% error is the f16 activation conversion inside the mma kernel,
not a forward-pass difference: the token path itself matches the CPU f32
reference to 0.000027% and 0.000863% respectively, and tiled against the CPU
at 5 tokens equals tiled against token. Prompt ingestion is about half the
artifact-only tiled rate (1743 tok/s at 512); batched attention, four
transposes per layer, the norms and 512 embedding row dequeues make up the
gap. Not measured: model-level quality (perplexity or full-text KL) on any
path.

Kernel on one matrix (layer 0 down_proj, dim 4 / 8192 / 2), one run:

    DQV3 block kernel     0.0717 ms   299 GB/s
    dim-4 per-index       0.0881 ms   243 GB/s
    generic warp          0.2140 ms   100 GB/s
    plain f32 GEMV        0.2231 ms   470 GB/s of f32 weights

Gates, on all five payload shapes (dim 4 with 1024, 4096 and 8192 codewords,
dim 8 with 8192, and the DQVQ build of 8192):

    C decode vs checkpoint      matches the recorded quality of each point
    Rust vs C dump              26,214,400 / 26,214,400 exact, all five
    CUDA fused vs host f64      maxabs 2e-6, all five
    CUDA prefill vs host f64    relL2 0.0159%, maxabs 1.3e-3, tokens
                                0/1/63/64/127/128/511, all five

Kernel tuning, 2026-10-31, the whole 4x8192 x2 DQV3 artifact (bin/vq-tune,
512-token tiles, 7 timed launches per candidate; 6 distinct geometries, not
216 matrices):

    131072x2560 embedding   1 matrix    decode 378.0 GB/s   prefill v1 256x64   22,083 tok/s
    2560x10240 down        36           decode 328.1 GB/s   prefill v2  64x64  321,246 tok/s
    10240x2560 gate/up     72           decode 389.2 GB/s   prefill v2 128x64  354,108 tok/s
    16x2560 g_proj         36           decode  16.1 GB/s   prefill v1  64x64  4,273,504 tok/s
    2560x4096 out          36           decode 274.8 GB/s   prefill v2  64x64  788,644 tok/s
    6144x2560 qkv          36           decode 337.7 GB/s   prefill v2  64x64  545,275 tok/s

Profile against the heuristic on the same binary, 500-token prompt, ctx 512:
917.8 tok/s against 892.3 (+2.9%), and the logits are bit-identical
(relL2 0.000000%, maxabs 0, top-100 100/100, KL 0 both directions): a tile
changes which tokens share a block, never the per-token K order. Decode in
that pair of runs was 64.3 against 62.4 tok/s, inside run-to-run noise. The
sweep itself takes about 4 s per artifact. A DQVQ artifact sweeps too, but
records legacy prefill for every geometry (no tiled path), so its prompt stays
token by token.

## 3. Decisions and why

- DQV3 is the payload to build. Same container size, same quality, and it is
  the only payload the block kernel and the tiled prefill can use: 20-25
  percent of decode rate for free when the encoder writes it.
- nc 8192 is both the quality pick and the format ceiling. DQV3 carries 13
  index bits (12 main + one plane), so wider codebooks are impossible, and the
  measured gains come from stages, not codewords: 4096 to 8192 bought 0.675
  points for 0.53 bpw, while a third stage bought 2.07 points for 3.02 bpw.
- The index width is enforced, and it has two parts. First, 13 bits: builder
  and all three decoders refuse nc above 8192. Second, the main stream is
  min(bit width, 12) bits, so a payload below nc 4096 has narrower slots and
  shorter rows; all three decoders read a fixed 12 until 2026-10-31 and
  decoded those payloads at the wrong offsets (the CUDA shared-codebook GEMV
  ran off the codebook and faulted). The width now comes from the header, and
  the 12-bit-only kernels refuse narrower streams.
- The codebook stays f16 in DQV3. ds4's E4M3 would round our learned codewords
  by about 3%, which is a quality change, not a layout change. The tiled
  prefill keeps f16 products for the same reason: bf16 would round a
  10-mantissa-bit codeword by 2^-8. The named cost is the activation rounding
  f32 -> f16, which the host f64 gate prices at relL2 0.0159%.
- The Rust tools are a cargo workspace (tools/Cargo.toml). build.sh fails when
  cargo is missing; there is no "skipped rustc" branch any more.
- The block kernel is not further tuned here. ds4's version is pinned at the
  64-register wall and its two escalation attempts measured negative there.

## 4. Open work, in priority order

1. Prefill micro-tuning in the GPU model runtime. The tiled prefill is wired
   and gated (section 2), and the tile shapes are now measured per geometry
   (851-918 tok/s at 500-512 tokens against 1743-1909 tok/s for artifact-only
   matrix work), so roughly half the prompt time is the batched attention,
   four transposes per layer, the norms and 512 embedding row dequeues. Candidate work, in the order the numbers suggest: batch the
   embedding row dequeue into one launch; fuse the transpose into the norm and
   gelu kernels; and revisit the B-tile load's 4-way bank conflict (it reads
   xT column-major and writes shared token-major) plus a 256-token B tile if
   shared memory allows. Acceptance: raise the 512-token prompt rate without
   changing the logits beyond the f16 gate in section 2 (argmax and top-25
   must hold, and the new relL2 is reported).
2. DSpark in the VQ runtime. DEFERRED on purpose (2026-10-31), not forgotten.
   DSpark is this model's MTP equivalent and it is already wired on the
   llama.cpp path (measured 1.13-1.40x there), but it is not wired into the VQ
   runtime. The prompt path is now fast enough that this is a priority
   decision rather than a dependency. When it is taken up, the work is: expose
   target hidden states at five layers, load the draft GGUF, implement the
   5-block SWA dflash graph with attn_gate and gelu, then the propose/verify
   loop. Acceptance against a VQ target is unmeasured and must be measured,
   because the draft was trained on bf16 hidden states and VQ shifts them
   (manifest relL2 about 2.8-3.4 per tensor).

Why "add MTP" is not an option, checked 2026-10-31: the released checkpoint has
no MTP layers. model.safetensors.index.json has no mtp or nextn tensors,
models/hf/config.json has no next-token-prediction field, and
models/hf/modeling_spark.py has no MTP path. The DSpark draft is the vendor's
trained multi-token module, shipped separately: its GGUF has no
token_embd.weight and no output.weight, only output_norm.weight, so it borrows
the target's embedding and LM head and consumes hidden states at
dflash.target_layers (the card lists [2, 10, 18, 25, 33]). That is the
EAGLE-style pattern, a separate 624 M draft fed by target features, not an
in-checkpoint MTP head. Anything called "fixing MTP" would mean training a
new module, not repairing one.

3. A 4x16384 point would need the DQVQ payload and is unmeasured; it would
   also lose both fast kernels, so expect the per-index rates.
4. Tuner extensions, none started. A BK 128 tile needs a 96-bit A-tile window
   first (three words, the shape v3_rounds uses); the tile table would then
   carry it. Per-layer profiles are possible but pointless while matrices
   share a geometry: the sweep already shows the winner depends on the shape,
   not the layer. A load-time autotune inside the runtime is the alternative to
   the profile file, and it would pay the sweep on every cold start.

## 5. Resume

    ./tools/vq/build.sh                       # C tools + cargo workspace
    ./bin/decode info <file.bin>              # header, format, width
    ./bin/decode check <file.bin> models/hf <tensor>
    ./bin/decode dump <file.bin> <out.f32> && ./bin/dqvq check <file.bin> <out.f32>
    ./bin/decode-gpu --manifest models/vq-build-8192-2-v3 --iters 10
    ./bin/decode-gpu <file.bin> --check
    ./bin/decode-gpu <file.bin> --prefill 128 --check [--prefill-variant 0..3]
    ./bin/vq-bench models/vq-build-8192-2-v3 --iters 10 [--prefill 512] [--csv]
    ./bin/vq-bench compare a.csv b.csv
    ./bin/vq-runtime --weights hf --dir models/hf --ids 1046,11862,314,17349,390 --gen 8 --logits /tmp/hf.f32
    ./bin/vq-runtime --weights vq --dir models/vq-build-8192-2-v3 --ids 1046,11862,314,17349,390 --gen 8 --logits /tmp/vq.f32
    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 --ids 1046,11862,314,17349,390 --gen 64 --ctx 512 --logits /tmp/gpu.f32
    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 --ids 1046,11862,314,17349,390 --gen 8 --ctx 512 --no-prefill --logits /tmp/gpu-tok.f32
    ./bin/vq-tune models/vq-build-8192-2-v3 --tokens 512 --iters 7 [--dry-run] [--force]   # writes vq-tune.profile
    ./bin/vq-tune models/vq-build-8192-2 --tokens 512 --iters 5                 # DQVQ: legacy prefill only
    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 --ids 1046 --ctx 64 --profile other.profile   # refused (foreign artifact)
    ./bin/vq-model-gpu --dir models/vq-build-8192-2-v3 --ids <512 ids> --gen 0 --ctx 1024 [--chunk 512]   # prompt rate
    ./bin/logits-diff /tmp/gpu-tok.f32 /tmp/gpu.f32 --top 25   # prompt paths against each other
    ./bin/logits-diff /tmp/hf.f32 /tmp/vq.f32 --top 20
    ./bin/logits-diff /tmp/vq.f32 /tmp/gpu.f32 --top 20   # GPU against the CPU reference
    ./bin/build-vq-gpu models/hf <out> 4 8192 2   # DQV3 by default, ~4 min

## 6. Traps

- /tmp is a 16 GB tmpfs here. A VQ build written there dies mid-file, leaves a
  truncated manifest and a plausible-looking aggregate. Build into out/ or
  /data; build-vq-gpu now checks every malloc and write and removes the partial
  file, but the tmpfs is still the wrong place.
- Stage payloads are copied to an 8-byte-aligned device offset before the
  kernels see them; the file packs them contiguously and the f16 codebook is
  read as uint2. The DQV3 header is 24 bytes on purpose (4 pad after flags);
  20 put the codebook at 4 mod 8 and faulted the uint2 load.
- The block kernel and the tiled prefill are 12-bit by construction. They need
  nidx_row % 32 == 0 (block kernel), a 4-byte-aligned row stream, DQV3 (not
  DQVQ), dim 4 or 8 (prefill), and nc >= 4096. vqg_gemv_ready and
  vqg_prefill_ready are the only places that decide it; do not call a variant
  directly on an unchecked geometry.
- A prefill gate must zero y and take one clean pass before comparing: the
  timing loop accumulates into y, so comparing straight after it compares
  iters results against one reference. The first version of the new gate did
  exactly that and reported 400 percent error.
- rustc defaults to edition 2015; the crates are edition 2021 in
  tools/Cargo.toml, and Cargo.toml sets lto and codegen-units for release.
- ds4-core's Rust f16 decoder (crates/ds4-core/src/vq.rs, the e == 0 branch at
  112) decodes subnormals one binade low; vq_fmt.h:22-28 and dqvq use 113.
- Two llama-server instances do not fit on this card; one at 128K q8_0 plus the
  draft is about 8.8 GB of 12.3 GB.
- vq-runtime takes token ids, never text, so the logits gate only means
  anything if both implementations see the same ids: tokenize with llama.cpp's
  own /tokenize (add_special false) and pass that list, or the prompts differ
  by a BOS and the comparison is void. The server reports tokens_evaluated;
  check it equals the id count.
- A run that has just loaded 7.7 GB of weights reports a slow prompt rate: the
  same 5-token prompt measured 1.57 tok/s cold and 2.78 tok/s warm on
  2026-10-31. Quote the generated rate of a 32-token run as the steady-state
  number, not the first prompt rate of the first run.
- The Q6_K Base checkpoint loops under greedy chat decoding; that is the BASE
  model, not damage. Send temperature 0.8-1.0 and always send max_tokens.
- vq-model-gpu's prompt path uses the tiled prefill by default and runs the
  prompt in chunks of up to 512 tokens (--chunk). Chunk boundaries are
  bit-identical: 600 tokens as 512+88 and as one 600-token chunk gave relL2 0,
  so a rate difference between chunk sizes is scheduling, not arithmetic.
  --no-prefill runs the token-by-token path and is the gate's reference.
- A tuning profile is bound to the artifact fingerprint and the device name.
  A profile from another artifact, another GPU, or a changed container is
  refused with a printed reason and the heuristic runs; --no-tune skips the
  file entirely. Never copy a profile between artifacts to save a sweep.
- Stage buffers need 8 bytes of slack past the payload. The kernels read
  aligned 4- and 8-byte index windows, so the last index of the last row
  reaches up to 7 bytes past the stream; the C driver, the Rust harness, the
  model runtime and the tuner all allocate per + 8 and zero it. compute-
  sanitizer flags the read as out of bounds without it (2026-10-31), even
  though cudaMalloc's granularity had hidden it for the whole project.
- BK 128 is not in the tile table on purpose: the A-tile extracts four code
  words from one 64-bit window, and eight would need 96 bits. It produced
  relL2 0.669 when it was offered, which is what the gate exists to catch.
- The prefill gate's relL2 (0.0055% at 5 tokens, 0.15% at 600) is larger than
  the token path's against the CPU (0.000027%, 0.000863%) because the mma
  kernel rounds activations to f16. Do not read that as a regression, and do
  not expect top-100 identity at long contexts (600 tokens: top-25 25/25,
  top-100 99/100, argmax and KL unchanged at 1e-6).
- One layer matrix whose geometry the tiled kernel refuses (a 10-bit stream,
  as in 4x1024 x1, or cols not a multiple of 64) disables the whole prompt
  prefill by design: the runtime says so and runs token by token. The legacy
  batched kernel is not mixed in, because for such a matrix it is slower than
  decode.
- The embedding row dequeue is still one tiny launch per token (512 per
  chunk), part of the prompt-rate gap against the artifact-only 1743 tok/s.
- vq-model-gpu allocates the KV cache from --ctx at startup, and its default is
  4096 (about 1.2 GB of device memory). A prompt longer than --ctx exits with
  "position N exceeds the M-token cache", which is the intended error, not a
  crash. The gate runs used 512 and 1024.
- The GPU runtime picks each tensor's GEMV variant at load with
  vqg_gemv_ready (block kernel, else dim-4 warp, else generic warp). A payload
  that cannot take the block kernel therefore still runs, just slower; that is
  deliberate, not a silent specialization to one artifact point.
