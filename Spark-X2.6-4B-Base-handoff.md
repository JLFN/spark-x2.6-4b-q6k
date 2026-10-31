# Spark-X2.6-4B-Base — session handoff

Date: 2026-10-31. Repo: /data/Spark-X2.6-4B-Base, HEAD is this handoff commit,
working tree clean. Remote https://github.com/JLFN/spark-x2.6-4b-q6k is at
f2978e1, so seven local commits are unpushed: d22cce1, 3c2fef3, d967368,
1dcaba1, 652198d, d64c289, and this one.

Everything below was run in this repository; nothing is a plan dressed as a
result. Where a number is missing it says so.

## 1. Where things stand

- Q6_K GGUF: runs in llama.cpp at 128K, 3.46 GB, 107 tok/s without the draft,
  122-150 tok/s with DSpark.
- VQ artifacts are now built in the DQV3 row-local payload:
  4x8192 x2 (3.375 GB, 6.5653 bpw, whole-model relL2 3.569%) is the pick;
  4x4096 x3 (4.655 GB, 9.0564 bpw, 2.174%) and 4x1024 x1 (1.289 GB,
  2.5084 bpw, 23.548%, a throughput probe and not a usable model) are built
  too. The DQVQ builds of the same points are still on disk for comparison.
- Decoders: C (bin/decode), Rust (bin/dqvq) and CUDA (bin/decode-gpu) read the
  container. C and Rust are bit-exact; CUDA matches a host f64 matvec.
- Decode rate for 4x8192 x2: 10.18-10.84 ms/token, 92-98 tok/s, 311-331 GB/s of
  container. The DQV3 block kernel is what got there; the same bytes in DQVQ
  measure 13.2-14.6 ms in the same sessions.
- Prefill: NOT measured. Both kernel shapes tried were wrong for a dense model
  (see 3) and their numbers must not be quoted.
- VQ runtime: still does not exist. Nothing but these tools decodes the format.

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

Same run, the DQVQ build of 4x8192 x2: 14.629 ms, 230.7 GB/s, 68.4 tok/s.
Repeats of the v3 4x8192 x2 point that session: 10.18, 10.29, 10.80, 10.84 ms.

The rate drifts between sessions on the same binary and artifact: a 2026-10-30
run recorded 10.266 ms for the DQVQ build, which measures 13.2-14.6 ms now.
Compare kernels and artifacts only inside one run; the ratios are stable, the
absolute number is not.

Kernel on one matrix (layer 0 down_proj, dim 4 / 8192 / 2), same run:

    DQV3 block kernel     0.0717 ms   299 GB/s
    dim-4 per-index       0.0881 ms   243 GB/s
    generic warp          0.2140 ms   100 GB/s
    plain f32 GEMV        0.2231 ms   470 GB/s of f32 weights

Gates, on all three v3 shapes built (dim 4 with 4096 and 8192 codewords, dim 8
with 8192):

    C decode vs checkpoint      matches the DQVQ build of the same point
    Rust vs C dump              26,214,400 / 26,214,400 exact, all three
    CUDA fused vs host f64      maxabs 2e-6, all three

## 3. Decisions and why

- DQV3 is the payload to build. Same container size, same quality, and it is
  the only payload the block kernel can use: 20-25 percent of decode rate for
  free when the encoder writes it.
- nc 8192 is both the quality pick and the format ceiling. DQV3 carries 13
  index bits (12 main + one plane), so wider codebooks are impossible, and the
  measured gains come from stages, not codewords: 4096 to 8192 bought 0.675
  points for 0.53 bpw, while a third stage bought 2.07 points for 3.02 bpw.
  A 4x16384 point would need the DQVQ payload and is unmeasured.
- The 13-bit width is enforced, not documented. Without the guard, nc 16384
  packed the low 13 bits of each index and every decoder read them back: no
  error, different weights. Builder and all three decoders now refuse it.
- The codebook stays f16 in DQV3. ds4's E4M3 would round our learned codewords
  by about 3%, which is a quality change, not a layout change.
- The Rust tools are a cargo workspace (tools/Cargo.toml). build.sh fails when
  cargo is missing; there is no "skipped rustc" branch any more.
- The prefill kernel was tried twice and abandoned, not tuned: the first
  uncoalesced the activations by 8 KB (84 s for 512 tokens, 6 tok/s), the second
  gave each thread one token and no reuse (494 ms for one token). A dense model
  needs a tiled GEMM; ds4's fused prefill is MoE-shaped and does not transfer.
- The block kernel is not further tuned here. ds4's version is pinned at the
  64-register wall and its two escalation attempts measured negative there.

## 4. Open work, in priority order

1. Tiled prefill. Reference: cuda/ds41_vq_prefill_mma.cuh:1-40 in
   /data/ds4-dfm-rs, 128 rows x 128 tokens x BK 64, bf16 products with f32
   accumulate, codebook decoded into shared. Gate: host matvec per token block,
   then prefill_tps against the current unusable numbers.
2. `vq-bench` prefill numbers are not usable until 1; the harness itself (ds4
   CSV columns, compare) is fine.
3. A VQ runtime. Nothing serves these containers yet; a decode-rate number is
   not a model.
4. Push: seven commits local, origin at f2978e1.

## 5. Resume

    ./tools/vq/build.sh                       # C tools + cargo workspace
    ./bin/decode info <file.bin>              # header, format, consistency
    ./bin/decode check <file.bin> models/hf <tensor>
    ./bin/decode dump <file.bin> <out.f32> && ./bin/dqvq check <file.bin> <out.f32>
    ./bin/decode-gpu --manifest models/vq-build-8192-2-v3 --iters 10
    ./bin/decode-gpu <file.bin> --check
    ./bin/vq-bench models/vq-build-8192-2-v3 --iters 10 [--prefill N] [--csv]
    ./bin/vq-bench compare a.csv b.csv
    ./bin/build-vq-gpu models/hf <out> 4 8192 2   # DQV3 by default, ~4 min

## 6. Traps

- /tmp is a 16 GB tmpfs here. A VQ build written there dies mid-file, leaves a
  truncated manifest and a plausible-looking aggregate. Build into out/ or
  /data; build-vq-gpu now checks every malloc and write and removes the partial
  file, but the tmpfs is still the wrong place.
- Stage payloads are copied to an 8-byte-aligned device offset before the
  kernels see them; the file packs them contiguously and the f16 codebook is
  read as uint2/half2.
- DQV3 header is 24 bytes on purpose (4 pad after flags). 20 put the codebook at
  4 mod 8 and faulted the uint2 load.
- The block kernel needs nidx_row % 32 == 0 and a 4-byte-aligned row stream;
  vqg_gemv_ready() is the only place that decides it. Do not call variant 3
  directly on an unchecked geometry.
- rustc defaults to edition 2015; the crates are edition 2021 in
  tools/Cargo.toml, and Cargo.toml sets lto and codegen-units for release.
- ds4-core's Rust f16 decoder (crates/ds4-core/src/vq.rs, the e == 0 branch at
  112) decodes subnormals one binade low; vq_fmt.h:22-28 and dqvq use 113.
- Two llama-server instances do not fit on this card; one at 128K q8_0 plus the
  draft is about 8.8 GB of 12.3 GB.
- The Q6_K Base checkpoint loops under greedy chat decoding; that is the BASE
  model, not damage. Send temperature 0.8-1.0 and always send max_tokens.
