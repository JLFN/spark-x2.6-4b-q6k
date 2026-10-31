# Spark-X2.6-4B-Base — session handoff

Date: 2026-10-31. Repo: /data/Spark-X2.6-4B-Base, HEAD is this handoff commit,
working tree clean. Remote https://github.com/JLFN/spark-x2.6-4b-q6k is at
9f988c7, so these are unpushed: add6d67 (the C and Rust stream-width fix),
4c62df9 (the tiled prefill kernel and its gates), and this handoff commit.

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

1. Push: three commits local, origin at 9f988c7.
2. A VQ runtime. Nothing serves these containers yet; a decode-rate or a
   prefill-rate number is not a model. This is the last thing between the
   format and a usable artifact.
3. Prefill tuning, only if a runtime needs it. The B-tile load reads xT
   column-major and writes shared token-major (4-way bank conflict on a path
   that is a minor share of the traffic); a 256-token tile would halve the
   A-tile decode per token if shared memory allowed it.
4. A 4x16384 point would need the DQVQ payload and is unmeasured; it would
   also lose both fast kernels, so expect the per-index rates.

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
- The Q6_K Base checkpoint loops under greedy chat decoding; that is the BASE
  model, not damage. Send temperature 0.8-1.0 and always send max_tokens.
