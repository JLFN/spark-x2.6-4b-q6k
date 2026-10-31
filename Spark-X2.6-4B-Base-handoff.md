# Spark-X2.6-4B-Base — session handoff

Date: 2026-10-31. Repo: /data/Spark-X2.6-4B-Base, HEAD 3c2fef3, working tree
clean. Remote https://github.com/JLFN/spark-x2.6-4b-q6k is at f2978e1, so
d22cce1 and 3c2fef3 are local-only until pushed.

Everything below was run in this repository; nothing is a plan dressed as a
result. Where a number is missing it says so.

## 1. Where things stand

- Q6_K GGUF: runs in llama.cpp at 128K, 3.46 GB, 107 tok/s without the draft,
  122-150 tok/s with DSpark.
- VQ artifacts: 4x8192 x2 is the recommended one (3.375 GB, 6.5653 bpw,
  whole-model relL2 3.568%); 4x4096 x2 (3.104 GB, 4.244%) and 4x4096 x3
  (4.655 GB, 2.175%) are built too.
- Decoders: C (bin/decode), Rust (bin/dqvq) and CUDA (bin/decode-gpu) all read
  the container. C and Rust are bit-exact; CUDA matches a host f64 matvec.
- Decode rate: 97.4 tok/s for matrix work at 4x8192 x2, 328.7 GB/s of container.
- Prefill: NOT measured. Both kernel shapes tried today were wrong for a dense
  model (see 3) and their numbers must not be quoted.
- VQ runtime: still does not exist. Nothing but these tools decodes the format.

## 2. Measurements (do not re-derive)

Quality, whole checkpoint, 4,111,892,480 weights, weighted relL2:

    Q5_K            3.06 GB   5.75 bpw   3.626%
    VQ 4x4096 x2    3.10 GB   6.04 bpw   4.244%
    VQ 4x8192 x2    3.38 GB   6.57 bpw   3.569%
    Q6_K            3.46 GB   6.72 bpw   1.784%
    VQ 4x4096 x3    4.66 GB   9.06 bpw   2.175%

Decode rate, batch 1, matrix work only, best of 5 (`decode-gpu --manifest`):

    4x1024 x1    1.289 GB   5.064 ms/token   254.6 GB/s   197.5 tok/s
    4x8192 x2    3.375 GB  10.266 ms/token   328.7 GB/s    97.4 tok/s
    4x4096 x3    4.655 GB  16.196 ms/token   287.4 GB/s    61.7 tok/s

DQV3 (row-local 12-bit + plane) gate on layer 0 down_proj, dim 4 / 8192 / 2:

    C decode vs checkpoint      3.282% relL2   (DQVQ build of the same rate: 3.277%)
    Rust vs C dump              26,214,400 / 26,214,400 exact
    CUDA fused vs host f64      maxabs 2e-6

DQV3 decode speed, same matrix: 0.0881 ms (243 GB/s) against 0.0717 ms for
DQVQ (299 GB/s). The payload is in place; the kernel that exploits it is not.

## 3. Decisions and why

- VQ is not a size win at Q5_K quality on this model, and VQ at 6.57 bpw still
  loses to Q6_K at 6.72. Q6_K is the artifact to serve; 4x8192 x2 is the VQ
  pick when a VQ artifact is wanted.
- The codebook stays f16 in DQV3. ds4's E4M3 would round our learned codewords
  by about 3%, which is a quality change, not a layout change.
- The Rust tools are a cargo workspace (tools/Cargo.toml). build.sh fails when
  cargo is missing; there is no "skipped rustc" branch any more.
- The prefill kernel was tried twice and abandoned, not tuned: the first
  uncoalesced the activations by 8 KB (84 s for 512 tokens, 6 tok/s), the second
  gave each thread one token and no reuse (494 ms for one token). A dense model
  needs a tiled GEMM; ds4's fused prefill is MoE-shaped and does not transfer.

## 4. Open work, in priority order

1. Warp block-load decode kernel. Three 128-byte lines per warp, 96 words in
   registers, funnelshift per round, plane bit by one shuffle. Reference:
   cuda/ds41_vq_row.cuh:122-137 and :200-220 in /data/ds4-dfm-rs. Gate: the
   three checks above, then speed against 299 GB/s.
2. Tiled prefill. Reference: cuda/ds41_vq_prefill_mma.cuh:1-40, 128 rows x
   128 tokens x BK 64, bf16 products with f32 accumulate, codebook decoded into
   shared. Gate: host matvec per token block, then prefill_tps.
3. `vq-bench` prefill numbers are not usable until 2; the harness itself (ds4
   CSV columns, compare) is fine.
4. Nothing has been pushed since f2978e1.

## 5. Resume

    ./tools/vq/build.sh                       # C tools + cargo workspace
    ./bin/decode info <file.bin>              # header, format, consistency
    ./bin/decode check <file.bin> models/hf <tensor>
    ./bin/decode dump <file.bin> <out.f32> && ./bin/dqvq check <file.bin> <out.f32>
    ./bin/decode-gpu --manifest models/vq-build-8192-2 --iters 5
    ./bin/decode-gpu <file.bin> --check
    ./bin/vq-bench models/vq-build-8192-2 --iters 5 [--prefill N] [--csv]
    ./bin/vq-bench compare a.csv b.csv

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
- rustc defaults to edition 2015; the crates are edition 2021 in
  tools/Cargo.toml, and Cargo.toml sets lto and codegen-units for release.
- ds4-core's Rust f16 decoder (crates/ds4-core/src/vq.rs, the e == 0 branch at
  112) decodes subnormals one binade low; vq_fmt.h:22-28 and dqvq use 113.
- Two llama-server instances do not fit on this card; one at 128K q8_0 plus the
  draft is about 8.8 GB of 12.3 GB.
- The Q6_K Base checkpoint loops under greedy chat decoding; that is the BASE
  model, not damage. Send temperature 0.8-1.0 and always send max_tokens.
