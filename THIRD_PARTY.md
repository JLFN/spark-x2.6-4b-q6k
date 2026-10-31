# Third-party code in this repository

## tools/vq/vendor and tools/vq/eg2_vq.c

Dense-matrix VQ encoder, copied into this repository so the Spark VQ work does
not depend on checkouts outside it.

| File | Origin | License |
|---|---|---|
| tools/vq/eg2_vq.c | /data/embeddinggemma2/tools/eg2_vq.c, two `#include` paths rewritten to `vendor/` | MIT (ds4.c authors) |
| tools/vq/vendor/vq_qc.h | /data/YoungAi, gguf-tools/quantize/vq_qc.h @ a4e6d31 | MIT |
| tools/vq/vendor/vq_qc_bytes.h | /data/YoungAi, gguf-tools/quantize/vq_qc_bytes.h @ a4e6d31 | MIT |
| tools/vq/vendor/go2b_qc.h | /data/YoungAi, gguf-tools/quantize/go2b_qc.h @ a4e6d31 | MIT |
| tools/vq/vendor/onebit_quant.h | /data/YoungAi, gguf-tools/quantize/onebit_quant.h @ a4e6d31 | MIT |
| tools/vq/vendor/onebit_quant.c | /data/YoungAi, gguf-tools/quantize/onebit_quant.c @ a4e6d31 | MIT |
| tools/vq/vendor/ds4_st41.h | /data/YoungAi, src/common/ds4_st41.h @ a4e6d31 | MIT |
| tools/vq/vendor/ds4_float.h | /data/YoungAi, src/common/ds4_float.h @ a4e6d31 | MIT |
| tools/vq/vendor/ds41_vq_fmt.h | /data/ds4-dfm-rs @ 233dc7a, itself vendored from /data/YoungAi vq_fmt.h @ 3946dbc | MIT |

YoungAi is MIT licensed, Copyright (c) 2026 The ds4.c authors. No file was
modified in content; only eg2_vq.c's include paths changed, and the two local
tools (build.sh, quant_err.c) are new.

## tools/vq/quant_err.c and tools/kquant

quant_err.c is new in this repository. It compares the vendored k-quant
reference encoders with the VQ rows on the same tensor.

tools/kquant/vendor holds an unmodified copy of the llama.cpp quantizer:
`src/ggml-quants.c`, `src/ggml-quants.h`, `src/ggml-common.h`, `src/ggml-impl.h`,
`src/ggml-cpu/ggml-cpu-impl.h`, `include/ggml.h`, `include/ggml-cpu.h`,
`include/ggml-backend.h`, `include/ggml-alloc.h` and `include/gguf.h`, taken from
ggml-org/llama.cpp at fa3645feb5f47cbb30df5e6d031c22543d43d240 (MIT). It is
compiled into `.build/ggml-quants.o`. tools/kquant/stubs.c is new and supplies the
four ggml entry points that file expects (ggml_abort, ggml_type_size,
ggml_blck_size, ggml_row_size, ggml_type_name), so no libggml is needed.

llama.cpp is MIT licensed, Copyright (c) 2023-2024 The ggml authors.
