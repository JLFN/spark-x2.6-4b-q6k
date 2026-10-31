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

## tools/vq/quant_err.c

New in this repository. It links llama.cpp's libggml-base and calls the exported
reference quantizers (quantize_row_q*_ref, dequantize_row_q*), so the k-quant
error numbers come from llama.cpp itself rather than a reimplementation.
