/* quant_err.c — llama.cpp quantization error on one HF safetensors matrix.
 *
 * Reports relL2 (percent) and bits per weight for a named llama.cpp type, so a
 * VQ measurement from eg2_vq can be compared against the k-quants on the exact
 * same tensor. It calls llama.cpp's own reference quantizers, so the numbers are
 * the real encodings, not a proxy.
 *
 * Build: tools/vq/build.sh. Usage:
 *   quant_err <hf-dir> <tensor-name> <type>
 * type: q8_0 | q6_K | q5_K | q4_K | q3_K | q2_K
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>

#include "ggml.h"
#include "ggml-quants.h"

#include "vendor/ds4_st41.h"
#include "vendor/ds4_float.h"

static int quantize_and_back(const char * type, const float * W, size_t n,
                             float ** out_back, double * out_bpw) {
    size_t bs = 0;
    int rc = 0;
    void * q = NULL;
    float * back = malloc(n * sizeof(float));

    if (!strcmp(type, "q8_0")) {
        bs = sizeof(block_q8_0);
        q = malloc((n / QK8_0) * bs);
        quantize_row_q8_0_ref(W, (block_q8_0 *) q, (int64_t) n);
        dequantize_row_q8_0((const block_q8_0 *) q, back, (int64_t) n);
        *out_bpw = 8.0 * (double) (n / QK8_0) * (double) bs / (double) n;
    } else if (!strcmp(type, "q6_K")) {
        bs = sizeof(block_q6_K);
        q = malloc((n / QK_K) * bs);
        quantize_row_q6_K_ref(W, (block_q6_K *) q, (int64_t) n);
        dequantize_row_q6_K((const block_q6_K *) q, back, (int64_t) n);
        *out_bpw = 8.0 * (double) (n / QK_K) * (double) bs / (double) n;
    } else if (!strcmp(type, "q5_K")) {
        bs = sizeof(block_q5_K);
        q = malloc((n / QK_K) * bs);
        quantize_row_q5_K_ref(W, (block_q5_K *) q, (int64_t) n);
        dequantize_row_q5_K((const block_q5_K *) q, back, (int64_t) n);
        *out_bpw = 8.0 * (double) (n / QK_K) * (double) bs / (double) n;
    } else if (!strcmp(type, "q4_K")) {
        bs = sizeof(block_q4_K);
        q = malloc((n / QK_K) * bs);
        quantize_row_q4_K_ref(W, (block_q4_K *) q, (int64_t) n);
        dequantize_row_q4_K((const block_q4_K *) q, back, (int64_t) n);
        *out_bpw = 8.0 * (double) (n / QK_K) * (double) bs / (double) n;
    } else if (!strcmp(type, "q3_K")) {
        bs = sizeof(block_q3_K);
        q = malloc((n / QK_K) * bs);
        quantize_row_q3_K_ref(W, (block_q3_K *) q, (int64_t) n);
        dequantize_row_q3_K((const block_q3_K *) q, back, (int64_t) n);
        *out_bpw = 8.0 * (double) (n / QK_K) * (double) bs / (double) n;
    } else if (!strcmp(type, "q2_K")) {
        bs = sizeof(block_q2_K);
        q = malloc((n / QK_K) * bs);
        quantize_row_q2_K_ref(W, (block_q2_K *) q, (int64_t) n);
        dequantize_row_q2_K((const block_q2_K *) q, back, (int64_t) n);
        *out_bpw = 8.0 * (double) (n / QK_K) * (double) bs / (double) n;
    } else {
        fprintf(stderr, "unsupported type %s\n", type);
        rc = 1;
    }
    free(q);
    if (rc) { free(back); return 1; }
    *out_back = back;
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <hf-dir> <tensor-name> <type: q8_0|q6_K|q5_K|q4_K|q3_K|q2_K>\n", argv[0]);
        return 2;
    }
    v41_st S;
    if (v41_st_open(&S, argv[1])) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    const v41_st_ent * E = v41_st_find(&S, argv[2]);
    if (!E) { fprintf(stderr, "tensor not found: %s\n", argv[2]); return 1; }
    if (E->nd != 2) { fprintf(stderr, "%s: nd=%d, only 2-D matrices\n", argv[2], E->nd); return 1; }

    const int rows = (int) E->shape[0];
    const int cols = (int) E->shape[1];
    const size_t n = (size_t) rows * cols;
    const uint8_t * raw = v41_st_data(&S, E);
    if (!raw) return 1;

    float * W = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) W[i] = ds4_bf16_to_f32(((const uint16_t *) raw)[i]);

    if (n % QK_K) { fprintf(stderr, "size not divisible by %d\n", QK_K); return 1; }

    float * back = NULL;
    double bpw = 0.0;
    if (quantize_and_back(argv[3], W, n, &back, &bpw)) return 1;

    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double) back[i] - (double) W[i];
        num += d * d;
        den += (double) W[i] * (double) W[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    printf("%-52s %-5s relL2=%7.4f%%  maxabs=%.5f  bpw=%.4f  n=%zu\n",
           argv[2], argv[3], sqrt(num / den) * 100.0, mx, bpw, n);
    free(W); free(back);
    return 0;
}
