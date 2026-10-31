/* rvq.c — residual / multi-stage VQ encoder for one HF matrix.
 *
 * Why this exists: single-stage dense VQ measured 12.5-17% relL2 at 3.0-3.5 bpw
 * on Spark's FFN matrices, which is Q3_K class and below the Q4_K bar. Each VQ
 * stage adds one index stream, so stacking stages raises the rate instead of the
 * size only. This tool measures whether stacked stages reach Q5_K-class error
 * (3.63% at 5.50 bpw) or Q6_K class (1.79% at 6.56 bpw) on this model.
 *
 * Each stage encodes the residual of the previous one with the reference
 * vq_encode_full (same code as eg2_vq), so stage 1 is byte-identical to eg2_vq.
 * Total bytes are summed with vq_payload_bytes, the shared byte account.
 *
 * Build: tools/vq/build.sh. Usage:
 *   rvq <hf-dir> <tensor-name> [dim nc stages]
 * Defaults: dim 4, nc 4096, stages 2.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* dq_matmul must be defined before the vendored headers: vq_qc.h calls it.
 * Same semantic as eg2_vq.c and the reference
 * (/data/YoungAi/gguf-tools/quantize/vq_shim.c:9): out[S,M] = X[S,K] @ W[M,K]^T */
static void dq_matmul(const float *X, const float *W, float *out, int S, int K, int M) {
    for (int s = 0; s < S; s++)
        for (int m = 0; m < M; m++) {
            const float *x = X + (size_t)s * K, *w = W + (size_t)m * K;
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)x[k] * w[k];
            out[(size_t)s * M + m] = (float)acc;
        }
}

#include "onebit_quant.c"
#include "go2b_qc.h"
#include "vq_qc.h"
#include "vendor/ds4_st41.h"
#include "vendor/ds4_float.h"

int g_vq_dim = 0, g_vq_nc = 0;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <hf-dir> <tensor-name> [dim nc stages]\n", argv[0]);
        return 2;
    }
    const int dim    = argc > 3 ? atoi(argv[3]) : 4;
    const int nc     = argc > 4 ? atoi(argv[4]) : 4096;
    const int stages = argc > 5 ? atoi(argv[5]) : 2;
    if (dim <= 0 || nc <= 1 || stages < 1) { fprintf(stderr, "bad parameters\n"); return 2; }

    v41_st S;
    if (v41_st_open(&S, argv[1])) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    const v41_st_ent * E = v41_st_find(&S, argv[2]);
    if (!E) { fprintf(stderr, "tensor not found: %s\n", argv[2]); return 1; }
    if (E->nd != 2) { fprintf(stderr, "%s: nd=%d, only 2-D matrices\n", argv[2], E->nd); return 1; }

    const int rows = (int) E->shape[0];
    const int cols = (int) E->shape[1];
    if (cols % dim) { fprintf(stderr, "cols %d not divisible by dim %d\n", cols, dim); return 1; }
    const size_t n = (size_t) rows * cols;
    const uint8_t * raw = v41_st_data(&S, E);
    if (!raw) return 1;

    float * W     = malloc(n * 4);
    float * resid = malloc(n * 4);
    float * recon = calloc(n, 4);
    float * Wq    = malloc(n * 4);
    float * Cb    = malloc((size_t) nc * dim * 4);
    float * gr    = malloc((size_t) rows * 4);
    int   * idx   = malloc((n / dim) * sizeof(int));
    for (size_t i = 0; i < n; i++) W[i] = ds4_bf16_to_f32(((const uint16_t *) raw)[i]);
    memcpy(resid, W, n * 4);

    size_t total_bytes = 0;
    const double t0 = now_s();
    for (int s = 0; s < stages; s++) {
        const double ts = now_s();
        vq_encode_full(resid, rows, cols, dim, nc, NULL, 0, Cb, idx, gr, Wq);
        for (size_t i = 0; i < n; i++) { recon[i] += Wq[i]; resid[i] -= Wq[i]; }
        total_bytes += vq_payload_bytes(rows, cols, dim, nc);
        double se = 0, sn = 0;
        for (size_t i = 0; i < n; i++) { double d = (double) resid[i]; se += d * d; }
        for (size_t i = 0; i < n; i++) { double a = (double) W[i]; sn += a * a; }
        printf("  stage %d: %.1fs  residual relL2 after stage = %.4f%%\n",
               s + 1, now_s() - ts, 100.0 * sqrt(se / (sn + 1e-30)));
    }
    const double secs = now_s() - t0;

    double dot = 0, na = 0, nb = 0, se = 0, maxe = 0;
    for (size_t i = 0; i < n; i++) {
        const double a = W[i], b = recon[i];
        dot += a * b; na += a * a; nb += b * b;
        const double d = a - b; se += d * d; if (fabs(d) > maxe) maxe = fabs(d);
    }
    const double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    const double bpw = 8.0 * (double) total_bytes / (double) n;
    printf("%-52s rvq dim=%d nc=%4d stages=%d  cos=%.4f  relL2=%7.4f%%  maxabs=%.5f  bpw=%.4f  %.1fs\n",
           argv[2], dim, nc, stages, cos, 100.0 * sqrt(se / (na + 1e-30)), maxe, bpw, secs);

    free(W); free(resid); free(recon); free(Wq); free(Cb); free(gr); free(idx);
    return 0;
}
