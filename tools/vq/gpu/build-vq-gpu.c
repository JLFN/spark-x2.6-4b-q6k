/* build-vq-gpu.c — GPU driver for the dense VQ encoder.
 *
 * Same container as build-vq (RQVQ header, then one vq_pack payload per stage),
 * but the Lloyd loop and the assignment run on the GPU through
 * tools/vq/gpu/vq_gpu.cu. The host keeps the parts that are cheap after the
 * assignment: the f16 codebook round-trip, the row gains (1.0 on the
 * uncalibrated path) and vq_pack.
 *
 * Usage: build-vq-gpu <hf-dir> <out-dir> <dim> <nc> <stages> [--match S] [--skip S] [--no-embed]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <sys/stat.h>
#include <time.h>

/* vq_qc.h calls dq_matmul, so it must exist before the include. */
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
#include "ds4_st41.h"
#include "ds4_float.h"

extern int vqg_train(const float * X, int nvec, int dim, int nc, int iters, float * C);
extern int vqg_assign(const float * X, int nvec, int dim, const float * C, int nc, int * idx);
extern const char * vqg_last_error(void);

#define MAXMATCH 16

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int match_any(const char * name, char ** pats, int n) {
    if (n == 0) return 1;
    for (int i = 0; i < n; i++) if (strstr(name, pats[i])) return 1;
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <hf-dir> <out-dir> <dim> <nc> <stages> "
                        "[--match S] [--skip S] [--no-embed]\n", argv[0]);
        return 2;
    }
    const char * hf = argv[1], * out = argv[2];
    const int dim = atoi(argv[3]), nc = atoi(argv[4]), stages = atoi(argv[5]);
    if (dim <= 0 || nc <= 1 || stages < 1) { fprintf(stderr, "bad encoding\n"); return 2; }

    char * match[MAXMATCH]; int nmatch = 0;
    char * skip[MAXMATCH];  int nskip = 0;
    int no_embed = 0;
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "--match") && i + 1 < argc && nmatch < MAXMATCH) match[nmatch++] = argv[++i];
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc && nskip < MAXMATCH) skip[nskip++] = argv[++i];
        else if (!strcmp(argv[i], "--no-embed")) no_embed = 1;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }

    v41_st S;
    if (v41_st_open(&S, hf)) { fprintf(stderr, "cannot open %s\n", hf); return 1; }
    mkdir(out, 0775);

    FILE * mf = NULL;
    char mpath[4096];
    snprintf(mpath, sizeof mpath, "%s/manifest.txt", out);
    mf = fopen(mpath, "w");
    fprintf(mf, "# encoding: dim=%d nc=%d stages=%d (gpu)\n", dim, nc, stages);
    fprintf(mf, "# name rows cols stages file_bytes relL2\n");

    int done = 0;
    const double t_start = now_s();
    for (int t = 0; t < S.ne; t++) {
        const v41_st_ent * E = &S.e[t];
        if (!match_any(E->name, match, nmatch)) continue;
        int skipped = 0;
        for (int i = 0; i < nskip; i++) if (strstr(E->name, skip[i])) skipped = 1;
        if (skipped) continue;
        if (E->nd == 2 && no_embed && strstr(E->name, "embedding")) continue;

        size_t n = 1;
        for (int d = 0; d < E->nd; d++) n *= (size_t) E->shape[d];
        char path[4096];
        snprintf(path, sizeof path, "%s/%s.bin", out, E->name);
        FILE * f = fopen(path, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", path); continue; }

        const uint8_t * raw = v41_st_data(&S, E);
        if (E->nd != 2) {
            float * v = malloc(n * 4);
            for (size_t k = 0; k < n; k++) v[k] = ds4_bf16_to_f32(((const uint16_t *) raw)[k]);
            fwrite("RQVF", 1, 4, f);
            uint32_t rows = (uint32_t) n, cols = 1;
            fwrite(&rows, 4, 1, f); fwrite(&cols, 4, 1, f);
            fwrite(v, 4, n, f);
            fprintf(mf, "%s %zu 1 0 %zu -1\n", E->name, n, 12 + 4 * n);
            fclose(f); free(v);
            continue;
        }

        const uint32_t rows = (uint32_t) E->shape[0], cols = (uint32_t) E->shape[1];
        if (cols % (uint32_t) dim) { fprintf(stderr, "%s: dim mismatch\n", E->name); fclose(f); remove(path); continue; }
        const size_t nvec = n / (size_t) dim;

        float * W = malloc(n * 4);
        for (size_t k = 0; k < n; k++) W[k] = ds4_bf16_to_f32(((const uint16_t *) raw)[k]);

        /* Same subsample rule as vq_encode_full: at most 200k vectors, even stride. */
        const int nsub = nvec < 200000 ? (int) nvec : 200000;
        const int stride = (int) (nvec / (size_t) nsub);
        float * sub = malloc((size_t) nsub * dim * 4);
        for (int i = 0; i < nsub; i++) memcpy(sub + (size_t) i * dim, W + (size_t) i * stride * dim, (size_t) dim * 4);

        float * C = malloc((size_t) nc * dim * 4);
        const int cst = nsub / nc < 1 ? 1 : nsub / nc;
        for (int c = 0; c < nc; c++) {
            const int src = c * cst < nsub ? c * cst : nsub - 1;
            memcpy(C + (size_t) c * dim, sub + (size_t) src * dim, (size_t) dim * 4);
        }

        const double t0 = now_s();
        if (vqg_train(sub, nsub, dim, nc, 8, C)) { fprintf(stderr, "gpu train failed: %s\n", vqg_last_error()); return 1; }
        for (int i = 0; i < nc * dim; i++) C[i] = go1b_fp16_to_fp32(go1b_fp32_to_fp16(C[i]));

        int * idx = malloc(nvec * sizeof(int));
        if (vqg_assign(W, (int) nvec, dim, C, nc, idx)) { fprintf(stderr, "gpu assign failed: %s\n", vqg_last_error()); return 1; }
        const double secs = now_s() - t0;

        float * gr = malloc((size_t) rows * 4);
        for (uint32_t r = 0; r < rows; r++) gr[r] = 1.0f;

        const size_t psz = vq_payload_bytes((int) rows, (int) cols, dim, nc);
        uint8_t * buf = malloc(psz + 64);
        const size_t wrote = vq_pack(C, idx, gr, (int) rows, (int) cols, dim, nc, buf);
        if (wrote != psz) { fprintf(stderr, "%s: pack %zu != %zu\n", E->name, wrote, psz); }

        uint8_t hdr[32];
        memcpy(hdr, "RQVQ", 4);
        uint16_t d16 = (uint16_t) dim, n16 = (uint16_t) nc, s16 = (uint16_t) stages, pad = 0;
        memcpy(hdr + 4, &d16, 2); memcpy(hdr + 6, &n16, 2); memcpy(hdr + 8, &s16, 2); memcpy(hdr + 10, &pad, 2);
        memcpy(hdr + 12, &rows, 4); memcpy(hdr + 16, &cols, 4);
        uint64_t psz64 = (uint64_t) psz * (uint64_t) stages;
        memcpy(hdr + 24, &psz64, 8);
        fwrite(hdr, 1, 32, f);
        fwrite(buf, 1, wrote, f);

        /* relL2 of the reconstruction so the manifest carries quality. */
        double se = 0, sn = 0;
        for (size_t k = 0; k < n; k++) {
            const size_t v = k / (size_t) dim;
            const float q = C[(size_t) idx[v] * dim + (k % (size_t) dim)];
            const double d = (double) q - (double) W[k];
            se += d * d; sn += (double) W[k] * (double) W[k];
        }
        fprintf(mf, "%s %u %u %d %zu %.4f\n", E->name, rows, cols, stages, 32 + psz, 100.0 * sqrt(se / (sn + 1e-30)));
        fclose(f);
        printf("%-52s %5u x %5u  %6.1fs  relL2 %6.2f%%\n", E->name, rows, cols, secs, 100.0 * sqrt(se / (sn + 1e-30)));

        free(W); free(sub); free(C); free(idx); free(gr); free(buf);
        done++;
    }
    fclose(mf);
    printf("build-vq-gpu: %d tensors in %.1fs, manifest %s\n", done, now_s() - t_start, mpath);
    return 0;
}
