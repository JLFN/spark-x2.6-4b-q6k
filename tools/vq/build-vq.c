/* build-vq.c — convert a HF safetensors checkpoint into VQ tensors.
 *
 * Walks model.safetensors.index.json, encodes every 2-D matrix with the
 * reference VQ encoder (vq_encode_full, the same code eg2_vq and rvq use), and
 * writes one file per tensor plus a manifest. 1-D tensors (norms) are copied as
 * f32 so the result is a complete weight set, not just the matrices.
 *
 * Container per tensor, concatenated stages:
 *   [8B magic "RQVQ"][u16 dim][u16 nc][u16 stages][u16 pad]
 *   [u32 rows][u32 cols][u64 per-stage payload bytes]
 *   then `stages` payloads, each exactly what vq_pack writes:
 *   [u32 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]
 *   [codebook nc*dim f16][gains rows f16][index stream]
 *
 * Build: tools/vq/build.sh. Usage:
 *   build-vq <hf-dir> <out-dir> <dim> <nc> <stages> [options]
 *     --match SUBSTR   only tensors whose name contains SUBSTR (repeatable)
 *     --skip SUBSTR    skip tensors whose name contains SUBSTR (repeatable)
 *     --jobs N         worker processes (default 1)
 *     --no-embed       copy the tied embedding instead of encoding it
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

/* dq_matmul must exist before the vendored headers (vq_qc.h calls it). */
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

#define MAXMATCH 16

static int match_any(const char * name, char ** pats, int n) {
    if (n == 0) return 1;
    for (int i = 0; i < n; i++) if (strstr(name, pats[i])) return 1;
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <hf-dir> <out-dir> <dim> <nc> <stages> "
                        "[--match S] [--skip S] [--jobs N] [--no-embed]\n", argv[0]);
        return 2;
    }
    const char * hf  = argv[1];
    const char * out = argv[2];
    const int dim = atoi(argv[3]), nc = atoi(argv[4]), stages = atoi(argv[5]);
    if (dim <= 0 || nc <= 1 || stages < 1) { fprintf(stderr, "bad encoding\n"); return 2; }

    char * match[MAXMATCH]; int nmatch = 0;
    char * skip[MAXMATCH];  int nskip = 0;
    int jobs = 1, no_embed = 0;
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "--match") && i + 1 < argc && nmatch < MAXMATCH) match[nmatch++] = argv[++i];
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc && nskip < MAXMATCH) skip[nskip++] = argv[++i];
        else if (!strcmp(argv[i], "--jobs") && i + 1 < argc) jobs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-embed")) no_embed = 1;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (jobs < 1) jobs = 1;

    v41_st S;
    if (v41_st_open(&S, hf)) { fprintf(stderr, "cannot open %s\n", hf); return 1; }
    mkdir(out, 0775);

    /* Build the work list once, before forking, so every worker sees the same
     * ordering without touching the parent's mmaps. */
    int * work = malloc((size_t) S.ne * sizeof(int));
    int nwork = 0;
    for (int t = 0; t < S.ne; t++) {
        const v41_st_ent * E = &S.e[t];
        if (!match_any(E->name, match, nmatch)) continue;
        int skipped = 0;
        for (int i = 0; i < nskip; i++) if (strstr(E->name, skip[i])) skipped = 1;
        if (skipped) continue;
        if (E->nd == 2 && no_embed && strstr(E->name, "embedding")) continue;
        work[nwork++] = t;
    }
    printf("build-vq: %d tensors selected, dim=%d nc=%d stages=%d jobs=%d\n",
           nwork, dim, nc, stages, jobs);

    char mpath[4096];
    snprintf(mpath, sizeof mpath, "%s/manifest-%04d.txt", out, 0);
    (void) mpath;

    int workers = jobs < nwork ? jobs : nwork;
    pid_t * pids = calloc((size_t) workers, sizeof(pid_t));
    for (int w = 0; w < workers; w++) {
        pid_t p = fork();
        if (p == 0) {
            char man[4096];
            snprintf(man, sizeof man, "%s/manifest-%04d.txt", out, w);
            FILE * mf = fopen(man, "w");
            for (int i = w; i < nwork; i += workers) {
                const v41_st_ent * E = &S.e[work[i]];
                size_t n = 1;
                for (int d = 0; d < E->nd; d++) n *= (size_t) E->shape[d];

                char path[4096];
                snprintf(path, sizeof path, "%s/%s.bin", out, E->name);
                FILE * f = fopen(path, "wb");
                if (!f) { fprintf(stderr, "cannot write %s\n", path); continue; }

                if (E->nd != 2) {
                    /* 1-D tensor: copy as f32 */
                    const uint8_t * raw = v41_st_data(&S, E);
                    float * v = malloc(n * 4);
                    for (size_t k = 0; k < n; k++) v[k] = ds4_bf16_to_f32(((const uint16_t *) raw)[k]);
                    fwrite("RQVF", 1, 4, f);
                    uint32_t rows = (uint32_t) n, cols = 1;
                    fwrite(&rows, 4, 1, f); fwrite(&cols, 4, 1, f);
                    fwrite(v, 4, n, f);
                    fprintf(mf, "%s %zu 1 0 %zu\n", E->name, n, 12 + 4 * n);
                    fclose(f); free(v);
                    continue;
                }

                const uint32_t rows = (uint32_t) E->shape[0];
                const uint32_t cols = (uint32_t) E->shape[1];
                if (cols % (uint32_t) dim) {
                    fprintf(stderr, "%s: cols %u not divisible by dim %d, skipped\n", E->name, cols, dim);
                    fclose(f); remove(path);
                    continue;
                }

                const uint8_t * raw = v41_st_data(&S, E);
                float * W = malloc(n * 4);
                float * resid = malloc(n * 4);
                float * Wq = malloc(n * 4);
                float * Cb = malloc((size_t) nc * dim * 4);
                float * gr = malloc((size_t) rows * 4);
                int   * idx = malloc((n / (size_t) dim) * sizeof(int));
                uint8_t * buf = malloc(vq_payload_bytes((int) rows, (int) cols, dim, nc) + 64);

                for (size_t k = 0; k < n; k++) W[k] = ds4_bf16_to_f32(((const uint16_t *) raw)[k]);
                memcpy(resid, W, n * 4);

                const size_t psz = vq_payload_bytes((int) rows, (int) cols, dim, nc);
                uint8_t hdr[32];
                memcpy(hdr, "RQVQ", 4);
                uint16_t d16 = (uint16_t) dim, n16 = (uint16_t) nc, s16 = (uint16_t) stages;
                memcpy(hdr + 4, &d16, 2); memcpy(hdr + 6, &n16, 2); memcpy(hdr + 8, &s16, 2);
                uint16_t pad = 0; memcpy(hdr + 10, &pad, 2);
                memcpy(hdr + 12, &rows, 4); memcpy(hdr + 16, &cols, 4);
                uint64_t psz64 = (uint64_t) psz * (uint64_t) stages;
                memcpy(hdr + 24, &psz64, 8);
                fwrite(hdr, 1, 32, f);

                for (int s = 0; s < stages; s++) {
                    vq_encode_full(resid, (int) rows, (int) cols, dim, nc, NULL, 0, Cb, idx, gr, Wq);
                    const size_t wrote = vq_pack(Cb, idx, gr, (int) rows, (int) cols, dim, nc, buf);
                    if (wrote != psz) { fprintf(stderr, "%s: pack %zu != %zu\n", E->name, wrote, psz); break; }
                    fwrite(buf, 1, wrote, f);
                    for (size_t k = 0; k < n; k++) resid[k] -= Wq[k];
                }
                fprintf(mf, "%s %u %u %d %zu\n", E->name, rows, cols, stages, 32 + psz64);
                fflush(mf);
                fclose(f);
                free(W); free(resid); free(Wq); free(Cb); free(gr); free(idx); free(buf);
            }
            fclose(mf);
            _exit(0);
        }
        pids[w] = p;
    }
    for (int w = 0; w < workers; w++) waitpid(pids[w], NULL, 0);

    /* Concatenate the per-worker manifests. */
    char all[4096];
    snprintf(all, sizeof all, "%s/manifest.txt", out);
    FILE * mf = fopen(all, "w");
    fprintf(mf, "# encoding: dim=%d nc=%d stages=%d\n", dim, nc, stages);
    fprintf(mf, "# name rows cols stages file_bytes\n");
    for (int w = 0; w < workers; w++) {
        char man[4096];
        snprintf(man, sizeof man, "%s/manifest-%04d.txt", out, w);
        FILE * in = fopen(man, "r");
        if (!in) continue;
        char line[8192];
        while (fgets(line, sizeof line, in)) fputs(line, mf);
        fclose(in);
    }
    fclose(mf);
    printf("build-vq: wrote %s\n", all);
    return 0;
}
