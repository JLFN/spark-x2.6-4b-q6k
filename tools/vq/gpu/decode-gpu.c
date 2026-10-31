/* decode-gpu.c — host driver for the CUDA decode and GEMV kernels.
 *
 * Reads one RQVQ container and prices three ways to serve the same matrix on a
 * single token:
 *
 *   fused gemv   the runtime that never materialises weights: g Gather+dot per
 *                row, per stage, once per token
 *   dequant      expand the matrix to f32 once (the naive loader)
 *   f32 gemv     multiply against the expanded matrix, the shape whose cost is
 *                set by 4x the bytes per token
 *
 * A host double-precision matvec is the gate: the fused result must match it.
 *
 * Usage: decode-gpu <file.bin> [--iters N] [--check]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

extern const char * vqg_decode_error(void);
extern int vqg_dequant(const void * payload, int dim, int nc, int rows, int cols, float * out, int iters, double * ms);
extern int vqg_gemv(const void * payload, int dim, int nc, int rows, int cols, const float * x, float * y, int variant, int iters, double * ms);
extern int vqg_gemv_f32(const float * w, const float * x, float * y, int rows, int cols, int iters, double * ms);

static uint8_t * slurp(const char * path, size_t * out_n) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    uint8_t * buf = calloc((size_t) sz + 8, 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t) sz, f) != (size_t) sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *out_n = (size_t) sz;
    return buf;
}

static int cuda_ok(cudaError_t e, const char * what) {
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return -1;
}

/* Benchmark every matrix listed in <dir>/manifest.txt and report the decode
 * rate of the whole artifact: container bytes per second, and the modeled
 * tokens per second of the fused path (one pass over all weights per token). */
static int cmd_manifest(const char * dir, int iters) {
    char mpath[1024];
    snprintf(mpath, sizeof mpath, "%s/manifest.txt", dir);
    FILE * mf = fopen(mpath, "r");
    if (!mf) { fprintf(stderr, "cannot read %s\n", mpath); return 1; }

    double fused_ms = 0.0, dequant_ms = 0.0;
    size_t bytes = 0, weights = 0;
    int tensors = 0;
    float * scratch = NULL;
    size_t scratch_bytes = 0;

    struct { char name[96]; double ms; } slow[5];
    for (int i = 0; i < 5; i++) { slow[i].name[0] = 0; slow[i].ms = -1; }

    char line[1024];
    while (fgets(line, sizeof line, mf)) {
        char name[256];
        unsigned rows, cols;
        int stages;
        size_t fbytes;
        float rel;
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%255s %u %u %d %zu %f", name, &rows, &cols, &stages, &fbytes, &rel) != 6) continue;
        if (stages <= 0) continue;   /* 1-D tensors are copied f32, not decoded */

        char path[1200];
        snprintf(path, sizeof path, "%s/%s.bin", dir, name);
        size_t n = 0;
        uint8_t * buf = slurp(path, &n);
        if (!buf) { fprintf(stderr, "cannot read %s\n", path); fclose(mf); return 1; }
        if (n < 32 || memcmp(buf, "RQVQ", 4)) { fprintf(stderr, "%s: not a container\n", path); free(buf); fclose(mf); return 1; }

        uint16_t dim, nc, st;
        uint32_t r32, c32;
        uint64_t payload;
        memcpy(&dim, buf + 4, 2);
        memcpy(&nc, buf + 6, 2);
        memcpy(&st, buf + 8, 2);
        memcpy(&r32, buf + 12, 4);
        memcpy(&c32, buf + 16, 4);
        memcpy(&payload, buf + 24, 8);
        if (n != 32 + payload || st != (uint16_t) stages || r32 != rows || c32 != cols) {
            fprintf(stderr, "%s: header disagrees with the manifest\n", path);
            free(buf); fclose(mf); return 1;
        }

        const size_t per = payload / stages;
        const size_t aper = (per + 7) & ~(size_t) 7;
        const size_t dbuf_bytes = 32 + aper * stages + 8;

        uint8_t * dbuf = NULL;
        float * dx = NULL, * dy = NULL;
        if (cudaMalloc((void **) &dbuf, dbuf_bytes) != cudaSuccess ||
            cudaMemset(dbuf, 0, dbuf_bytes) != cudaSuccess ||
            cudaMalloc((void **) &dx, (size_t) cols * 4) != cudaSuccess ||
            cudaMalloc((void **) &dy, (size_t) rows * 4) != cudaSuccess) {
            fprintf(stderr, "%s: cudaMalloc failed\n", path);
            cudaFree(dbuf); cudaFree(dx); cudaFree(dy); free(buf); fclose(mf);
            return 1;
        }
        for (uint16_t s = 0; s < stages; s++)
            cudaMemcpy(dbuf + 32 + aper * s, buf + 32 + per * s, per, cudaMemcpyHostToDevice);

        float * x = malloc((size_t) cols * 4);
        for (uint32_t i = 0; i < cols; i++) x[i] = sinf((float) i * 0.001f);
        cudaMemcpy(dx, x, (size_t) cols * 4, cudaMemcpyHostToDevice);
        cudaMemset(dy, 0, (size_t) rows * 4);

        double f = 0.0, d = 0.0;
        for (uint16_t s = 0; s < stages; s++) {
            double ms = 0.0;
            if (vqg_gemv(dbuf + 32 + aper * s, dim, nc, (int) rows, (int) cols, dx, dy, 2, iters, &ms)) {
                fprintf(stderr, "%s: %s\n", path, vqg_decode_error());
                free(x); cudaFree(dbuf); cudaFree(dx); cudaFree(dy); free(buf); fclose(mf);
                return 1;
            }
            f += ms;
        }
        for (uint16_t s = 0; s < stages; s++) {
            const size_t need = (size_t) rows * cols * 4;
            if (need > scratch_bytes) {
                if (scratch) cudaFree(scratch);
                if (cudaMalloc((void **) &scratch, need) != cudaSuccess) {
                    fprintf(stderr, "%s: cudaMalloc scratch failed\n", path);
                    free(x); cudaFree(dbuf); cudaFree(dx); cudaFree(dy); free(buf); fclose(mf);
                    return 1;
                }
                scratch_bytes = need;
            }
            cudaMemset(scratch, 0, need);
            double ms = 0.0;
            if (vqg_dequant(dbuf + 32 + aper * s, dim, nc, (int) rows, (int) cols, scratch, iters, &ms)) {
                fprintf(stderr, "%s: %s\n", path, vqg_decode_error());
                free(x); cudaFree(dbuf); cudaFree(dx); cudaFree(dy); free(buf); fclose(mf);
                return 1;
            }
            d += ms;
        }

        fused_ms += f;
        dequant_ms += d;
        bytes += n;
        weights += (size_t) rows * cols;
        tensors++;

        for (int i = 0; i < 5; i++) {
            if (f > slow[i].ms) {
                for (int j = 4; j > i; j--) slow[j] = slow[j - 1];
                snprintf(slow[i].name, sizeof slow[i].name, "%s", name);
                slow[i].ms = f;
                break;
            }
        }

        free(x);
        cudaFree(dbuf);
        cudaFree(dx);
        cudaFree(dy);
        free(buf);
    }
    fclose(mf);

    if (!tensors) { fprintf(stderr, "%s: no matrices\n", mpath); return 1; }
    if (scratch) cudaFree(scratch);

    printf("artifact    %s\n", dir);
    printf("tensors     %d matrices, %zu weights\n", tensors, weights);
    printf("container   %.4f GB (%.4f bits/weight)\n",
           (double) bytes / 1e9, (double) bytes * 8.0 / (double) weights);
    printf("fused       %.3f ms/token, %.2f GB/s of container\n",
           fused_ms, (double) bytes / 1e9 / (fused_ms * 1e-3));
    printf("dequant     %.3f ms, %.2f GB/s of container\n",
           dequant_ms, (double) bytes / 1e9 / (dequant_ms * 1e-3));
    printf("decode rate %.1f tok/s (matrices only, batch 1)\n", 1000.0 / fused_ms);
    printf("slowest     ");
    for (int i = 0; i < 5 && slow[i].ms >= 0; i++)
        printf("%s %.3f ms%s", slow[i].name, slow[i].ms, i < 4 && slow[i + 1].ms >= 0 ? ", " : "\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <file.bin> [--iters N] [--check]\n"
            "       %s --manifest <artifact-dir> [--iters N]\n", argv[0], argv[0]);
        return 2;
    }
    int iters = 20, check = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
    }
    if (iters < 1) iters = 1;
    if (!strcmp(argv[1], "--manifest")) return cmd_manifest(argv[2], iters);

    size_t n = 0;
    uint8_t * buf = slurp(argv[1], &n);
    if (!buf) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    if (n < 32 || memcmp(buf, "RQVQ", 4)) { fprintf(stderr, "%s: not an RQVQ container\n", argv[1]); return 1; }

    uint16_t dim, nc, stages;
    uint32_t rows, cols;
    uint64_t payload;
    memcpy(&dim, buf + 4, 2);
    memcpy(&nc, buf + 6, 2);
    memcpy(&stages, buf + 8, 2);
    memcpy(&rows, buf + 12, 4);
    memcpy(&cols, buf + 16, 4);
    memcpy(&payload, buf + 24, 8);
    if (n != 32 + payload || stages == 0 || dim == 0 || nc == 0) {
        fprintf(stderr, "%s: header and file size disagree\n", argv[1]);
        return 1;
    }
    const size_t per = payload / stages;
    if (per * stages != payload) { fprintf(stderr, "payload does not split into %u stages\n", stages); return 1; }

    uint8_t * dbuf = NULL;
    float * dx = NULL, * dy = NULL, * dw = NULL;
    /* Each stage is copied at an 8-byte boundary: a stage payload in the file
     * starts wherever the previous one ended, which is not necessarily even, and
     * the f16 codebook is read as __half. Eight spare bytes at the end cover the
     * 3-byte index window of the last index. */
    const size_t aper = (per + 7) & ~(size_t) 7;
    const size_t dbuf_bytes = 32 + aper * stages + 8;
    if (cuda_ok(cudaMalloc((void **) &dbuf, dbuf_bytes), "cudaMalloc container")) return 1;
    if (cuda_ok(cudaMemset(dbuf, 0, dbuf_bytes), "zero container")) return 1;
    for (uint16_t s = 0; s < stages; s++) {
        if (cuda_ok(cudaMemcpy(dbuf + 32 + aper * s, buf + 32 + per * s, per, cudaMemcpyHostToDevice),
                    "upload stage")) return 1;
    }
    if (cuda_ok(cudaMalloc((void **) &dx, (size_t) cols * 4), "cudaMalloc x")) return 1;
    if (cuda_ok(cudaMalloc((void **) &dy, (size_t) rows * 4), "cudaMalloc y")) return 1;

    float * x = malloc((size_t) cols * 4);
    for (uint32_t i = 0; i < cols; i++) x[i] = sinf((float) i * 0.001f);
    if (cuda_ok(cudaMemcpy(dx, x, (size_t) cols * 4, cudaMemcpyHostToDevice), "upload x")) return 1;
    if (cuda_ok(cudaMemset(dy, 0, (size_t) rows * 4), "zero y")) return 1;

    /* Fused path: dim-4 fast kernel, plain warp-per-row, and the shared-memory
     * variant, all pumping the same bytes. */
    double fused_ms[3] = {0, 0, 0};
    for (int variant = 2; variant >= 0; variant--) {
        double total = 0.0;
        if (cuda_ok(cudaMemset(dy, 0, (size_t) rows * 4), "zero y")) return 1;
        for (uint16_t s = 0; s < stages; s++) {
            double ms = 0.0;
            if (vqg_gemv(dbuf + 32 + aper * s, dim, nc, (int) rows, (int) cols, dx, dy, variant, iters, &ms)) {
                fprintf(stderr, "stage %u gemv (variant %d): %s\n", s, variant, vqg_decode_error());
                return 1;
            }
            total += ms;
        }
        fused_ms[variant] = total;
    }

    /* Expanded path: dequant every stage into one f32 matrix, then multiply. */
    if (cuda_ok(cudaMalloc((void **) &dw, (size_t) rows * cols * 4), "cudaMalloc w")) return 1;
    if (cuda_ok(cudaMemset(dw, 0, (size_t) rows * cols * 4), "zero w")) return 1;
    double dequant_ms = 0.0;
    for (uint16_t s = 0; s < stages; s++) {
        double ms = 0.0;
        if (vqg_dequant(dbuf + 32 + aper * s, dim, nc, (int) rows, (int) cols, dw, iters, &ms)) {
            fprintf(stderr, "stage %u dequant: %s\n", s, vqg_decode_error());
            return 1;
        }
        dequant_ms += ms;
    }

    /* The timing loop accumulates into the matrix too, so rebuild it once clean
     * for the gate and the f32 baseline. */
    if (cuda_ok(cudaMemset(dw, 0, (size_t) rows * cols * 4), "zero w")) return 1;
    for (uint16_t s = 0; s < stages; s++) {
        double ms = 0.0;
        if (vqg_dequant(dbuf + 32 + aper * s, dim, nc, (int) rows, (int) cols, dw, 1, &ms)) {
            fprintf(stderr, "stage %u final dequant: %s\n", s, vqg_decode_error());
            return 1;
        }
    }

    if (cuda_ok(cudaMemset(dy, 0, (size_t) rows * 4), "zero y")) return 1;
    double f32_ms = 0.0;
    if (vqg_gemv_f32(dw, dx, dy, (int) rows, (int) cols, iters, &f32_ms)) {
        fprintf(stderr, "f32 gemv: %s\n", vqg_decode_error());
        return 1;
    }

    /* The timing loops accumulate into y, so take one clean fused pass for the
     * result and the gate. */
    if (cuda_ok(cudaMemset(dy, 0, (size_t) rows * 4), "zero y")) return 1;
    for (uint16_t s = 0; s < stages; s++) {
        double ms = 0.0;
        if (vqg_gemv(dbuf + 32 + aper * s, dim, nc, (int) rows, (int) cols, dx, dy, 2, 1, &ms)) {
            fprintf(stderr, "stage %u final gemv: %s\n", s, vqg_decode_error());
            return 1;
        }
    }

    float * y = malloc((size_t) rows * 4);
    if (cuda_ok(cudaMemcpy(y, dy, (size_t) rows * 4, cudaMemcpyDeviceToHost), "download y")) return 1;

    const double gb_container = (double) n / 1e9;
    printf("file        %s\n", argv[1]);
    printf("shape       %u x %u, %u stage(s), %.4f bits/weight\n",
           rows, cols, stages, (double) n * 8.0 / ((double) rows * cols));
    printf("container   %zu bytes (%.4f GB)\n", n, gb_container);
    printf("fused fast  %.4f ms, %.2f GB/s of container\n", fused_ms[2], gb_container / (fused_ms[2] * 1e-3));
    printf("fused warp  %.4f ms, %.2f GB/s of container\n", fused_ms[1], gb_container / (fused_ms[1] * 1e-3));
    printf("fused smem  %.4f ms, %.2f GB/s of container\n", fused_ms[0], gb_container / (fused_ms[0] * 1e-3));
    printf("dequant     %.4f ms, %.2f GB/s of container\n", dequant_ms, gb_container / (dequant_ms * 1e-3));
    printf("f32 gemv    %.4f ms, %.2f GB/s of f32 weights\n",
           f32_ms, ((double) rows * cols * 4) / 1e9 / (f32_ms * 1e-3));
    printf("speedup     %.2fx fast vs expand-once-then-multiply\n", f32_ms / fused_ms[2]);

    if (check) {
        float * w = malloc((size_t) rows * cols * 4);
        if (cuda_ok(cudaMemcpy(w, dw, (size_t) rows * cols * 4, cudaMemcpyDeviceToHost), "download w")) return 1;
        double mx = 0.0, se = 0.0, sn = 0.0;
        for (uint32_t r = 0; r < rows; r++) {
            double acc = 0.0;
            const float * wr = w + (size_t) r * cols;
            for (uint32_t c = 0; c < cols; c++) acc += (double) wr[c] * (double) x[c];
            const double d = acc - (double) y[r];
            se += d * d;
            sn += acc * acc;
            if (fabs(d) > mx) mx = fabs(d);
        }
        printf("check       maxabs %.6f, relL2 %.6f%% (fused vs host f64)\n", mx, 100.0 * sqrt(se / (sn + 1e-30)));
        free(w);
    }

    free(y);
    free(x);
    free(buf);
    cudaFree(dw);
    cudaFree(dy);
    cudaFree(dx);
    cudaFree(dbuf);
    return 0;
}
