/* decode.c — reference decoder for an RQVQ VQ container, plus a GEMV and a
 * source check. It is deliberately small and dependency-free: it reads the
 * container bytes and the HF checkpoint, and nothing else.
 *
 * Container written by build-vq / build-vq-gpu:
 *   [u8 'RQVQ'][u16 dim][u16 nc][u16 stages][u16 pad][u32 rows][u32 cols]
 *   [u64 payload_bytes]                      total bytes of all stage payloads
 *   then `stages` payloads, each:
 *   [u8 'DQVQ'][u16 dim][u16 nc][u32 rows][u32 cols]
 *   [codebook nc*dim f16][row gains rows f16][bit-packed indices]
 *
 * Indices are ceil(log2 nc) bits, little-endian bit order; when the width is
 * exactly 8 they are stored one per byte, and otherwise the stream carries one
 * trailing byte so a 3-byte window read never leaves the payload. This is the
 * same layout vq_pack writes in vendor/vq_qc.h and vq_qc_bytes.h prices.
 *
 * Usage:
 *   decode info   <file.bin>
 *   decode check  <file.bin> <hf-dir> <tensor-name>
 *   decode matvec <file.bin> <vec.f32> [iters]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "vendor/ds4_st41.h"
#include "vendor/ds4_float.h"

#define RQVQ_MAGIC 0x51565152u   /* 'RQVQ' little-endian */
#define DQVQ_MAGIC 0x51565144u   /* 'DQVQ' little-endian */

typedef struct {
    uint16_t dim, nc, stages;
    uint32_t rows, cols;
    uint64_t payload;
    size_t file_bytes;
} hdr_t;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) & 1u;
    uint32_t e = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t m = (uint32_t)h & 0x3ffu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) { f = s << 31; }
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3ffu;
            f = (s << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = (s << 31) | (0xffu << 23) | (m << 13);
    } else {
        f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static uint8_t * slurp(const char * path, size_t * out_n) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    /* Eight spare bytes: the 3-byte index window may look past the last index. */
    uint8_t * buf = calloc((size_t) sz + 8, 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t) sz, f) != (size_t) sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *out_n = (size_t) sz;
    return buf;
}

static int parse_header(const uint8_t * buf, size_t n, hdr_t * h) {
    if (n < 32 || memcmp(buf, "RQVQ", 4)) return -1;
    memcpy(&h->dim, buf + 4, 2);
    memcpy(&h->nc, buf + 6, 2);
    memcpy(&h->stages, buf + 8, 2);
    memcpy(&h->rows, buf + 12, 4);
    memcpy(&h->cols, buf + 16, 4);
    memcpy(&h->payload, buf + 24, 8);
    h->file_bytes = n;
    if (!h->dim || !h->nc || !h->stages || !h->rows || !h->cols) return -1;
    return 0;
}

static int expected_payload(const hdr_t * h, size_t * out) {
    const size_t nidx = (size_t) h->rows * h->cols / h->dim;
    int bits = 1;
    while ((1u << bits) < h->nc) bits++;
    const size_t idx = (bits == 8) ? nidx : (nidx * (size_t) bits + 7) / 8 + 1;
    *out = (size_t) h->stages * (16 + (size_t) h->nc * h->dim * 2 + (size_t) h->rows * 2 + idx);
    return 0;
}

/* Decode one DQVQ payload and add it into out[rows*cols]. */
static int decode_stage(const uint8_t * p, float * out, uint32_t * rows_o, uint32_t * cols_o) {
    if (memcmp(p, "DQVQ", 4)) return -1;
    uint16_t dim, nc;
    uint32_t rows, cols;
    memcpy(&dim, p + 4, 2);
    memcpy(&nc, p + 6, 2);
    memcpy(&rows, p + 8, 4);
    memcpy(&cols, p + 12, 4);
    *rows_o = rows;
    *cols_o = cols;

    const uint8_t * q = p + 16;
    float * C = malloc((size_t) nc * dim * 4);
    float * gr = malloc((size_t) rows * 4);
    if (!C || !gr) { free(C); free(gr); return -1; }
    for (size_t i = 0; i < (size_t) nc * dim; i++) {
        uint16_t v;
        memcpy(&v, q + i * 2, 2);
        C[i] = h2f(v);
    }
    q += (size_t) nc * dim * 2;
    for (uint32_t r = 0; r < rows; r++) {
        uint16_t v;
        memcpy(&v, q + (size_t) r * 2, 2);
        gr[r] = h2f(v);
    }
    q += (size_t) rows * 2;

    const size_t nidx = (size_t) rows * cols / dim;
    int bits = 1;
    while ((1u << bits) < nc) bits++;
    const uint32_t mask = (1u << bits) - 1u;

    for (size_t i = 0; i < nidx; i++) {
        uint32_t v;
        if (bits == 8) {
            v = q[i];
        } else {
            const size_t bit = i * (size_t) bits;
            v = (uint32_t) q[bit >> 3]
              | ((uint32_t) q[(bit >> 3) + 1] << 8)
              | ((uint32_t) q[(bit >> 3) + 2] << 16);
            v = (v >> (bit & 7)) & mask;
        }
        if (v >= nc) { free(C); free(gr); return -1; }
        const uint32_t r = (uint32_t) ((i * dim) / cols);
        const float g = gr[r];
        const float * cb = C + (size_t) v * dim;
        float * o = out + i * dim;
        for (uint16_t d = 0; d < dim; d++) o[d] += cb[d] * g;
    }
    free(C);
    free(gr);
    return 0;
}

static int decode_all(const uint8_t * buf, size_t n, hdr_t * h, float ** out_w) {
    if (parse_header(buf, n, h)) return -1;
    size_t want = 0;
    expected_payload(h, &want);
    if (want != (size_t) h->payload) {
        fprintf(stderr, "payload mismatch: header says %llu, layout implies %zu\n",
                (unsigned long long) h->payload, want);
        return -1;
    }
    if (n != 32 + want) {
        fprintf(stderr, "file is %zu bytes, header+payload is %zu\n", n, 32 + want);
        return -1;
    }

    const size_t nw = (size_t) h->rows * h->cols;
    float * W = calloc(nw, 4);
    if (!W) return -1;

    const size_t per = want / h->stages;
    for (uint16_t s = 0; s < h->stages; s++) {
        uint32_t r = 0, c = 0;
        if (decode_stage(buf + 32 + per * s, W, &r, &c) || r != h->rows || c != h->cols) {
            fprintf(stderr, "stage %u does not match the container header\n", s);
            free(W);
            return -1;
        }
    }
    *out_w = W;
    return 0;
}

static int cmd_info(const char * path) {
    size_t n = 0;
    uint8_t * buf = slurp(path, &n);
    if (!buf) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    hdr_t h;
    if (parse_header(buf, n, &h)) { fprintf(stderr, "%s: not an RQVQ container\n", path); return 1; }
    size_t want = 0;
    expected_payload(&h, &want);

    const size_t nidx = (size_t) h.rows * h.cols / h.dim;
    int bits = 1;
    while ((1u << bits) < h.nc) bits++;
    const size_t idxbytes = (bits == 8) ? nidx : (nidx * (size_t) bits + 7) / 8 + 1;

    printf("file        %s\n", path);
    printf("bytes       %zu (%.4f GB)\n", n, (double) n / 1e9);
    printf("shape       %u x %u, %zu weights\n", h.rows, h.cols, (size_t) h.rows * h.cols);
    printf("per stage   dim %u, nc %u (%d-bit index), %zu bytes\n",
           h.dim, h.nc, bits, want / h.stages);
    printf("stages      %u, payload %llu, total %zu\n",
           h.stages, (unsigned long long) h.payload, 32 + want);
    printf("bits/weight %.4f\n", n * 8.0 / ((double) h.rows * h.cols));
    printf("layout      codebook %zu, gains %zu, indices %zu\n",
           (size_t) h.nc * h.dim * 2, (size_t) h.rows * 2, idxbytes);
    printf("consistent  %s\n",
           (n == 32 + want && h.payload == want) ? "yes" : "NO (truncated or mislabelled)");
    free(buf);
    return 0;
}

static int cmd_check(const char * path, const char * hf, const char * name) {
    size_t n = 0;
    uint8_t * buf = slurp(path, &n);
    if (!buf) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    hdr_t h;
    float * W = NULL;
    if (decode_all(buf, n, &h, &W)) { free(buf); return 1; }

    v41_st S;
    if (v41_st_open(&S, hf)) { fprintf(stderr, "cannot open %s\n", hf); free(W); free(buf); return 1; }
    const v41_st_ent * E = v41_st_find(&S, name);
    if (!E) { fprintf(stderr, "tensor not found: %s\n", name); free(W); free(buf); return 1; }
    const uint32_t rows = (uint32_t) E->shape[0], cols = (uint32_t) E->shape[1];
    if (rows != h.rows || cols != h.cols) {
        fprintf(stderr, "%s is %u x %u, container is %u x %u\n", name, rows, cols, h.rows, h.cols);
        free(W); free(buf); return 1;
    }
    const uint8_t * raw = v41_st_data(&S, E);
    const size_t nw = (size_t) rows * cols;
    double se = 0, sn = 0, mx = 0;
    for (size_t k = 0; k < nw; k++) {
        const double w = ds4_bf16_to_f32(((const uint16_t *) raw)[k]);
        const double d = (double) W[k] - w;
        se += d * d;
        sn += w * w;
        if (fabs(d) > mx) mx = fabs(d);
    }
    printf("%-52s %5u x %5u  stages %u  relL2 %6.3f%%  maxabs %.5f\n",
           name, rows, cols, h.stages, 100.0 * sqrt(se / (sn + 1e-30)), mx);
    free(W);
    free(buf);
    return 0;
}

static int cmd_dump(const char * path, const char * out) {
    size_t n = 0;
    uint8_t * buf = slurp(path, &n);
    if (!buf) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    hdr_t h;
    float * W = NULL;
    if (decode_all(buf, n, &h, &W)) { free(buf); return 1; }
    FILE * f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); free(W); free(buf); return 1; }
    const size_t nw = (size_t) h.rows * h.cols;
    const size_t wrote = fwrite(W, 4, nw, f);
    fclose(f);
    free(W);
    free(buf);
    if (wrote != nw) { fprintf(stderr, "%s: short write\n", out); return 1; }
    printf("%s: %u x %u, %zu floats\n", out, h.rows, h.cols, nw);
    return 0;
}

static int cmd_matvec(const char * path, const char * vecpath, int iters) {
    size_t n = 0;
    uint8_t * buf = slurp(path, &n);
    if (!buf) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    hdr_t h;

    const double t0 = now_s();
    float * W = NULL;
    if (decode_all(buf, n, &h, &W)) { free(buf); return 1; }
    const double t_decode = now_s() - t0;

    size_t vn = 0;
    uint8_t * vb = slurp(vecpath, &vn);
    if (!vb) { fprintf(stderr, "cannot read %s\n", vecpath); free(W); free(buf); return 1; }
    if (vn != (size_t) h.cols * 4) {
        fprintf(stderr, "%s holds %zu bytes, expected %zu (%u floats)\n",
                vecpath, vn, (size_t) h.cols * 4, h.cols);
        free(vb); free(W); free(buf);
        return 1;
    }
    const float * x = (const float *) vb;
    float * y = malloc((size_t) h.rows * 4);
    if (!y) { free(vb); free(W); free(buf); return 1; }

    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        const double t = now_s();
        for (uint32_t r = 0; r < h.rows; r++) {
            const float * w = W + (size_t) r * h.cols;
            float acc = 0.0f;
            for (uint32_t c = 0; c < h.cols; c++) acc += w[c] * x[c];
            y[r] = acc;
        }
        const double dt = now_s() - t;
        if (dt < best) best = dt;
    }

    double sum = 0, nrm = 0;
    for (uint32_t r = 0; r < h.rows; r++) { sum += y[r]; nrm += (double) y[r] * y[r]; }

    const double wbytes = (double) n;
    printf("file        %s (%u x %u, %u stages)\n", path, h.rows, h.cols, h.stages);
    printf("decode      %.3f ms, %.2f GB/s of container\n",
           t_decode * 1e3, wbytes / t_decode / 1e9);
    printf("matvec      %.3f ms best of %d, %.2f GB/s of weights\n",
           best * 1e3, iters, ((double) h.rows * h.cols * 4) / best / 1e9);
    printf("checksum    sum %.6f, norm %.6f\n", sum, sqrt(nrm));

    free(y); free(vb); free(W); free(buf);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s info <file.bin>\n"
            "       %s check <file.bin> <hf-dir> <tensor-name>\n"
            "       %s dump <file.bin> <out.f32>\n"
            "       %s matvec <file.bin> <vec.f32> [iters]\n",
            argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "info")) return cmd_info(argv[2]);
    if (!strcmp(argv[1], "check") && argc >= 5) return cmd_check(argv[2], argv[3], argv[4]);
    if (!strcmp(argv[1], "dump") && argc >= 4) return cmd_dump(argv[2], argv[3]);
    if (!strcmp(argv[1], "matvec") && argc >= 4) {
        const int iters = argc >= 5 ? atoi(argv[4]) : 20;
        return cmd_matvec(argv[2], argv[3], iters > 0 ? iters : 1);
    }
    fprintf(stderr, "unknown command or missing arguments\n");
    return 2;
}
