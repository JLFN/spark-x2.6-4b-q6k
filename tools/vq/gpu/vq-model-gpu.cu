/* vq-model-gpu.cu — the whole Spark2_5 forward pass on the GPU over an RQVQ
 * artifact. This is the model-level counterpart of tools/vq-runtime, which is
 * the CPU f32 reference.
 *
 * It is a second implementation, and the CPU reference is its acceptance
 * test: the same ids must pick the same argmax and the two logit vectors must
 * agree within kernel precision (compare the two --logits dumps with
 * bin/logits-diff). Correctness evidence, not the assertion, is what makes
 * this a runtime.
 *
 * Every matrix is served by the fused DQV3 block GEMV from decode-gpu.cu
 * (vqg_gemv_launch) one stage at a time, accumulated into the destination, so
 * no weight is ever expanded: a token streams the 3.2 GB container and
 * nothing else. The tied embedding is a container too: a token's embedding
 * row is rebuilt by vqg_row_dequant, and the logits are one more GEMV against
 * the same matrix.
 *
 * The arithmetic of the non-matrix work follows tools/vq-runtime/src/model.rs
 * line for line:
 *   - rms_norm accumulates the sum of squares in f64 (model.rs:211-219);
 *   - rope is the interleaved-half rotation at model.rs:316-341, the table
 *     computed per position per layer with that layer type's theta and
 *     partial factor (config.rs:95-101);
 *   - the causal start is 0 for full layers and pos+1-window for sliding ones
 *     (model.rs:397-400);
 *   - GQA maps head h to kv head h*kv_heads/heads (model.rs:410-435);
 *   - the per-head sigmoid gate is applied to the head output before
 *     out_proj (model.rs:436-438);
 *   - the MLP is the exact erf gelu, not the tanh form (model.rs:227-233);
 *   - the residual adds the projection output in place (model.rs:447-449,
 *     458-461), which the accumulating GEMV does directly.
 *
 * Named deviation: the CPU reference holds the residual stream in f32 and so
 * does this runtime, but the GEMV sums in a different order than the CPU's
 * 8-way dot, and rope uses double-precision cos/sin before the f32 rotation.
 * Both are kernel-precision differences; the gate measures them instead of
 * hiding them.
 *
 * Usage:
 *   vq-model-gpu --dir models/vq-build-8192-2-v3 \
 *                --ids 1046,11862,314,17349,390 [--gen 8] [--ctx 512] \
 *                [--config models/hf/config.json] [--top 5] [--logits FILE]
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "vq-profile.h"

extern "C" {
const char * vqg_decode_error(void);
int vqg_gemv_launch(const void * payload, int dim, int nc, int rows, int cols, int v3,
                    const float * x, float * y, int variant);
int vqg_gemv_launch_t(const void * payload, int dim, int nc, int rows, int cols, int v3,
                      const float * x, float * y, int variant, int threads);
int vqg_gemv_ready(int dim, int nc, int rows, int cols, int v3, int variant);
int vqg_row_dequant(const void * payload, int dim, int nc, int rows, int cols, int v3,
                    int row, float * out);
int vqg_prefill_launch(const void * payload, int dim, int nc, int rows, int cols, int v3,
                       const float * xT, int ntok, float * y, int variant);
int vqg_prefill_launch_tile(const void * payload, int dim, int nc, int rows, int cols, int v3,
                            const float * xT, int ntok, float * y,
                            int bn, int bk, int variant);
int vqg_prefill_ready(int dim, int nc, int rows, int cols, int v3, int variant);
int vqg_prefill_cfg_ready(int dim, int nc, int rows, int cols, int v3, int bn, int bk, int variant);
}

/* The tuning profile, when one matched this artifact and device. A zero
 * g_prof_on keeps the readiness heuristics above (see vq-profile.h). */
static vqprof g_prof;
static int g_prof_on = 0;

/* ---------------------------------------------------------------- config -- */

typedef struct {
    int hidden, layers, heads, kv_heads, head_dim, inter, vocab;
    float eps;
    int window;
    int *full;                 /* 1 = full attention layer, 0 = sliding */
    double rope_full_theta, rope_full_partial;
    double rope_slide_theta, rope_slide_partial;
    int q_dim, kv_dim;
} Cfg;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

static int ck(cudaError_t e, const char * what) {
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return -1;
}

static char * slurp(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    char * buf = (char *) calloc((size_t) sz + 1, 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t) sz, f) != (size_t) sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    if (n) *n = (size_t) sz;
    return buf;
}

/* Exact quoted-key lookup; a prefix match would confuse hidden_size with any
 * future hidden_size_* key. */
static const char * key_at(const char * s, const char * key) {
    char pat[128];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    return strstr(s, pat);
}

static int json_num(const char * s, const char * key, double * out) {
    const char * p = key_at(s, key);
    if (!p) return -1;
    p = strchr(p, ':');
    if (!p) return -1;
    *out = strtod(p + 1, NULL);
    return 0;
}

static int json_num_from(const char * s, const char * key, double * out) {
    const char * p = key_at(s, key);
    if (!p) return -1;
    p = strchr(p, ':');
    if (!p) return -1;
    *out = strtod(p + 1, NULL);
    return 0;
}

static int parse_config(const char * path, Cfg * c) {
    size_t n = 0;
    char * text = slurp(path, &n);
    if (!text) { fprintf(stderr, "%s: cannot read\n", path); return -1; }
    double v;
    memset(c, 0, sizeof *c);
    int bad = 0;
    #define NEED(k, dst) do { if (json_num(text, k, &v)) { fprintf(stderr, "%s: no %s\n", path, k); bad = 1; } else dst = (int) v; } while (0)
    NEED("hidden_size", c->hidden);
    NEED("num_hidden_layers", c->layers);
    NEED("num_attention_heads", c->heads);
    NEED("num_key_value_heads", c->kv_heads);
    NEED("head_dim", c->head_dim);
    NEED("intermediate_size", c->inter);
    NEED("vocab_size", c->vocab);
    NEED("sliding_window", c->window);
    #undef NEED
    if (json_num(text, "rms_norm_eps", &v)) bad = 1; else c->eps = (float) v;
    if (bad) { free(text); return -1; }

    c->q_dim = c->heads * c->head_dim;
    c->kv_dim = c->kv_heads * c->head_dim;
    c->full = (int *) calloc((size_t) c->layers, sizeof(int));
    if (!c->full) { free(text); return -1; }

    /* layer_types is the list that decides the mask and the rope; a config
     * without it cannot be run, because guessing the mask would silently
     * change the model. */
    const char * p = key_at(text, "layer_types");
    if (!p || !(p = strchr(p, '['))) { fprintf(stderr, "%s: no layer_types\n", path); free(text); return -1; }
    int i = 0;
    while (*p && *p != ']') {
        if (*p == '"') {
            const char * e = strchr(p + 1, '"');
            if (!e) break;
            if (i < c->layers)
                c->full[i] = (e - (p + 1) == 14 && !strncmp(p + 1, "full_attention", 14));
            i++;
            p = e + 1;
        } else {
            p++;
        }
    }
    if (i != c->layers) {
        fprintf(stderr, "%s: layer_types has %d entries, the model has %d layers\n", path, i, c->layers);
        free(text); return -1;
    }

    /* rope_parameters holds one block per layer type. A missing block falls
     * back to the values this checkpoint ships, exactly as config.rs:54-69
     * does, so an older config still runs and the fallback is stated here. */
    const char * rp = key_at(text, "rope_parameters");
    c->rope_full_theta = 5000000.0;
    c->rope_full_partial = 0.25;
    c->rope_slide_theta = 10000.0;
    c->rope_slide_partial = 1.0;
    if (rp) {
        const char * f = strstr(rp, "\"full_attention\"");
        const char * s = strstr(rp, "\"sliding_attention\"");
        if (f) {
            if (!json_num_from(f, "rope_theta", &c->rope_full_theta)) { }
            if (!json_num_from(f, "partial_rotary_factor", &c->rope_full_partial)) { }
        }
        if (s) {
            if (!json_num_from(s, "rope_theta", &c->rope_slide_theta)) { }
            if (!json_num_from(s, "partial_rotary_factor", &c->rope_slide_partial)) { }
        }
    }
    free(text);
    return 0;
}

/* --------------------------------------------------------------- weights -- */

typedef struct {
    int dim, nc, rows, cols, v3, nst, variant, pvariant;
    int threads;               /* GEMV launch width, from the profile or 256 */
    int pf_bn, pf_bk;          /* prefill tile, from the profile or 128x64 */
    unsigned char * st[4];
} Mat;

typedef struct {
    Mat qkv, attn_gate, out, gate, up, down;
    float * in_norm, * post_norm;
} Layer;

static size_t g_container_bytes = 0;

static int load_mat(const char * dir, const char * name, int want_rows, int want_cols, Mat * m) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    size_t n = 0;
    unsigned char * buf = (unsigned char *) slurp(path, &n);
    if (!buf) { fprintf(stderr, "%s: cannot read (the manifest must list it)\n", path); return -1; }
    if (n < 32 || memcmp(buf, "RQVQ", 4)) {
        fprintf(stderr, "%s: not an RQVQ container\n", path);
        free(buf);
        return -1;
    }
    uint16_t dim, nc, nst;
    uint32_t rows, cols;
    uint64_t payload;
    memcpy(&dim, buf + 4, 2);
    memcpy(&nc, buf + 6, 2);
    memcpy(&nst, buf + 8, 2);
    memcpy(&rows, buf + 12, 4);
    memcpy(&cols, buf + 16, 4);
    memcpy(&payload, buf + 24, 8);
    if ((int) rows != want_rows || (int) cols != want_cols) {
        fprintf(stderr, "%s: is %ux%u, the architecture needs %dx%d\n", path, rows, cols, want_rows, want_cols);
        free(buf);
        return -1;
    }
    if (nst < 1 || nst > 4 || payload % nst || n != 32 + payload) {
        fprintf(stderr, "%s: header and file size disagree\n", path);
        free(buf);
        return -1;
    }
    const size_t per = (size_t) (payload / nst);
    const int v3 = per >= 4 && !memcmp(buf + 32, "DQV3", 4);
    for (int s = 1; s < nst; s++) {
        const int sv3 = per >= 4 && !memcmp(buf + 32 + per * (size_t) s, "DQV3", 4);
        if (sv3 != v3) {
            fprintf(stderr, "%s: stage %d is not the same payload format\n", path, s);
            free(buf);
            return -1;
        }
    }
    memset(m, 0, sizeof *m);
    m->dim = dim;
    m->nc = nc;
    m->rows = (int) rows;
    m->cols = (int) cols;
    m->v3 = v3;
    m->nst = nst;
    for (int s = 0; s < nst; s++) {
        /* 8 bytes of slack: the kernels read aligned 4- and 8-byte index
         * windows, and the last index of the last row can reach up to 7 bytes
         * past the payload (see rd_idx_fmt in decode-gpu.cu). The C driver
         * carries the same slack; without it compute-sanitizer reports an
         * out-of-bounds read even though the driver allocator rounds up. */
        if (ck(cudaMalloc((void **) &m->st[s], per + 8), "cudaMalloc stage")) { free(buf); return -1; }
        if (ck(cudaMemcpy(m->st[s], buf + 32 + per * (size_t) s, per, cudaMemcpyHostToDevice), "stage upload")) {
            free(buf);
            return -1;
        }
        if (ck(cudaMemset(m->st[s] + per, 0, 8), "pad stage")) { free(buf); return -1; }
    }
    m->variant = vqg_gemv_ready(dim, nc, (int) rows, (int) cols, v3, 3) ? 3
                : vqg_gemv_ready(dim, nc, (int) rows, (int) cols, v3, 2) ? 2 : 1;
    /* The prefill path has its own geometry rules (a DQV3 12-bit stream and
     * whole 64-column rounds), so it is chosen independently of the GEMV
     * variant; 3 means the legacy batched kernel, which the model runtime
     * treats as "token by token is better for this matrix". */
    m->pvariant = vqg_prefill_ready(dim, nc, (int) rows, (int) cols, v3, 2) ? 2
                : vqg_prefill_ready(dim, nc, (int) rows, (int) cols, v3, 1) ? 1 : 3;
    m->threads = 256;
    m->pf_bn = 128;
    m->pf_bk = 64;
    /* A matched profile replaces the heuristic per geometry. Both entries are
     * re-checked for runnability here, so a hand-edited or stale profile can
     * lose to the heuristic but can never launch an illegal kernel. */
    if (g_prof_on) {
        const vqprof_entry * g = vqprof_find(&g_prof, 0, dim, nc, (int) rows, (int) cols, v3);
        if (g && (g->variant < 0 || g->variant > 3
                  || !vqg_gemv_ready(dim, nc, (int) rows, (int) cols, v3, g->variant)
                  || g->threads < 32 || g->threads > 1024 || g->threads % 32)) {
            fprintf(stderr, "profile: gemv entry for %dx%d is not runnable here; heuristic kept\n",
                    (int) rows, (int) cols);
        } else if (g) {
            m->variant = g->variant;
            m->threads = g->threads;
        }
        const vqprof_entry * p = vqprof_find(&g_prof, 1, dim, nc, (int) rows, (int) cols, v3);
        if (p && !vqg_prefill_cfg_ready(dim, nc, (int) rows, (int) cols, v3, p->bn, p->bk, p->variant)) {
            fprintf(stderr, "profile: prefill entry for %dx%d is not runnable here; heuristic kept\n",
                    (int) rows, (int) cols);
        } else if (p) {
            m->pvariant = p->variant;
            m->pf_bn = p->bn;
            m->pf_bk = p->bk;
        }
    }
    g_container_bytes += payload;
    free(buf);
    return 0;
}

static float * load_vec(const char * dir, const char * name, int want) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    size_t n = 0;
    unsigned char * buf = (unsigned char *) slurp(path, &n);
    if (!buf) { fprintf(stderr, "%s: cannot read\n", path); return NULL; }
    uint32_t rows, cols;
    if (n < 12 || memcmp(buf, "RQVF", 4)) {
        fprintf(stderr, "%s: not an RQVF sidecar\n", path);
        free(buf);
        return NULL;
    }
    memcpy(&rows, buf + 4, 4);
    memcpy(&cols, buf + 8, 4);
    if ((int) rows * (int) cols != want || n != 12 + (size_t) want * 4) {
        fprintf(stderr, "%s: is %ux%u, the architecture needs %d values\n", path, rows, cols, want);
        free(buf);
        return NULL;
    }
    float * d = NULL;
    if (ck(cudaMalloc((void **) &d, (size_t) want * 4), "cudaMalloc vector") ||
        ck(cudaMemcpy(d, buf + 12, (size_t) want * 4, cudaMemcpyHostToDevice), "vector upload")) {
        free(buf);
        return NULL;
    }
    free(buf);
    return d;
}

/* ---------------------------------------------------------------- kernels -- */

/* The CPU norm sums in f64 (model.rs:211-219); so does this one, because the
 * squares of 2560 f32 values lose bits in an f32 tree and the norm is applied
 * 36 times per token. */
__global__ void rmsnorm_kernel(const float * x, const float * w, float * out, int n, float eps) {
    __shared__ double ss[256];
    const int t = threadIdx.x;
    double s = 0.0;
    for (int i = t; i < n; i += blockDim.x) s += (double) x[i] * (double) x[i];
    ss[t] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) ss[t] += ss[t + off];
        __syncthreads();
    }
    const float inv = (float) (1.0 / sqrt(ss[0] / (double) n + (double) eps));
    for (int i = t; i < n; i += blockDim.x) out[i] = x[i] * inv * w[i];
}

/* Interleaved-half rotation over the first `pdim` elements of every q and k
 * head (model.rs:316-341). One thread per (head, i) pair in the first half. */
__global__ void rope_kernel(float * qkv, int qd, int heads, int kv_heads, int hd,
                            int pdim, double theta, int pos) {
    const int half = pdim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (heads + kv_heads) * half) return;
    const int h = idx / half, i = idx % half;
    const double freq = 1.0 / pow(theta, 2.0 * (double) i / (double) pdim);
    const double a = (double) pos * freq;
    const float c = (float) cos(a), s = (float) sin(a);
    float * base = (h < heads) ? qkv + (size_t) h * hd : qkv + qd + (size_t) (h - heads) * hd;
    const float x0 = base[i], x1 = base[i + half];
    base[i] = x0 * c - x1 * s;
    base[i + half] = x1 * c + x0 * s;
}

__global__ void kv_store_kernel(const float * qkv, int qd, int kvd,
                                float * kc, float * vc, int pos) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kvd; i += blockDim.x * gridDim.x) {
        kc[(size_t) pos * kvd + i] = qkv[qd + i];
        vc[(size_t) pos * kvd + i] = qkv[qd + kvd + i];
    }
}

/* One block per head. Phase 1 scores every cached position (a warp-strided
 * dot per position), phase 2 each thread owns one head element and walks the
 * softmax weights. The sigmoid gate is applied on the way out
 * (model.rs:410-438). Shared memory is n scores plus one reduction row. */
__global__ void attn_kernel(const float * qkv, int qd, int kvd, const float * gate,
                            const float * kc, const float * vc, int start, int n,
                            int heads, int kv_heads, int hd, float * out) {
    extern __shared__ float sh[];
    float * sc = sh;
    float * red = sh + n;
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int kvh = h * kv_heads / heads;
    const float * q = qkv + (size_t) h * hd;
    const float * kb = kc + (size_t) start * kvd + (size_t) kvh * hd;
    const float scale = 1.0f / sqrtf((float) hd);

    for (int j = t; j < n; j += blockDim.x) {
        const float * k = kb + (size_t) j * kvd;
        float s = 0.0f;
        for (int d = 0; d < hd; d++) s += q[d] * k[d];
        sc[j] = s * scale;
    }
    __syncthreads();

    float mx = -INFINITY;
    for (int j = t; j < n; j += blockDim.x) mx = fmaxf(mx, sc[j]);
    if (t < 256) red[t] = mx;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off && t < 256) red[t] = fmaxf(red[t], red[t + off]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();

    float sm = 0.0f;
    for (int j = t; j < n; j += blockDim.x) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sm += e;
    }
    if (t < 256) red[t] = sm;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off && t < 256) red[t] += red[t + off];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    const float g = 1.0f / (1.0f + expf(-gate[h]));
    const float * vb = vc + (size_t) start * kvd + (size_t) kvh * hd;

    for (int d = t; d < hd; d += blockDim.x) {
        float o = 0.0f;
        for (int j = 0; j < n; j++) o += sc[j] * inv * vb[(size_t) j * kvd + d];
        out[(size_t) h * hd + d] = o * g;
    }
}

/* The MLP activation: exact erf gelu times the up projection (model.rs:456-457). */
__global__ void gelu_mul_kernel(const float * g, const float * u, float * act, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float x = g[i];
        act[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)) * u[i];
    }
}

__global__ void argmax_kernel(const float * v, int n, int * out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const int t = threadIdx.x;
    float best = -INFINITY;
    int b = 0;
    for (int i = t; i < n; i += blockDim.x) {
        if (v[i] > best) { best = v[i]; b = i; }
    }
    bv[t] = best;
    bi[t] = b;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off && bv[t + off] > bv[t]) { bv[t] = bv[t + off]; bi[t] = bi[t + off]; }
        __syncthreads();
    }
    if (t == 0) *out = bi[0];
}

/* ------- batched prompt kernels: the same arithmetic, one block per token --- */

/* The same f64-accumulating norm as rmsnorm_kernel, one block per token, so
 * the prompt path does not launch 36*2*ntok single-token norms. */
__global__ void rmsnorm_batch_kernel(const float * x, const float * w, float * out,
                                     int n, float eps) {
    __shared__ double ss[256];
    const int t = threadIdx.x;
    const float * xr = x + (size_t) blockIdx.x * n;
    float * orow = out + (size_t) blockIdx.x * n;
    double s = 0.0;
    for (int i = t; i < n; i += blockDim.x) s += (double) xr[i] * (double) xr[i];
    ss[t] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) ss[t] += ss[t + off];
        __syncthreads();
    }
    const float inv = (float) (1.0 / sqrt(ss[0] / (double) n + (double) eps));
    for (int i = t; i < n; i += blockDim.x) orow[i] = xr[i] * inv * w[i];
}

/* Token-major [rows][cols] -> column-major [cols][rows], which is the shape
 * the tiled prefill reads. 32x32 tiles keep both sides coalesced. */
__global__ void transpose_f32_kernel(const float * in, float * out, int rows, int cols) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[(size_t) y * cols + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < rows && y < cols) out[(size_t) y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

/* rope over every token of the chunk; each token's own absolute position
 * feeds the same interleaved-half rotation as rope_kernel. */
__global__ void rope_batch_kernel(float * qkv, int qkv_w, int qd, int heads, int kv_heads,
                                  int hd, int pdim, double theta, int t0) {
    const int half = pdim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (heads + kv_heads) * half) return;
    const int h = idx / half, i = idx % half;
    const int pos = t0 + (int) blockIdx.y;
    const double freq = 1.0 / pow(theta, 2.0 * (double) i / (double) pdim);
    const double a = (double) pos * freq;
    const float c = (float) cos(a), s = (float) sin(a);
    float * row = qkv + (size_t) blockIdx.y * qkv_w;
    float * base = (h < heads) ? row + (size_t) h * hd : row + qd + (size_t) (h - heads) * hd;
    const float x0 = base[i], x1 = base[i + half];
    base[i] = x0 * c - x1 * s;
    base[i + half] = x1 * c + x0 * s;
}

__global__ void kv_store_batch_kernel(const float * qkv, int qkv_w, int qd, int kvd,
                                      float * kc, float * vc, int t0) {
    const int pos = t0 + (int) blockIdx.y;
    const float * row = qkv + (size_t) blockIdx.y * qkv_w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kvd; i += blockDim.x * gridDim.x) {
        kc[(size_t) pos * kvd + i] = row[qd + i];
        vc[(size_t) pos * kvd + i] = row[qd + kvd + i];
    }
}

/* One block per (head, token). Each block answers its own token's causal
 * range, so the whole chunk's queries run in one launch; the cache already
 * holds every key and value the block can see (all kv stores precede this
 * launch). Shared layout: 256 reduction floats, then the score row. */
__global__ void attn_batch_kernel(const float * qkv, int qkv_w, int qd, int kvd,
                                  const float * gate, const float * kc, const float * vc,
                                  int t0, int window, int full, int heads, int kv_heads,
                                  int hd, float * out) {
    extern __shared__ float sh[];
    float * red = sh;
    float * sc = sh + 256;
    const int h = blockIdx.x;
    const int tok = blockIdx.y;
    const int t = threadIdx.x;
    const int pos = t0 + tok;
    const int start = full ? 0 : (pos + 1 > window ? pos + 1 - window : 0);
    const int n = pos + 1 - start;
    const int kvh = h * kv_heads / heads;
    const float * q = qkv + (size_t) tok * qkv_w + (size_t) h * hd;
    const float * kb = kc + (size_t) start * kvd + (size_t) kvh * hd;
    const float scale = 1.0f / sqrtf((float) hd);

    for (int j = t; j < n; j += blockDim.x) {
        const float * k = kb + (size_t) j * kvd;
        float s = 0.0f;
        for (int d = 0; d < hd; d++) s += q[d] * k[d];
        sc[j] = s * scale;
    }
    __syncthreads();

    float mx = -INFINITY;
    for (int j = t; j < n; j += blockDim.x) mx = fmaxf(mx, sc[j]);
    if (t < 256) red[t] = mx;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off && t < 256) red[t] = fmaxf(red[t], red[t + off]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();

    float sm = 0.0f;
    for (int j = t; j < n; j += blockDim.x) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sm += e;
    }
    if (t < 256) red[t] = sm;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off && t < 256) red[t] += red[t + off];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    const float g = 1.0f / (1.0f + expf(-gate[(size_t) tok * heads + h]));
    const float * vb = vc + (size_t) start * kvd + (size_t) kvh * hd;

    for (int d = t; d < hd; d += blockDim.x) {
        float o = 0.0f;
        for (int j = 0; j < n; j++) o += sc[j] * inv * vb[(size_t) j * kvd + d];
        out[(size_t) tok * qd + (size_t) h * hd + d] = o * g;
    }
}

/* ----------------------------------------------------------------- model -- */

typedef struct {
    Cfg c;
    Mat embed;
    float * d_norm;
    Layer * layers;
    float * kv_k, * kv_v;
    int ctx;
    float * h, * xn, * qkv, * g16, * attn, * mlp_g, * mlp_u, * act, * logits;
    /* Prompt chunk buffers, sized by m->chunk; hb is [chunk][hidden],
     * xT is [hidden][chunk], and every other batch buffer is [chunk][rows]. */
    int chunk, prefill_ok;
    float * hb, * xb, * xT, * qkv_b, * g_b, * attn_b, * attn_t, * mlp_gb, * mlp_ub, * act_b, * act_t;
    /* Layer-input capture for a draft. cap_slot[li] is the destination slot
     * for target layer li, or -1 when that layer is not captured. cap is
     * [cap_n][ctx][hidden], position-major, written by forward_batch before
     * the layer runs (the same row vq-runtime captures). */
    int cap_n;
    int cap_layer[8];
    int * cap_slot;
    float * cap;
} Model;

static float * dev_f32(size_t n) {
    void * p = NULL;
    if (cudaMalloc(&p, n * 4) != cudaSuccess) return NULL;
    return (float *) p;
}

/* One matrix times one vector, every stage accumulated into y. The caller
 * zeroes y when the result must not add to what is there; the residual path
 * deliberately adds into h. */
static int gemv(const Mat * m, const float * x, float * y) {
    for (int s = 0; s < m->nst; s++) {
        if (vqg_gemv_launch_t(m->st[s], m->dim, m->nc, m->rows, m->cols, m->v3, x, y,
                              m->variant, m->threads)) {
            fprintf(stderr, "gemv: %s\n", vqg_decode_error());
            return -1;
        }
    }
    return 0;
}

static int alloc_model(Model * m, const Cfg * c, int ctx, int chunk) {
    m->c = *c;
    m->ctx = ctx;
    m->chunk = chunk;
    m->layers = (Layer *) calloc((size_t) c->layers, sizeof(Layer));
    if (!m->layers) return -1;
    /* The KV cache is position-major: [layer][pos][kv_dim], f32, both k and v. */
    const size_t kv = (size_t) c->layers * (size_t) ctx * (size_t) c->kv_dim;
    m->kv_k = dev_f32(kv);
    m->kv_v = dev_f32(kv);
    m->h = dev_f32((size_t) c->hidden);
    m->xn = dev_f32((size_t) c->hidden);
    m->qkv = dev_f32((size_t) (c->q_dim + 2 * c->kv_dim));
    m->g16 = dev_f32((size_t) c->heads);
    m->attn = dev_f32((size_t) c->q_dim);
    m->mlp_g = dev_f32((size_t) c->inter);
    m->mlp_u = dev_f32((size_t) c->inter);
    m->act = dev_f32((size_t) c->inter);
    m->logits = dev_f32((size_t) c->vocab);
    const size_t nt = (size_t) chunk;
    m->hb = dev_f32(nt * (size_t) c->hidden);
    m->xb = dev_f32(nt * (size_t) c->hidden);
    m->xT = dev_f32(nt * (size_t) c->hidden);
    m->qkv_b = dev_f32(nt * (size_t) (c->q_dim + 2 * c->kv_dim));
    m->g_b = dev_f32(nt * (size_t) c->heads);
    m->attn_b = dev_f32(nt * (size_t) c->q_dim);
    m->attn_t = dev_f32(nt * (size_t) c->q_dim);
    m->mlp_gb = dev_f32(nt * (size_t) c->inter);
    m->mlp_ub = dev_f32(nt * (size_t) c->inter);
    m->act_b = dev_f32(nt * (size_t) c->inter);
    m->act_t = dev_f32(nt * (size_t) c->inter);
    m->cap_n = 0;
    m->cap = NULL;
    m->cap_slot = (int *) malloc((size_t) c->layers * sizeof(int));
    if (m->cap_slot)
        for (int li = 0; li < c->layers; li++) m->cap_slot[li] = -1;
    return (m->kv_k && m->kv_v && m->h && m->xn && m->qkv && m->g16 && m->attn &&
            m->mlp_g && m->mlp_u && m->act && m->logits && m->hb && m->xb && m->xT &&
            m->qkv_b && m->g_b && m->attn_b && m->attn_t && m->mlp_gb && m->mlp_ub &&
            m->act_b && m->act_t && m->cap_slot) ? 0 : -1;
}

/* Turn on layer-input capture for the target layers a draft names. The buffer
 * is position-major per slot, so a later round can read the committed rows
 * without a host copy. */
static int enable_capture(Model * m, const int * layers, int n) {
    if (n <= 0 || n > 8) {
        fprintf(stderr, "capture: %d layers requested, the limit is 8\n", n);
        return -1;
    }
    if (m->cap) {
        fprintf(stderr, "capture: already enabled\n");
        return -1;
    }
    for (int k = 0; k < n; k++) {
        const int li = layers[k];
        if (li < 0 || li >= m->c.layers) {
            fprintf(stderr, "capture: layer %d is outside 0..%d\n", li, m->c.layers);
            return -1;
        }
        if (m->cap_slot[li] >= 0) {
            fprintf(stderr, "capture: layer %d is listed twice\n", li);
            return -1;
        }
        m->cap_slot[li] = k;
        m->cap_layer[k] = li;
    }
    m->cap_n = n;
    m->cap = dev_f32((size_t) n * (size_t) m->ctx * (size_t) m->c.hidden);
    if (!m->cap) {
        fprintf(stderr, "capture: cannot allocate %d x %d x %d floats\n",
                n, m->ctx, m->c.hidden);
        return -1;
    }
    return 0;
}

static int load_model(Model * m, const char * dir) {
    const Cfg * c = &m->c;
    m->prefill_ok = 1;
    if (load_mat(dir, "model.embedding.weight", c->vocab, c->hidden, &m->embed)) return -1;
    m->d_norm = load_vec(dir, "model.norm.weight", c->hidden);
    if (!m->d_norm) return -1;
    for (int li = 0; li < c->layers; li++) {
        Layer * L = &m->layers[li];
        char name[256];
        #define MAT(field, nm, r, cc) do { \
            snprintf(name, sizeof name, "model.layers.%d." nm, li); \
            if (load_mat(dir, name, r, cc, &L->field)) return -1; } while (0)
        MAT(qkv, "self_attn.q_k_v_proj.weight", c->q_dim + 2 * c->kv_dim, c->hidden);
        MAT(attn_gate, "self_attn.g_proj.weight", c->heads, c->hidden);
        MAT(out, "self_attn.out_proj.weight", c->hidden, c->q_dim);
        MAT(gate, "mlp.gate_proj.weight", c->inter, c->hidden);
        MAT(up, "mlp.up_proj.weight", c->inter, c->hidden);
        MAT(down, "mlp.down_proj.weight", c->hidden, c->inter);
        #undef MAT
        snprintf(name, sizeof name, "model.layers.%d.input_layernorm.weight", li);
        if (!(L->in_norm = load_vec(dir, name, c->hidden))) return -1;
        snprintf(name, sizeof name, "model.layers.%d.post_attention_layernorm.weight", li);
        if (!(L->post_norm = load_vec(dir, name, c->hidden))) return -1;
        /* One matrix whose geometry the tiled kernel cannot take makes the
         * whole prompt fall back to the token-by-token path: the legacy
         * batched kernel is slower than decode for such a matrix, and mixing
         * it with tiled stages would only be a partial win. */
        if (L->qkv.pvariant > 2 || L->attn_gate.pvariant > 2 || L->out.pvariant > 2 ||
            L->gate.pvariant > 2 || L->up.pvariant > 2 || L->down.pvariant > 2)
            m->prefill_ok = 0;
    }
    return 0;
}

/* The forward pass for one token at `pos`: embedding row, 36 layers,
 * final norm, tied-head logits. Every launch is asynchronous; the caller
 * synchronizes once. */
static int forward(Model * m, uint32_t token, int pos) {
    const Cfg * c = &m->c;
    if (pos >= m->ctx) { fprintf(stderr, "position %d exceeds the %d-token cache\n", pos, m->ctx); return -1; }
    if ((int) token >= c->vocab) { fprintf(stderr, "token %u is outside the vocabulary\n", token); return -1; }

    if (ck(cudaMemsetAsync(m->h, 0, (size_t) c->hidden * 4), "memset h")) return -1;
    for (int s = 0; s < m->embed.nst; s++) {
        if (vqg_row_dequant(m->embed.st[s], m->embed.dim, m->embed.nc, m->embed.rows,
                            m->embed.cols, m->embed.v3, (int) token, m->h)) {
            fprintf(stderr, "embedding row: %s\n", vqg_decode_error());
            return -1;
        }
    }

    const int qkv_w = c->q_dim + 2 * c->kv_dim;
    for (int li = 0; li < c->layers; li++) {
        Layer * L = &m->layers[li];
        const int full = c->full[li];

        rmsnorm_kernel<<<1, 256>>>(m->h, L->in_norm, m->xn, c->hidden, c->eps);
        if (ck(cudaMemsetAsync(m->qkv, 0, (size_t) qkv_w * 4), "memset qkv")) return -1;
        if (gemv(&L->qkv, m->xn, m->qkv)) return -1;
        if (ck(cudaMemsetAsync(m->g16, 0, (size_t) c->heads * 4), "memset gate")) return -1;
        if (gemv(&L->attn_gate, m->xn, m->g16)) return -1;

        const double theta = full ? c->rope_full_theta : c->rope_slide_theta;
        const double partial = full ? c->rope_full_partial : c->rope_slide_partial;
        int pdim = (int) ((double) c->head_dim * partial);
        if (pdim > c->head_dim) pdim = c->head_dim;
        if (pdim < 2) pdim = 2;
        const int rot = (c->heads + c->kv_heads) * (pdim / 2);
        rope_kernel<<<(rot + 127) / 128, 128>>>(m->qkv, c->q_dim, c->heads, c->kv_heads,
                                                c->head_dim, pdim, theta, pos);

        const size_t kv_off = (size_t) li * (size_t) m->ctx * (size_t) c->kv_dim;
        kv_store_kernel<<<(c->kv_dim + 255) / 256, 256>>>(m->qkv, c->q_dim, c->kv_dim,
                                                          m->kv_k + kv_off, m->kv_v + kv_off, pos);

        const int start = full ? 0 : (pos + 1 > c->window ? pos + 1 - c->window : 0);
        const int n = pos + 1 - start;
        const size_t sh = ((size_t) n + 256) * sizeof(float);
        attn_kernel<<<c->heads, 256, sh>>>(m->qkv, c->q_dim, c->kv_dim, m->g16,
                                           m->kv_k + kv_off, m->kv_v + kv_off, start, n,
                                           c->heads, c->kv_heads, c->head_dim, m->attn);
        if (gemv(&L->out, m->attn, m->h)) return -1;

        rmsnorm_kernel<<<1, 256>>>(m->h, L->post_norm, m->xn, c->hidden, c->eps);
        if (ck(cudaMemsetAsync(m->mlp_g, 0, (size_t) c->inter * 4), "memset mlp gate")) return -1;
        if (ck(cudaMemsetAsync(m->mlp_u, 0, (size_t) c->inter * 4), "memset mlp up")) return -1;
        if (gemv(&L->gate, m->xn, m->mlp_g)) return -1;
        if (gemv(&L->up, m->xn, m->mlp_u)) return -1;
        gelu_mul_kernel<<<(c->inter + 255) / 256, 256>>>(m->mlp_g, m->mlp_u, m->act, c->inter);
        if (gemv(&L->down, m->act, m->h)) return -1;
    }

    rmsnorm_kernel<<<1, 256>>>(m->h, m->d_norm, m->xn, c->hidden, c->eps);
    if (ck(cudaMemsetAsync(m->logits, 0, (size_t) c->vocab * 4), "memset logits")) return -1;
    if (gemv(&m->embed, m->xn, m->logits)) return -1;
    return 0;
}

/* One matrix times a chunk of tokens, every stage accumulated into y[ntok][rows]. */
static int prefill_mat(const Mat * m, const float * xT, int ntok, float * y) {
    for (int s = 0; s < m->nst; s++) {
        const int rc = (m->pvariant == 3)
            ? vqg_prefill_launch(m->st[s], m->dim, m->nc, m->rows, m->cols, m->v3,
                                 xT, ntok, y, m->pvariant)
            : vqg_prefill_launch_tile(m->st[s], m->dim, m->nc, m->rows, m->cols, m->v3,
                                      xT, ntok, y, m->pf_bn, m->pf_bk, m->pvariant);
        if (rc) {
            fprintf(stderr, "prefill: %s\n", vqg_decode_error());
            return -1;
        }
    }
    return 0;
}

/* The forward pass for `ntok` prompt tokens starting at position t0. The
 * activations stay token-major ([ntok][cols]) except where the tiled prefill
 * requires [cols][ntok]; the two transposes per layer are the price of using
 * that kernel without changing its layout contract. No logits are produced:
 * the caller runs the final norm and tied head on the last row. */
static int forward_batch(Model * m, const uint32_t * ids, int t0, int ntok) {
    const Cfg * c = &m->c;
    if (ntok <= 0 || ntok > m->chunk) {
        fprintf(stderr, "a chunk of %d tokens does not fit the %d-token buffer\n", ntok, m->chunk);
        return -1;
    }
    if (t0 < 0 || t0 + ntok > m->ctx) {
        fprintf(stderr, "positions %d..%d exceed the %d-token cache\n", t0, t0 + ntok - 1, m->ctx);
        return -1;
    }
    for (int t = 0; t < ntok; t++)
        if ((int) ids[t] >= c->vocab) { fprintf(stderr, "token %u is outside the vocabulary\n", ids[t]); return -1; }

    if (ck(cudaMemsetAsync(m->hb, 0, (size_t) ntok * c->hidden * 4), "memset hb")) return -1;
    for (int t = 0; t < ntok; t++) {
        for (int s = 0; s < m->embed.nst; s++) {
            if (vqg_row_dequant(m->embed.st[s], m->embed.dim, m->embed.nc, m->embed.rows,
                                m->embed.cols, m->embed.v3, (int) ids[t],
                                m->hb + (size_t) t * c->hidden)) {
                fprintf(stderr, "embedding row: %s\n", vqg_decode_error());
                return -1;
            }
        }
    }

    const int qkv_w = c->q_dim + 2 * c->kv_dim;
    for (int li = 0; li < c->layers; li++) {
        Layer * L = &m->layers[li];
        const int full = c->full[li];

        if (m->cap_n && m->cap_slot[li] >= 0) {
            const int slot = m->cap_slot[li];
            if (ck(cudaMemcpy2DAsync(
                       m->cap + ((size_t) slot * m->ctx + (size_t) t0) * c->hidden,
                       (size_t) c->hidden * 4, m->hb, (size_t) c->hidden * 4,
                       (size_t) c->hidden * 4, ntok, cudaMemcpyDeviceToDevice),
                   "capture copy"))
                return -1;
        }

        rmsnorm_batch_kernel<<<ntok, 256>>>(m->hb, L->in_norm, m->xb, c->hidden, c->eps);
        transpose_f32_kernel<<<dim3((c->hidden + 31) / 32, (ntok + 31) / 32), dim3(32, 32)>>>(
            m->xb, m->xT, ntok, c->hidden);
        if (ck(cudaMemsetAsync(m->qkv_b, 0, (size_t) ntok * qkv_w * 4), "memset qkv batch")) return -1;
        if (prefill_mat(&L->qkv, m->xT, ntok, m->qkv_b)) return -1;
        if (ck(cudaMemsetAsync(m->g_b, 0, (size_t) ntok * c->heads * 4), "memset gate batch")) return -1;
        if (prefill_mat(&L->attn_gate, m->xT, ntok, m->g_b)) return -1;

        const double theta = full ? c->rope_full_theta : c->rope_slide_theta;
        const double partial = full ? c->rope_full_partial : c->rope_slide_partial;
        int pdim = (int) ((double) c->head_dim * partial);
        if (pdim > c->head_dim) pdim = c->head_dim;
        if (pdim < 2) pdim = 2;
        const int rot = (c->heads + c->kv_heads) * (pdim / 2);
        rope_batch_kernel<<<dim3((rot + 127) / 128, ntok), 128>>>(
            m->qkv_b, qkv_w, c->q_dim, c->heads, c->kv_heads, c->head_dim, pdim, theta, t0);

        const size_t kv_off = (size_t) li * (size_t) m->ctx * (size_t) c->kv_dim;
        kv_store_batch_kernel<<<dim3((c->kv_dim + 255) / 256, ntok), 256>>>(
            m->qkv_b, qkv_w, c->q_dim, c->kv_dim, m->kv_k + kv_off, m->kv_v + kv_off, t0);

        const int nmax = full ? (t0 + ntok)
                             : ((t0 + ntok) > c->window ? c->window : (t0 + ntok));
        const size_t sh = ((size_t) nmax + 256) * sizeof(float);
        attn_batch_kernel<<<dim3(c->heads, ntok), 256, sh>>>(
            m->qkv_b, qkv_w, c->q_dim, c->kv_dim, m->g_b, m->kv_k + kv_off, m->kv_v + kv_off,
            t0, c->window, full, c->heads, c->kv_heads, c->head_dim, m->attn_b);
        transpose_f32_kernel<<<dim3((c->q_dim + 31) / 32, (ntok + 31) / 32), dim3(32, 32)>>>(
            m->attn_b, m->attn_t, ntok, c->q_dim);
        if (prefill_mat(&L->out, m->attn_t, ntok, m->hb)) return -1;

        rmsnorm_batch_kernel<<<ntok, 256>>>(m->hb, L->post_norm, m->xb, c->hidden, c->eps);
        transpose_f32_kernel<<<dim3((c->hidden + 31) / 32, (ntok + 31) / 32), dim3(32, 32)>>>(
            m->xb, m->xT, ntok, c->hidden);
        if (ck(cudaMemsetAsync(m->mlp_gb, 0, (size_t) ntok * c->inter * 4), "memset mlp gate batch")) return -1;
        if (ck(cudaMemsetAsync(m->mlp_ub, 0, (size_t) ntok * c->inter * 4), "memset mlp up batch")) return -1;
        if (prefill_mat(&L->gate, m->xT, ntok, m->mlp_gb)) return -1;
        if (prefill_mat(&L->up, m->xT, ntok, m->mlp_ub)) return -1;
        const int els = ntok * c->inter;
        gelu_mul_kernel<<<(els + 255) / 256, 256>>>(m->mlp_gb, m->mlp_ub, m->act_b, els);
        transpose_f32_kernel<<<dim3((c->inter + 31) / 32, (ntok + 31) / 32), dim3(32, 32)>>>(
            m->act_b, m->act_t, ntok, c->inter);
        if (prefill_mat(&L->down, m->act_t, ntok, m->hb)) return -1;
    }
    return 0;
}

/* The tail of the prompt: the final norm and the tied head on one row. */
static int tail_logits(Model * m, const float * h_last) {
    rmsnorm_kernel<<<1, 256>>>(h_last, m->d_norm, m->xn, m->c.hidden, m->c.eps);
    if (ck(cudaMemsetAsync(m->logits, 0, (size_t) m->c.vocab * 4), "memset logits")) return -1;
    return gemv(&m->embed, m->xn, m->logits);
}

/* A verification batch: process `ntok` tokens at t0 (capture stays on when a
 * draft enabled it), then produce the tied-head logits for every row, not only
 * the last one the prompt tail wants. `logits` is [ntok][vocab] device f32. */
static int verify_logits(Model * m, const uint32_t * ids, int t0, int ntok, float * logits) {
    if (forward_batch(m, ids, t0, ntok)) return -1;
    /* The same tail as tail_logits, once per row: the prompt path's single
     * GEMV is the reference this batch must match, so do not route the tied
     * head through the prefill kernel until it is shown to agree. */
    for (int r = 0; r < ntok; r++) {
        rmsnorm_kernel<<<1, 256>>>(m->hb + (size_t) r * m->c.hidden, m->d_norm,
                                   m->xn, m->c.hidden, m->c.eps);
        if (ck(cudaMemsetAsync(logits + (size_t) r * m->c.vocab, 0,
                               (size_t) m->c.vocab * 4), "memset verify logits"))
            return -1;
        if (gemv(&m->embed, m->xn, logits + (size_t) r * m->c.vocab)) return -1;
    }
    return 0;
}

static int argmax_dev(const Model * m) {
    int * d = NULL;
    if (ck(cudaMalloc((void **) &d, 4), "cudaMalloc argmax")) return -1;
    argmax_kernel<<<1, 1024>>>(m->logits, m->c.vocab, d);
    int best = -1;
    if (ck(cudaMemcpy(&best, d, 4, cudaMemcpyDeviceToHost), "argmax read")) best = -1;
    cudaFree(d);
    return best;
}

#include "dspark-gpu.h"

/* ------------------------------------------------------------------ main -- */

static const char * USAGE =
    "usage: vq-model-gpu --dir DIR --ids LIST [options]\n"
    "  --config PATH     model config.json (default models/hf/config.json)\n"
    "  --ids LIST        comma or space separated prompt token ids\n"
    "  --gen N           greedy continuation length (default 0)\n"
    "  --ctx N           KV cache size in tokens (default 4096)\n"
    "  --chunk N         prompt tokens per prefill launch (default 512)\n"
    "  --no-prefill      run the prompt token by token even when the tiled path fits\n"
    "  --profile PATH    tuning profile (default DIR/vq-tune.profile when present)\n"
    "  --no-tune         ignore any tuning profile\n"
    "  --top N           print the N best next tokens (default 5)\n"
    "  --logits FILE     write the last prompt position's f32 logits\n"
    "  --draft-info DIR  read a converted draft and print its header, no target\n"
    "  --draft DIR       run the DSpark propose/verify loop against the target\n"
    "  --n-max K         draft tokens per round (default the manifest block_size)\n"
    "  --rounds          print one line per speculative round\n"
    "  --out FILE        write the generated ids to FILE\n"
    "  --gate1           batch permutation invariance across four draft blocks\n"
    "  --bench           run the spec path and the greedy baseline, report break-even\n";

static int parse_ids(const char * s, uint32_t ** out, int * n) {
    char * copy = strdup(s);
    if (!copy) return -1;
    int cap = 16, k = 0;
    uint32_t * ids = (uint32_t *) malloc((size_t) cap * sizeof(uint32_t));
    char * p = copy;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == ';' || *p == '\t') p++;
        if (!*p) break;
        char * e = NULL;
        unsigned long v = strtoul(p, &e, 10);
        if (e == p) { free(ids); free(copy); return -1; }
        if (k == cap) {
            cap *= 2;
            ids = (uint32_t *) realloc(ids, (size_t) cap * sizeof(uint32_t));
            if (!ids) { free(copy); return -1; }
        }
        ids[k++] = (uint32_t) v;
        p = e;
    }
    free(copy);
    *out = ids;
    *n = k;
    return k > 0 ? 0 : -1;
}

int main(int argc, char ** argv) {
    const char * dir = NULL, * cfg_path = "models/hf/config.json", * ids_arg = NULL;
    const char * logits_out = NULL, * profile_arg = NULL, * draft_info = NULL;
    const char * draft_dir = NULL, * out_path = NULL;
    int gen = 0, ctx = 4096, topn = 5, chunk = 512, no_prefill = 0, no_tune = 0;
    int n_max = 0, trace = 0, gate1 = 0, bench = 0;

    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        const char * val = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "--dir") && val) { dir = val; i++; }
        else if (!strcmp(a, "--config") && val) { cfg_path = val; i++; }
        else if (!strcmp(a, "--ids") && val) { ids_arg = val; i++; }
        else if (!strcmp(a, "--gen") && val) { gen = atoi(val); i++; }
        else if (!strcmp(a, "--ctx") && val) { ctx = atoi(val); i++; }
        else if (!strcmp(a, "--chunk") && val) { chunk = atoi(val); i++; }
        else if (!strcmp(a, "--no-prefill")) { no_prefill = 1; }
        else if (!strcmp(a, "--profile") && val) { profile_arg = val; i++; }
        else if (!strcmp(a, "--no-tune")) { no_tune = 1; }
        else if (!strcmp(a, "--top") && val) { topn = atoi(val); i++; }
        else if (!strcmp(a, "--logits") && val) { logits_out = val; i++; }
        else if (!strcmp(a, "--draft-info") && val) { draft_info = val; i++; }
        else if (!strcmp(a, "--draft") && val) { draft_dir = val; i++; }
        else if (!strcmp(a, "--n-max") && val) { n_max = atoi(val); i++; }
        else if (!strcmp(a, "--rounds")) { trace = 1; }
        else if (!strcmp(a, "--gate1")) { gate1 = 1; }
        else if (!strcmp(a, "--bench")) { bench = 1; }
        else if (!strcmp(a, "--out") && val) { out_path = val; i++; }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { fputs(USAGE, stdout); return 0; }
        else { fprintf(stderr, "unknown argument %s\n%s", a, USAGE); return 2; }
    }
    if (ctx < 2 || gen < 0 || topn < 1) { fprintf(stderr, "bad --ctx, --gen or --top\n"); return 2; }
    if (chunk < 1) { fprintf(stderr, "bad --chunk\n"); return 2; }
    if (chunk > ctx) chunk = ctx;

    if (draft_info) {
        Draft d;
        if (draft_open(&d, draft_info, ctx)) return 2;
        printf("draft       %s, %d layers, %d hidden, %d ffn, %d heads / %d kv of %d\n",
               d.m.arch, d.layers, d.hidden, d.ffn, d.heads, d.kv_heads, d.head_dim);
        printf("draft       vocab %d, mask %d, block %d, anchor mode %d, swa %d, causal %d\n",
               d.vocab, d.mask_token, d.block_size, d.anchor_mode, d.swa_window, d.causal);
        printf("draft       targets [");
        for (int i = 0; i < d.n_target; i++)
            printf("%d%s", d.target_layers[i], i + 1 < d.n_target ? ", " : "");
        printf("], markov %d, confidence %d, tensors %d\n",
               d.has_markov, d.has_conf, d.m.n_tensors);
        return 0;
    }

    if (!dir || !ids_arg) { fputs(USAGE, stderr); return 2; }

    Cfg c;
    if (parse_config(cfg_path, &c)) return 2;
    printf("model       %d layers, %d heads / %d kv, %d hidden, %d intermediate, vocab %d\n",
           c.layers, c.heads, c.kv_heads, c.hidden, c.inter, c.vocab);
    printf("rope        full theta %g partial %g, sliding theta %g partial %g\n",
           c.rope_full_theta, c.rope_full_partial, c.rope_slide_theta, c.rope_slide_partial);

    uint32_t * ids = NULL;
    int nids = 0;
    if (parse_ids(ids_arg, &ids, &nids)) { fprintf(stderr, "--ids needs at least one token id\n"); return 2; }

    /* The tuning profile is advisory: a missing file, a different format, a
     * different artifact or a different device all fall back to the readiness
     * heuristic, each with its own line of explanation. */
    if (!no_tune) {
        char def[4096];
        const char * pf = profile_arg;
        if (!pf) { snprintf(def, sizeof def, "%s/vq-tune.profile", dir); pf = def; }
        unsigned long long fp = 0;
        if (vqprof_artifact(dir, &fp)) {
            printf("profile     none; %s/manifest.txt could not be fingerprinted\n", dir);
        } else if (vqprof_load(pf, &g_prof)) {
            printf("profile     %s is absent or not version %d; using heuristics\n", pf, VQPROF_VERSION);
        } else if (g_prof.artifact != fp) {
            printf("profile     %s was tuned on artifact %016llx, this one is %016llx; ignored\n",
                   pf, g_prof.artifact, fp);
        } else {
            cudaDeviceProp prop;
            memset(&prop, 0, sizeof prop);
            if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
                printf("profile     device query failed; using heuristics\n");
            } else if (strcmp(prop.name, g_prof.gpu)) {
                printf("profile     %s was tuned on %s, this device is %s; ignored\n",
                       pf, g_prof.gpu, prop.name);
            } else {
                g_prof_on = 1;
                printf("profile     %s matches artifact %016llx on %s (%d entries, driver %s)\n",
                       pf, fp, prop.name, g_prof.n, g_prof.driver);
            }
        }
    }

    const double t_load = now_s();
    Model m;
    memset(&m, 0, sizeof m);
    if (alloc_model(&m, &c, ctx, chunk)) { fprintf(stderr, "device allocation failed (ctx %d, chunk %d)\n", ctx, chunk); return 2; }
    if (load_model(&m, dir)) return 2;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    printf("weights     %s (%.2f GB container streamed per token, %.2f GB free of %.2f GB after load) in %.1f s\n",
           dir, (double) g_container_bytes / 1e9, (double) fr / 1e9, (double) tot / 1e9,
           now_s() - t_load);

    if (draft_dir) {
        Draft d;
        if (draft_open(&d, draft_dir, ctx)) return 2;
        printf("draft       %s, %d layers, block %d, anchor mode %d, markov %d, confidence %d\n",
               d.m.arch, d.layers, d.block_size, d.anchor_mode, d.has_markov, d.has_conf);
        if (gate1) {
            const int rc = draft_gate1(&d, &m, ids, nids);
            return rc == 0 ? 0 : 2;
        }
        if (bench) {
            const int rc = draft_bench(&d, &m, ids, nids, gen, n_max);
            return rc == 0 ? 0 : 2;
        }
        if (draft_spec_run(&d, &m, ids, nids, gen, n_max, trace, out_path, NULL)) return 2;
        return 0;
    }

    const int use_prefill = (!no_prefill && m.prefill_ok);
    if (use_prefill)
        printf("prefill     tiled f16 mma, %d tokens per chunk\n", chunk);
    else if (!m.prefill_ok)
        printf("prefill     off, a matrix's geometry cannot take the tiled kernel; the prompt runs token by token\n");
    else
        printf("prefill     off, the prompt runs token by token\n");

    const double t_prompt = now_s();
    if (use_prefill) {
        int last_ntok = 0;
        for (int t0 = 0; t0 < nids; t0 += chunk) {
            const int ntok = (nids - t0 < chunk) ? nids - t0 : chunk;
            if (forward_batch(&m, ids + t0, t0, ntok)) return 2;
            last_ntok = ntok;
        }
        if (tail_logits(&m, m.hb + (size_t) (last_ntok - 1) * c.hidden)) return 2;
        if (ck(cudaDeviceSynchronize(), "prompt sync")) return 2;
        const double prompt_s = now_s() - t_prompt;
        printf("prompt      %d tokens in %.2f s (%.2f tok/s, tiled prefill)\n",
               nids, prompt_s, (double) nids / prompt_s);
    } else {
        /* One token per forward, which is the decode path; the same ids the
         * CPU reference was given, so the two logit vectors are comparable. */
        for (int i = 0; i < nids; i++) {
            if (forward(&m, ids[i], i)) return 2;
            if (i + 1 < nids && ck(cudaDeviceSynchronize(), "prompt sync")) return 2;
        }
        if (ck(cudaDeviceSynchronize(), "prompt sync")) return 2;
        const double prompt_s = now_s() - t_prompt;
        printf("prompt      %d tokens in %.2f s (%.2f tok/s)\n", nids, prompt_s, (double) nids / prompt_s);
    }

    float * host_logits = (float *) malloc((size_t) c.vocab * 4);
    if (!host_logits) { fprintf(stderr, "cannot allocate the logit buffer\n"); return 2; }
    if (ck(cudaMemcpy(host_logits, m.logits, (size_t) c.vocab * 4, cudaMemcpyDeviceToHost), "logit read")) return 2;

    if (logits_out) {
        FILE * f = fopen(logits_out, "wb");
        if (!f) { fprintf(stderr, "%s: cannot write\n", logits_out); return 2; }
        if (fwrite(host_logits, 4, (size_t) c.vocab, f) != (size_t) c.vocab) {
            fprintf(stderr, "%s: short write\n", logits_out);
            fclose(f);
            return 2;
        }
        fclose(f);
        printf("logits      %d floats written to %s\n", c.vocab, logits_out);
    }

    /* Top-n, best first, by index (the CPU prints the same form). */
    int * order = (int *) malloc((size_t) c.vocab * sizeof(int));
    for (int i = 0; i < c.vocab; i++) order[i] = i;
    /* Partial selection: only the first `topn` positions matter. */
    for (int i = 0; i < topn && i < c.vocab; i++) {
        int best = i;
        for (int j = i + 1; j < c.vocab; j++) if (host_logits[order[j]] > host_logits[order[best]]) best = j;
        const int t = order[i];
        order[i] = order[best];
        order[best] = t;
    }
    printf("next        ");
    for (int i = 0; i < topn && i < c.vocab; i++)
        printf("[%d %.4f] ", order[i], host_logits[order[i]]);
    printf("\n");
    int next = order[0];
    free(order);

    /* Greedy continuation: the same loop as main.rs:181-198, including the
     * forward of the last emitted token so the printed ids are the tokens the
     * model actually produced next. */
    uint32_t * out = (uint32_t *) calloc((size_t) (gen > 0 ? gen : 1), sizeof(uint32_t));
    const double t_gen = now_s();
    for (int k = 0; k < gen; k++) {
        out[k] = (uint32_t) next;
        const int pos = nids + k;
        if (forward(&m, (uint32_t) next, pos)) return 2;
        if (ck(cudaDeviceSynchronize(), "gen sync")) return 2;
        next = argmax_dev(&m);
        if (next < 0) return 2;
    }
    const double gen_s = now_s() - t_gen;
    if (gen > 0) {
        printf("generated   [");
        for (int k = 0; k < gen; k++) printf("%u%s", out[k], k + 1 < gen ? ", " : "");
        printf("] in %.2f s (%.2f tok/s)\n", gen_s, (double) gen / gen_s);
    }
    free(out);
    free(host_logits);
    free(ids);
    return 0;
}
