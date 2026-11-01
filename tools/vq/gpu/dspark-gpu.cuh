/* dspark-gpu.cuh — the DSpark draft executor on the same device as the target.
 *
 * Included by vq-model-gpu.cu after the target Model, its Mat loader, gemv,
 * forward_batch, verify_logits and capture support are defined. Every
 * dimension comes from out/draft/manifest.txt; no Spark constant is compiled
 * in. The draft weights are loaded as f32 (F32, F16 or BF16 in the manifest)
 * so the GPU arithmetic follows the CPU reference in draft.rs, which is the
 * gate: the same ids at temperature 0.
 *
 * The block decoder is batched over S independent blocks (S=1 for the spec
 * loop, S>1 for gate 1, batch permutation invariance). The injected target
 * features are shared by every block in a batch; each block has its own anchor
 * and its own K/V rows, so its output cannot depend on the order of the batch.
 */

#ifndef DSPARK_GPU_CUH
#define DSPARK_GPU_CUH

#include <cublas_v2.h>

#define DSPARK_MAX_LAYERS 8
#define DSPARK_MAX_TARGET 8
#define DSPARK_MAX_PATTERN 128
#define DSPARK_MAXKEYS 1024

typedef struct {
    char role[32];
    int layer;
    int dtype;
    unsigned long long ne0, ne1, offset, bytes;
} DTensor;

typedef struct {
    char arch[32];
    int hidden, layers, ffn, heads, kv_heads, head_dim, vocab;
    int mask_token;
    int rope_full_dims, rope_swa_dims;
    double rope_full_theta, rope_swa_theta;
    float eps;
    char act[32];
    int swa_window;
    int swa_pattern[DSPARK_MAX_PATTERN];
    int n_swa;
    int causal;
    int block_size;
    int sample_from_anchor;
    int target_layers[DSPARK_MAX_TARGET];
    int n_target;
    DTensor tensors[128];
    int n_tensors;
} DManifest;

static unsigned long long dman_num(const char * v) {
    return strtoull(v, NULL, 10);
}

static double dman_float(const char * v) {
    return strtod(v, NULL);
}

static int dman_list(const char * v, int * out, int max) {
    int n = 0;
    const char * p = v;
    while (*p && n < max) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char * e = NULL;
        long x = strtol(p, &e, 10);
        if (e == p) return -1;
        out[n++] = (int) x;
        p = e;
    }
    return n;
}

/* Read manifest.txt. The format is the one dspark convert writes (DRAFT 1,
 * one key per line, then one tensor line per region, then end). */
static int dman_parse(const char * path, DManifest * m, unsigned long long weights_len) {
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "%s: cannot read\n", path); return -1; }
    memset(m, 0, sizeof *m);
    char line[1024];
    int first = 1, ended = 0;
    while (fgets(line, sizeof line, f)) {
        char * s = line;
        while (*s == ' ' || *s == '\t') s++;
        char * e = s + strlen(s);
        while (e > s && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) *--e = 0;
        if (!*s) continue;
        if (first) {
            if (strcmp(s, "DRAFT 1")) {
                fprintf(stderr, "%s: first line is '%s', expected 'DRAFT 1'\n", path, s);
                fclose(f);
                return -1;
            }
            first = 0;
            continue;
        }
        if (!strcmp(s, "end")) { ended = 1; break; }
        if (!strncmp(s, "tensor ", 7)) {
            if (m->n_tensors >= (int) (sizeof m->tensors / sizeof m->tensors[0])) {
                fprintf(stderr, "%s: more than %zu tensors\n", path,
                        sizeof m->tensors / sizeof m->tensors[0]);
                fclose(f);
                return -1;
            }
            DTensor * t = &m->tensors[m->n_tensors];
            char role[32];
            int layer, dtype;
            unsigned long long ne0, ne1, off, bytes;
            if (sscanf(s + 7, "%31s %d %d %llu %llu %llu %llu",
                       role, &layer, &dtype, &ne0, &ne1, &off, &bytes) != 7) {
                fprintf(stderr, "%s: malformed tensor line: %s\n", path, s);
                fclose(f);
                return -1;
            }
            snprintf(t->role, sizeof t->role, "%s", role);
            t->layer = layer;
            t->dtype = dtype;
            t->ne0 = ne0;
            t->ne1 = ne1;
            t->offset = off;
            t->bytes = bytes;
            if (off + bytes > weights_len) {
                fprintf(stderr, "%s: %s %d ends at %llu, weights.bin is %llu bytes\n",
                        path, role, layer, off + bytes, weights_len);
                fclose(f);
                return -1;
            }
            m->n_tensors++;
            continue;
        }
        char key[64], val[512];
        if (sscanf(s, "%63s %511[^\n]", key, val) != 2) {
            fprintf(stderr, "%s: malformed line: %s\n", path, s);
            fclose(f);
            return -1;
        }
        if (!strcmp(key, "arch")) snprintf(m->arch, sizeof m->arch, "%s", val);
        else if (!strcmp(key, "hidden")) m->hidden = (int) dman_num(val);
        else if (!strcmp(key, "layers")) m->layers = (int) dman_num(val);
        else if (!strcmp(key, "ffn")) m->ffn = (int) dman_num(val);
        else if (!strcmp(key, "heads")) m->heads = (int) dman_num(val);
        else if (!strcmp(key, "kv_heads")) m->kv_heads = (int) dman_num(val);
        else if (!strcmp(key, "head_dim")) m->head_dim = (int) dman_num(val);
        else if (!strcmp(key, "vocab")) m->vocab = (int) dman_num(val);
        else if (!strcmp(key, "mask_token")) m->mask_token = (int) dman_num(val);
        else if (!strcmp(key, "rope_full_dims")) m->rope_full_dims = (int) dman_num(val);
        else if (!strcmp(key, "rope_swa_dims")) m->rope_swa_dims = (int) dman_num(val);
        else if (!strcmp(key, "rope_full_theta")) m->rope_full_theta = dman_float(val);
        else if (!strcmp(key, "rope_swa_theta")) m->rope_swa_theta = dman_float(val);
        else if (!strcmp(key, "eps")) m->eps = (float) dman_float(val);
        else if (!strcmp(key, "act")) snprintf(m->act, sizeof m->act, "%s", val);
        else if (!strcmp(key, "swa_window")) m->swa_window = (int) dman_num(val);
        else if (!strcmp(key, "swa_pattern")) {
            m->n_swa = dman_list(val, m->swa_pattern, DSPARK_MAX_PATTERN);
            if (m->n_swa < 0) {
                fprintf(stderr, "%s: swa_pattern is malformed\n", path);
                fclose(f);
                return -1;
            }
        } else if (!strcmp(key, "causal")) m->causal = (int) dman_num(val);
        else if (!strcmp(key, "block_size")) m->block_size = (int) dman_num(val);
        else if (!strcmp(key, "sample_from_anchor")) m->sample_from_anchor = (int) dman_num(val);
        else if (!strcmp(key, "target_layers")) {
            m->n_target = dman_list(val, m->target_layers, DSPARK_MAX_TARGET);
            if (m->n_target < 0) {
                fprintf(stderr, "%s: target_layers is malformed\n", path);
                fclose(f);
                return -1;
            }
        } else {
            fprintf(stderr, "%s: unknown key '%s'\n", path, key);
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    if (first || !ended) {
        fprintf(stderr, "%s: the manifest is empty or has no 'end'\n", path);
        return -1;
    }
    if (m->hidden <= 0 || m->layers <= 0 || m->layers > DSPARK_MAX_LAYERS ||
        m->ffn <= 0 || m->heads <= 0 || m->kv_heads <= 0 || m->head_dim <= 0 ||
        m->vocab <= 0 || m->block_size < 1) {
        fprintf(stderr, "%s: a header dimension is zero or out of range\n", path);
        return -1;
    }
    if (m->heads % m->kv_heads) {
        fprintf(stderr, "%s: heads %d is not a multiple of kv_heads %d\n",
                path, m->heads, m->kv_heads);
        return -1;
    }
    if (m->mask_token < 0 || m->mask_token >= m->vocab) {
        fprintf(stderr, "%s: mask_token %d is outside the vocabulary\n", path, m->mask_token);
        return -1;
    }
    if (m->n_swa != m->layers) {
        fprintf(stderr, "%s: swa_pattern has %d entries for %d layers\n",
                path, m->n_swa, m->layers);
        return -1;
    }
    if (m->n_target <= 0) {
        fprintf(stderr, "%s: target_layers is empty\n", path);
        return -1;
    }
    if (m->n_tensors <= 0) {
        fprintf(stderr, "%s: no tensor lines\n", path);
        return -1;
    }
    return 0;
}

static const DTensor * dman_find(const DManifest * m, const char * role, int layer) {
    for (int i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].layer == layer && !strcmp(m->tensors[i].role, role))
            return &m->tensors[i];
    }
    return NULL;
}

typedef struct {
    int rows, cols;
    float * d;
} DMat;

static int dtype_bytes(int dtype) {
    if (dtype == 0) return 4;
    if (dtype == 1 || dtype == 2) return 2;
    return 0;
}

static float bf16_to_f32(unsigned short h) {
    unsigned int bits = (unsigned int) h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float f16_to_f32(unsigned short h) {
    const int sign = (h & 0x8000) != 0;
    const int exp = (h >> 10) & 0x1f;
    const float man = (float) (h & 0x03ff) / 1024.0f;
    float v;
    if (exp == 0) v = man * powf(2.0f, -14.0f);
    else if (exp == 0x1f) v = (man == 0.0f) ? INFINITY : NAN;
    else v = (1.0f + man) * powf(2.0f, (float) exp - 15.0f);
    return sign ? -v : v;
}

/* Load one manifest region and convert it to device f32. The conversion is
 * the same arithmetic as dspark draft.rs:load_mat, so the GPU starts from the
 * same values the CPU gate used. */
static int dmat_load(FILE * wf, const DTensor * t, DMat * out) {
    const int per = dtype_bytes(t->dtype);
    if (!per) {
        fprintf(stderr, "%s %d: dtype %d is not F32, F16 or BF16\n",
                t->role, t->layer, t->dtype);
        return -1;
    }
    const unsigned long long elems = t->ne0 * t->ne1;
    if (elems == 0 || elems > (1ull << 32)) {
        fprintf(stderr, "%s %d: %llu elements is out of range\n", t->role, t->layer, elems);
        return -1;
    }
    if (t->bytes != elems * (unsigned long long) per) {
        fprintf(stderr, "%s %d: %llu bytes, %llu x %llu of %d bytes needs %llu\n",
                t->role, t->layer, t->bytes, t->ne0, t->ne1, per, elems * (unsigned long long) per);
        return -1;
    }
    unsigned char * raw = (unsigned char *) malloc((size_t) t->bytes);
    float * host = (float *) malloc((size_t) elems * 4);
    if (!raw || !host) { free(raw); free(host); return -1; }
    if (fseeko(wf, (off_t) t->offset, SEEK_SET)) { free(raw); free(host); return -1; }
    if (fread(raw, 1, (size_t) t->bytes, wf) != (size_t) t->bytes) {
        fprintf(stderr, "%s %d: short read\n", t->role, t->layer);
        free(raw); free(host);
        return -1;
    }
    for (unsigned long long i = 0; i < elems; i++) {
        if (t->dtype == 0) {
            float v;
            memcpy(&v, raw + i * 4, 4);
            host[i] = v;
        } else {
            unsigned short h;
            memcpy(&h, raw + i * 2, 2);
            host[i] = (t->dtype == 2) ? bf16_to_f32(h) : f16_to_f32(h);
        }
    }
    out->rows = (int) t->ne1;
    out->cols = (int) t->ne0;
    out->d = NULL;
    if (cudaMalloc((void **) &out->d, (size_t) elems * 4) != cudaSuccess) {
        fprintf(stderr, "%s %d: cudaMalloc %llu floats failed\n", t->role, t->layer, elems);
        free(raw); free(host);
        return -1;
    }
    if (cudaMemcpy(out->d, host, (size_t) elems * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "%s %d: upload failed\n", t->role, t->layer);
        cudaFree(out->d); out->d = NULL;
        free(raw); free(host);
        return -1;
    }
    free(raw);
    free(host);
    return 0;
}

static int dvec_load(FILE * wf, const DTensor * t, int want, float ** out) {
    if (!t) { fprintf(stderr, "a required vector is missing\n"); return -1; }
    DMat m;
    if (dmat_load(wf, t, &m)) return -1;
    if (m.rows != 1 || m.cols != want) {
        fprintf(stderr, "%s %d: is %dx%d, expected 1x%d\n", t->role, t->layer,
                m.rows, m.cols, want);
        cudaFree(m.d);
        return -1;
    }
    *out = m.d;
    return 0;
}

/* An optional vector: absent is success with a NULL destination, a present but
 * malformed tensor is still an error. */
static int dvec_load_opt(FILE * wf, const DTensor * t, int want, float ** out) {
    *out = NULL;
    if (!t) return 0;
    return dvec_load(wf, t, want, out);
}

typedef struct {
    float * norm;
    DMat q, k, v, gate, out;
    float * q_norm, * k_norm;
    float * ffn_norm;
    DMat ffn_gate, ffn_up, ffn_down;
    float * attn_post, * ffn_post;
    float out_scale;
    int has_v, has_gate, has_q_norm, has_k_norm, has_attn_post, has_ffn_post, has_out_scale;
} DLayer;

typedef struct {
    DManifest m;
    int hidden, layers, ffn, heads, kv_heads, head_dim, vocab, qdim, kvdim;
    int block_size, mask_token, swa_window, causal, sample_from_anchor;
    int n_target;
    int target_layers[DSPARK_MAX_TARGET];
    int is_dspark, anchor_mode;
    DLayer layer[DSPARK_MAX_LAYERS];
    DMat fc;
    float * fc_norm;              /* [n_target][hidden] */
    float * enc_norm, * out_norm;
    DMat markov_w1, markov_w2, conf_proj;
    int has_markov, has_conf;
    float conf_bias;
    int ctx;
    cublasHandle_t blas;

    float * inj_k[DSPARK_MAX_LAYERS], * inj_v[DSPARK_MAX_LAYERS];

    float * X, * XN, * Q, * K, * V, * G, * ATTN, * O;
    float * FG, * FU, * ACT, * H;
    float * LOGITS, * BIAS, * W1ROW;
    float * EMB_ANCHOR, * EMB_MASK, * CAT, * GENC, * CONF;
    int * ROWARG;
    int max_rows, max_blocks;
} Draft;

static int draft_open(Draft * d, const char * dir, int ctx) {
    memset(d, 0, sizeof *d);
    char mpath[4096], wpath[4096];
    snprintf(mpath, sizeof mpath, "%s/manifest.txt", dir);
    snprintf(wpath, sizeof wpath, "%s/weights.bin", dir);
    FILE * wf = fopen(wpath, "rb");
    if (!wf) { fprintf(stderr, "%s: cannot read\n", wpath); return -1; }
    if (fseeko(wf, 0, SEEK_END)) { fclose(wf); return -1; }
    const unsigned long long wlen = (unsigned long long) ftello(wf);
    rewind(wf);
    if (dman_parse(mpath, &d->m, wlen)) { fclose(wf); return -1; }

    const DManifest * m = &d->m;
    d->hidden = m->hidden;
    d->layers = m->layers;
    d->ffn = m->ffn;
    d->heads = m->heads;
    d->kv_heads = m->kv_heads;
    d->head_dim = m->head_dim;
    d->vocab = m->vocab;
    d->qdim = m->heads * m->head_dim;
    d->kvdim = m->kv_heads * m->head_dim;
    d->block_size = m->block_size;
    d->mask_token = m->mask_token;
    d->swa_window = m->swa_window;
    d->causal = m->causal;
    d->sample_from_anchor = m->sample_from_anchor;
    d->n_target = m->n_target;
    for (int i = 0; i < m->n_target; i++) d->target_layers[i] = m->target_layers[i];
    d->is_dspark = dman_find(m, "markov_w1", -1) != NULL;
    d->anchor_mode = m->sample_from_anchor != 0;
    d->ctx = ctx;
    d->max_rows = m->block_size + 1;
    d->max_blocks = 8;
    if (m->swa_window + m->block_size + 2 > DSPARK_MAXKEYS) {
        fprintf(stderr, "draft: swa_window %d plus block_size %d exceeds the %d-key kernel\n",
                m->swa_window, m->block_size, DSPARK_MAXKEYS);
        return -1;
    }
    if (m->rope_full_dims <= 0 || m->rope_full_dims > m->head_dim || (m->rope_full_dims & 1) ||
        m->rope_swa_dims <= 0 || m->rope_swa_dims > m->head_dim || (m->rope_swa_dims & 1)) {
        fprintf(stderr, "draft: rope dims %d/%d are not positive even values within head_dim %d\n",
                m->rope_full_dims, m->rope_swa_dims, m->head_dim);
        return -1;
    }

    /* Globals. */
    const DTensor * t;
    if ((t = dman_find(m, "fc", -1))) {
        if (dmat_load(wf, t, &d->fc)) { fclose(wf); return -1; }
        if (d->fc.rows != d->hidden || d->fc.cols != d->n_target * d->hidden) {
            fprintf(stderr, "fc: is %dx%d, expected %dx%d\n", d->fc.rows, d->fc.cols,
                    d->hidden, d->n_target * d->hidden);
            fclose(wf);
            return -1;
        }
    } else { fprintf(stderr, "manifest has no fc\n"); fclose(wf); return -1; }
    if ((t = dman_find(m, "fc_norm", -1))) {
        DMat fcn;
        if (dmat_load(wf, t, &fcn)) { fclose(wf); return -1; }
        if (fcn.rows != d->n_target || fcn.cols != d->hidden) {
            fprintf(stderr, "fc_norm: is %dx%d, expected %dx%d\n", fcn.rows, fcn.cols,
                    d->n_target, d->hidden);
            fclose(wf);
            return -1;
        }
        d->fc_norm = fcn.d;
    } else { fprintf(stderr, "manifest has no fc_norm\n"); fclose(wf); return -1; }
    if (dvec_load(wf, dman_find(m, "enc_norm", -1), d->hidden, &d->enc_norm) ||
        dvec_load(wf, dman_find(m, "output_norm", -1), d->hidden, &d->out_norm)) {
        fclose(wf);
        return -1;
    }
    if ((t = dman_find(m, "markov_w1", -1))) {
        if (dmat_load(wf, t, &d->markov_w1)) { fclose(wf); return -1; }
        if (d->markov_w1.rows != d->vocab || d->markov_w1.cols <= 0) {
            fprintf(stderr, "markov_w1: is %dx%d, expected %dx<rank>\n",
                    d->markov_w1.rows, d->markov_w1.cols, d->vocab);
            fclose(wf);
            return -1;
        }
        d->has_markov = 1;
    }
    if ((t = dman_find(m, "markov_w2", -1))) {
        if (dmat_load(wf, t, &d->markov_w2)) { fclose(wf); return -1; }
        if (d->markov_w2.rows != d->vocab || d->markov_w2.cols != d->markov_w1.cols) {
            fprintf(stderr, "markov_w2: is %dx%d, expected %dx%d\n",
                    d->markov_w2.rows, d->markov_w2.cols, d->vocab, d->markov_w1.cols);
            fclose(wf);
            return -1;
        }
    }
    if (d->has_markov && !d->markov_w2.d) {
        fprintf(stderr, "markov_w1 is present without markov_w2\n");
        fclose(wf);
        return -1;
    }
    if ((t = dman_find(m, "conf_proj", -1))) {
        if (!d->has_markov) {
            fprintf(stderr, "conf_proj is present without markov_w1\n");
            fclose(wf);
            return -1;
        }
        if (dmat_load(wf, t, &d->conf_proj)) { fclose(wf); return -1; }
        if (d->conf_proj.rows != 1 || d->conf_proj.cols != d->hidden + d->markov_w1.cols) {
            fprintf(stderr, "conf_proj: is %dx%d, expected 1x%d\n",
                    d->conf_proj.rows, d->conf_proj.cols, d->hidden + d->markov_w1.cols);
            fclose(wf);
            return -1;
        }
        d->has_conf = 1;
    }
    if ((t = dman_find(m, "conf_bias", -1))) {
        DMat cb;
        if (dmat_load(wf, t, &cb)) { fclose(wf); return -1; }
        if (cb.rows != 1 || cb.cols != 1) {
            fprintf(stderr, "conf_bias is not one float\n");
            cudaFree(cb.d);
            fclose(wf);
            return -1;
        }
        if (cudaMemcpy(&d->conf_bias, cb.d, 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "conf_bias: read failed\n");
            cudaFree(cb.d);
            fclose(wf);
            return -1;
        }
        cudaFree(cb.d);
    }

    /* Per-layer weights. */
    for (int li = 0; li < d->layers; li++) {
        DLayer * L = &d->layer[li];
        const DTensor * tn = dman_find(m, "attn_norm", li);
        if (dvec_load(wf, tn, d->hidden, &L->norm)) { fclose(wf); return -1; }
        const DTensor * tq = dman_find(m, "attn_q", li);
        const DTensor * tk = dman_find(m, "attn_k", li);
        const DTensor * tv = dman_find(m, "attn_v", li);
        const DTensor * to = dman_find(m, "attn_out", li);
        if (!tq || !tk || !to) { fprintf(stderr, "layer %d: a required attention tensor is missing\n", li); fclose(wf); return -1; }
        if (dmat_load(wf, tq, &L->q) || dmat_load(wf, tk, &L->k) || dmat_load(wf, to, &L->out)) {
            fclose(wf);
            return -1;
        }
        if (L->q.rows != d->qdim || L->q.cols != d->hidden ||
            L->k.rows != d->kvdim || L->k.cols != d->hidden ||
            L->out.rows != d->hidden || L->out.cols != d->qdim) {
            fprintf(stderr, "layer %d: q/k/out box does not match the header\n", li);
            fclose(wf);
            return -1;
        }
        if (tv) {
            if (dmat_load(wf, tv, &L->v)) { fclose(wf); return -1; }
            if (L->v.rows != d->kvdim || L->v.cols != d->hidden) {
                fprintf(stderr, "layer %d: v box does not match the header\n", li);
                fclose(wf);
                return -1;
            }
            L->has_v = 1;
        }
        const DTensor * tg = dman_find(m, "attn_gate", li);
        if (tg) {
            if (dmat_load(wf, tg, &L->gate)) { fclose(wf); return -1; }
            if (L->gate.rows != d->heads || L->gate.cols != d->hidden) {
                fprintf(stderr, "layer %d: attn_gate box does not match the header\n", li);
                fclose(wf);
                return -1;
            }
            L->has_gate = 1;
        }
        if (dvec_load_opt(wf, dman_find(m, "attn_q_norm", li), d->head_dim, &L->q_norm)) {
            fclose(wf);
            return -1;
        }
        L->has_q_norm = L->q_norm != NULL;
        if (dvec_load_opt(wf, dman_find(m, "attn_k_norm", li), d->head_dim, &L->k_norm)) {
            fclose(wf);
            return -1;
        }
        L->has_k_norm = L->k_norm != NULL;
        if (L->has_q_norm != L->has_k_norm) {
            fprintf(stderr, "layer %d: q_norm and k_norm must be present together\n", li);
            fclose(wf);
            return -1;
        }
        if (dvec_load(wf, dman_find(m, "ffn_norm", li), d->hidden, &L->ffn_norm)) {
            fclose(wf);
            return -1;
        }
        const DTensor * tfg = dman_find(m, "ffn_gate", li);
        const DTensor * tfu = dman_find(m, "ffn_up", li);
        const DTensor * tfd = dman_find(m, "ffn_down", li);
        if (!tfg || !tfu || !tfd) { fprintf(stderr, "layer %d: an ffn tensor is missing\n", li); fclose(wf); return -1; }
        if (dmat_load(wf, tfg, &L->ffn_gate) || dmat_load(wf, tfu, &L->ffn_up) ||
            dmat_load(wf, tfd, &L->ffn_down)) { fclose(wf); return -1; }
        if (L->ffn_gate.rows != d->ffn || L->ffn_gate.cols != d->hidden ||
            L->ffn_up.rows != d->ffn || L->ffn_up.cols != d->hidden ||
            L->ffn_down.rows != d->hidden || L->ffn_down.cols != d->ffn) {
            fprintf(stderr, "layer %d: ffn box does not match the header\n", li);
            fclose(wf);
            return -1;
        }
        if (dvec_load_opt(wf, dman_find(m, "attn_post_norm", li), d->hidden, &L->attn_post)) {
            fclose(wf);
            return -1;
        }
        L->has_attn_post = L->attn_post != NULL;
        if (dvec_load_opt(wf, dman_find(m, "ffn_post_norm", li), d->hidden, &L->ffn_post)) {
            fclose(wf);
            return -1;
        }
        L->has_ffn_post = L->ffn_post != NULL;
        if (dman_find(m, "layer_out_scale", li)) {
            DMat sc;
            if (dmat_load(wf, dman_find(m, "layer_out_scale", li), &sc)) { fclose(wf); return -1; }
            if (sc.rows != 1 || sc.cols != 1) {
                fprintf(stderr, "layer %d: layer_out_scale is not one float\n", li);
                cudaFree(sc.d);
                fclose(wf);
                return -1;
            }
            if (cudaMemcpy(&L->out_scale, sc.d, 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
                fprintf(stderr, "layer %d: layer_out_scale read failed\n", li);
                cudaFree(sc.d);
                fclose(wf);
                return -1;
            }
            cudaFree(sc.d);
            L->has_out_scale = 1;
        }
    }
    fclose(wf);

    /* Device caches and scratch. */
    const size_t rows = (size_t) d->max_rows;
    const size_t nrows = rows * (size_t) d->max_blocks;
    for (int li = 0; li < d->layers; li++) {
        d->inj_k[li] = dev_f32((size_t) ctx * (size_t) d->kvdim);
        d->inj_v[li] = dev_f32((size_t) ctx * (size_t) d->kvdim);
        if (!d->inj_k[li] || !d->inj_v[li]) {
            fprintf(stderr, "draft: cache allocation failed at layer %d\n", li);
            return -1;
        }
    }
    d->X = dev_f32(nrows * (size_t) d->hidden);
    d->XN = dev_f32(nrows * (size_t) d->hidden);
    d->Q = dev_f32(nrows * (size_t) d->qdim);
    d->K = dev_f32(nrows * (size_t) d->kvdim);
    d->V = dev_f32(nrows * (size_t) d->kvdim);
    d->G = dev_f32(nrows * (size_t) d->heads);
    d->ATTN = dev_f32(nrows * (size_t) d->qdim);
    d->O = dev_f32(nrows * (size_t) d->hidden);
    d->FG = dev_f32(nrows * (size_t) d->ffn);
    d->FU = dev_f32(nrows * (size_t) d->ffn);
    d->ACT = dev_f32(nrows * (size_t) d->ffn);
    d->H = dev_f32(nrows * (size_t) d->hidden);
    d->LOGITS = dev_f32(nrows * (size_t) d->vocab);
    d->BIAS = dev_f32((size_t) d->vocab);
    d->W1ROW = dev_f32((size_t) (d->has_markov ? d->markov_w1.cols : 1));
    d->EMB_ANCHOR = dev_f32((size_t) d->max_blocks * (size_t) d->hidden);
    d->EMB_MASK = dev_f32((size_t) d->hidden);
    d->CAT = dev_f32(512 * (size_t) d->n_target * (size_t) d->hidden);
    d->GENC = dev_f32(512 * (size_t) d->hidden);
    d->CONF = dev_f32(nrows);
    if (cudaMalloc((void **) &d->ROWARG, nrows * sizeof(int)) != cudaSuccess) d->ROWARG = NULL;
    if (!d->X || !d->XN || !d->Q || !d->K || !d->V || !d->G || !d->ATTN || !d->O ||
        !d->FG || !d->FU || !d->ACT || !d->H || !d->LOGITS || !d->BIAS || !d->W1ROW ||
        !d->EMB_ANCHOR || !d->EMB_MASK || !d->CAT || !d->GENC || !d->CONF || !d->ROWARG) {
        fprintf(stderr, "draft: scratch allocation failed\n");
        return -1;
    }
    if (cublasCreate(&d->blas) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "draft: cublasCreate failed\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------- kernels -- */

/* The block input: row 0 of each block is its anchor, every other row is the
 * draft's mask token. The embeddings are already dequantized into the target's
 * f32 row form. */
__global__ void draft_fill_block_kernel(float * X, const float * anchors,
                                        const float * mask, int S, int n_block, int hidden) {
    const int r = blockIdx.x;
    const int s = r / n_block, i = r % n_block;
    const float * src = (i == 0) ? anchors + (size_t) s * hidden : mask;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
        X[(size_t) r * hidden + c] = src[c];
}

/* One block per (position, target layer): RMS-norm the captured target row
 * with that layer's fc_norm and write it into the concatenated fc input. */
__global__ void draft_fc_norm_kernel(const float * cap, const float * w, float * out,
                                     int npos, int n_target, int hidden, int ctx,
                                     float eps, int start) {
    __shared__ double ss[256];
    const int t = threadIdx.x;
    const int idx = blockIdx.x;
    const int p = idx / n_target, k = idx % n_target;
    const float * x = cap + ((size_t) k * ctx + start + p) * hidden;
    float * o = out + ((size_t) p * n_target + k) * hidden;
    double s = 0.0;
    for (int i = t; i < hidden; i += blockDim.x) s += (double) x[i] * (double) x[i];
    ss[t] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) ss[t] += ss[t + off];
        __syncthreads();
    }
    const float inv = (float) (1.0 / sqrt(ss[0] / (double) hidden + (double) eps));
    const float * wr = w + (size_t) k * hidden;
    for (int i = t; i < hidden; i += blockDim.x) o[i] = x[i] * inv * wr[i];
}

/* RMS-norm one head of every row with a per-head weight (q_norm / k_norm). */
__global__ void draft_qk_norm_kernel(float * x, const float * w, int rows,
                                     int nheads, int hd, float eps) {
    __shared__ double ss[256];
    const int t = threadIdx.x;
    const int r = blockIdx.x / nheads, h = blockIdx.x % nheads;
    float * xr = x + ((size_t) r * nheads + h) * hd;
    double s = 0.0;
    for (int i = t; i < hd; i += blockDim.x) s += (double) xr[i] * (double) xr[i];
    ss[t] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) ss[t] += ss[t + off];
        __syncthreads();
    }
    const float inv = (float) (1.0 / sqrt(ss[0] / (double) hd + (double) eps));
    for (int i = t; i < hd; i += blockDim.x) xr[i] = xr[i] * inv * w[i];
}

/* Interleaved-half rotation over the first nrot elements of every head. With
 * n_block > 0 the row order is [block][row], so the position is pos0 + r%n_block;
 * the feature encoder passes n_block 0 and uses pos0 + r. */
__global__ void draft_rope_kernel(float * x, int rows, int nheads, int hd, int nrot,
                                  double theta, int pos0, int n_block) {
    const int r = blockIdx.x / nheads, h = blockIdx.x % nheads;
    const int pos = pos0 + (n_block > 0 ? (r % n_block) : r);
    float * base = x + ((size_t) r * nheads + h) * hd;
    const int half = nrot / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const double freq = 1.0 / pow(theta, 2.0 * (double) i / (double) nrot);
        const double a = (double) pos * freq;
        const float c = (float) cos(a), s = (float) sin(a);
        const float x0 = base[i], x1 = base[i + half];
        base[i] = x0 * c - x1 * s;
        base[i + half] = x1 * c + x0 * s;
    }
}

/* Shared-KV drafts borrow V from the normed K before rope. */
__global__ void draft_v_from_k_kernel(float * v, const float * k, int rows,
                                      int nheads, int hd, float eps) {
    __shared__ double ss[256];
    const int t = threadIdx.x;
    const int r = blockIdx.x / nheads, h = blockIdx.x % nheads;
    const float * kr = k + ((size_t) r * nheads + h) * hd;
    float * vr = v + ((size_t) r * nheads + h) * hd;
    double s = 0.0;
    for (int i = t; i < hd; i += blockDim.x) s += (double) kr[i] * (double) kr[i];
    ss[t] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) ss[t] += ss[t + off];
        __syncthreads();
    }
    const float inv = (float) (1.0 / sqrt(ss[0] / (double) hd + (double) eps));
    for (int i = t; i < hd; i += blockDim.x) vr[i] = kr[i] * inv;
}

/* One block per (head, row). Every query uses the anchor row's mask, so the
 * key set is fixed for the block: the last nswa injected positions plus the
 * block's own rows (only the anchor row when causal). The gate is per row and
 * head, from that row's normed input. */
__global__ void draft_attn_kernel(const float * Q, const float * G,
                                  const float * inj_k, const float * inj_v, int inj_len,
                                  const float * own_k, const float * own_v,
                                  int S, int n_block, int pos0, int heads, int kv_heads,
                                  int hd, int nswa, int causal, float * out) {
    extern __shared__ float sh[];
    float * sc = sh;
    float * red = sh + DSPARK_MAXKEYS;
    const int h = blockIdx.x;
    const int r = blockIdx.y;
    const int s = r / n_block;
    const int t = threadIdx.x;
    const int kvh = h * kv_heads / heads;
    const float * q = Q + ((size_t) r * heads + h) * hd;
    const float scale = 1.0f / sqrtf((float) hd);

    int inj_end = pos0;
    if (inj_end > inj_len) inj_end = inj_len;
    int start = inj_end - nswa + 1;
    if (start < 0) start = 0;
    const int n_inj = inj_end - start;
    const int n_own = causal ? 1 : n_block;
    const int nkeys = n_inj + n_own;

    for (int j = t; j < nkeys; j += blockDim.x) {
        const float * k;
        if (j < n_inj) {
            k = inj_k + ((size_t) (start + j) * kv_heads + kvh) * hd;
        } else {
            const int oi = j - n_inj;
            k = own_k + ((size_t) (s * n_block + oi) * kv_heads + kvh) * hd;
        }
        float dot = 0.0f;
        for (int d = 0; d < hd; d++) dot += q[d] * k[d];
        sc[j] = dot * scale;
    }
    __syncthreads();

    float mx = -INFINITY;
    for (int j = t; j < nkeys; j += blockDim.x) mx = fmaxf(mx, sc[j]);
    if (t < 256) red[t] = mx;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off && t < 256) red[t] = fmaxf(red[t], red[t + off]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();

    float sm = 0.0f;
    for (int j = t; j < nkeys; j += blockDim.x) {
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
    const float gate = G ? (1.0f / (1.0f + expf(-G[(size_t) r * heads + h]))) : 1.0f;

    for (int d = t; d < hd; d += blockDim.x) {
        float o = 0.0f;
        for (int j = 0; j < nkeys; j++) {
            const float * v;
            if (j < n_inj) {
                v = inj_v + ((size_t) (start + j) * kv_heads + kvh) * hd;
            } else {
                const int oi = j - n_inj;
                v = own_v + ((size_t) (s * n_block + oi) * kv_heads + kvh) * hd;
            }
            o += sc[j] * inv * v[d];
        }
        out[((size_t) r * heads + h) * hd + d] = o * gate;
    }
}

__global__ void draft_add_kernel(float * x, const float * y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] += y[i];
}

__global__ void draft_scale_kernel(float * x, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] *= s;
}

/* The draft's gelu is ggml's LLM_FFN_GELU, the tanh form (ggml-cpu/vec.h:962);
 * kind 2 is silu. */
__global__ void draft_act_kernel(const float * g, const float * u, float * act, int n, int kind) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float x = g[i];
        float a;
        if (kind == 2) a = x / (1.0f + expf(-x));
        else a = 0.5f * x * (1.0f + tanhf(0.7978846f * x * (1.0f + 0.044715f * x * x)));
        act[i] = a * u[i];
    }
}

__global__ void draft_argmax_batch_kernel(const float * v, int n, int * out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const float * vr = v + (size_t) row * n;
    float best = -INFINITY;
    int b = 0;
    for (int i = t; i < n; i += blockDim.x) {
        if (vr[i] > best) { best = vr[i]; b = i; }
    }
    bv[t] = best;
    bi[t] = b;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off && bv[t + off] > bv[t]) { bv[t] = bv[t + off]; bi[t] = bi[t + off]; }
        __syncthreads();
    }
    if (t == 0) out[row] = bi[0];
}

__global__ void draft_markov_add_kernel(float * logits, const float * bias, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        logits[i] += bias[i];
}

__global__ void draft_conf_kernel(const float * h, const float * w1, const float * cp,
                                  float bias, int hidden, int rank, float * out) {
    __shared__ float red[256];
    const int t = threadIdx.x;
    float s = 0.0f;
    for (int i = t; i < hidden; i += blockDim.x) s += cp[i] * h[i];
    for (int i = t; i < rank; i += blockDim.x) s += cp[hidden + i] * w1[i];
    red[t] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        __syncthreads();
    }
    if (t == 0) out[0] = 1.0f / (1.0f + expf(-(red[0] + bias)));
}

/* ------------------------------------------------------------- executor -- */

/* Y[n][out] = X[n][in] * W[out][in]^T. W is row-major [out][in], which is the
 * column-major [in][out] cuBLAS wants with opA=T; X is row-major [n][in],
 * column-major [in][n] with opB=N. */
static int draft_gemm(Draft * d, const float * W, int out, int in,
                      const float * X, int n, float * Y) {
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasSgemm(d->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                          out, n, in, &alpha, W, in, X, in, &beta, Y, out);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "draft gemm %dx%d x %d: cublas status %d\n", out, in, n, (int) st);
        return -1;
    }
    return 0;
}

/* Encode the target captures at positions [start, end) into the draft's K/V
 * cache. The feature encoder is the reference's graph<true>: per target layer
 * RMS-norm with fc_norm, concatenate, fc, enc_norm, then each draft layer's K
 * (k_norm, rope) and V. Chunked so a long prompt does not need a ctx-sized
 * concatenation buffer. */
static int draft_encode(Draft * d, Model * tgt, int start, int end) {
    if (!tgt->cap) {
        fprintf(stderr, "draft: target capture is not enabled\n");
        return -1;
    }
    if (start < 0 || end > tgt->ctx || start > end) {
        fprintf(stderr, "draft: encode range %d..%d is outside the %d-token capture\n",
                start, end, tgt->ctx);
        return -1;
    }
    const int CH = 512;
    for (int s0 = start; s0 < end; s0 += CH) {
        int npos = end - s0;
        if (npos > CH) npos = CH;
        draft_fc_norm_kernel<<<npos * d->n_target, 256>>>(
            tgt->cap, d->fc_norm, d->CAT, npos, d->n_target, d->hidden, tgt->ctx,
            d->m.eps, s0);
        if (draft_gemm(d, d->fc.d, d->hidden, d->n_target * d->hidden,
                       d->CAT, npos, d->GENC))
            return -1;
        rmsnorm_batch_kernel<<<npos, 256>>>(d->GENC, d->enc_norm, d->GENC,
                                            d->hidden, d->m.eps);
        for (int li = 0; li < d->layers; li++) {
            DLayer * L = &d->layer[li];
            if (draft_gemm(d, L->k.d, d->kvdim, d->hidden, d->GENC, npos, d->K)) return -1;
            if (L->has_v) {
                if (draft_gemm(d, L->v.d, d->kvdim, d->hidden, d->GENC, npos, d->V)) return -1;
            }
            if (L->has_k_norm)
                draft_qk_norm_kernel<<<npos * d->kv_heads, 256>>>(
                    d->K, L->k_norm, npos, d->kv_heads, d->head_dim, d->m.eps);
            if (!L->has_v)
                draft_v_from_k_kernel<<<npos * d->kv_heads, 256>>>(
                    d->V, d->K, npos, d->kv_heads, d->head_dim, d->m.eps);
            const int swa = d->m.swa_pattern[li];
            const int nrot = swa ? d->m.rope_swa_dims : d->m.rope_full_dims;
            const double theta = swa ? d->m.rope_swa_theta : d->m.rope_full_theta;
            draft_rope_kernel<<<npos * d->kv_heads, 128>>>(
                d->K, npos, d->kv_heads, d->head_dim, nrot, theta, s0, 0);
            if (ck(cudaMemcpy2DAsync(d->inj_k[li] + (size_t) s0 * d->kvdim,
                                     (size_t) d->kvdim * 4, d->K, (size_t) d->kvdim * 4,
                                     (size_t) d->kvdim * 4, npos, cudaMemcpyDeviceToDevice),
                   "draft K inject"))
                return -1;
            if (ck(cudaMemcpy2DAsync(d->inj_v[li] + (size_t) s0 * d->kvdim,
                                     (size_t) d->kvdim * 4, d->V, (size_t) d->kvdim * 4,
                                     (size_t) d->kvdim * 4, npos, cudaMemcpyDeviceToDevice),
                   "draft V inject"))
                return -1;
        }
    }
    return 0;
}

/* One batched block-diffusion decode. S blocks share the injected target
 * cache and differ only in their anchor; the row order is [block][row]. The
 * proposals for block s land in proposals[s*n_draft .. +n_draft). */
static int draft_block_batch(Draft * d, Model * tgt, const uint32_t * anchors, int S,
                             int pos0, int n_draft, uint32_t * proposals, float * conf_out,
                             float * logits_host) {
    if (S < 1 || S > d->max_blocks) {
        fprintf(stderr, "draft: %d blocks is outside 1..%d\n", S, d->max_blocks);
        return -1;
    }
    if (n_draft < 1 || n_draft > d->block_size) {
        fprintf(stderr, "draft: n_draft %d is outside the trained block_size %d\n",
                n_draft, d->block_size);
        return -1;
    }
    const int n_block = d->is_dspark ? (d->anchor_mode ? n_draft : n_draft + 1)
                                     : n_draft + 1;
    const int R = S * n_block;
    if (R > d->max_blocks * d->max_rows) {
        fprintf(stderr, "draft: %d rows exceeds the %d-row scratch\n",
                R, d->max_blocks * d->max_rows);
        return -1;
    }
    if (pos0 + n_block > tgt->ctx) {
        fprintf(stderr, "draft: block at %d..%d exceeds the %d-token cache\n",
                pos0, pos0 + n_block - 1, tgt->ctx);
        return -1;
    }
    const int h = d->hidden, hd = d->head_dim, heads = d->heads;
    const int kv_heads = d->kv_heads, qdim = d->qdim, kvdim = d->kvdim;
    const int ffn = d->ffn, vocab = d->vocab;
    const float eps = d->m.eps;

    /* Anchor and mask rows come from the target's tied embedding, the same
     * rows the reference's llm_build_lora_mm reads. */
    for (int s = 0; s < S; s++) {
        float * dst = d->EMB_ANCHOR + (size_t) s * h;
        if (ck(cudaMemsetAsync(dst, 0, (size_t) h * 4), "memset anchor row")) return -1;
        for (int st = 0; st < tgt->embed.nst; st++) {
            if (vqg_row_dequant(tgt->embed.st[st], tgt->embed.dim, tgt->embed.nc,
                                tgt->embed.rows, tgt->embed.cols, tgt->embed.v3,
                                (int) anchors[s], dst)) {
                fprintf(stderr, "draft anchor row: %s\n", vqg_decode_error());
                return -1;
            }
        }
    }
    if (ck(cudaMemsetAsync(d->EMB_MASK, 0, (size_t) h * 4), "memset mask row")) return -1;
    for (int st = 0; st < tgt->embed.nst; st++) {
        if (vqg_row_dequant(tgt->embed.st[st], tgt->embed.dim, tgt->embed.nc,
                            tgt->embed.rows, tgt->embed.cols, tgt->embed.v3,
                            d->mask_token, d->EMB_MASK)) {
            fprintf(stderr, "draft mask row: %s\n", vqg_decode_error());
            return -1;
        }
    }
    draft_fill_block_kernel<<<R, 256>>>(d->X, d->EMB_ANCHOR, d->EMB_MASK, S, n_block, h);

    const int act_kind = strcmp(d->m.act, "silu") ? 0 : 2;
    const int nswa = d->swa_window + (d->is_dspark ? 1 : 0);
    const size_t sh = ((size_t) DSPARK_MAXKEYS + 256) * sizeof(float);
    for (int li = 0; li < d->layers; li++) {
        DLayer * L = &d->layer[li];
        const int swa = d->m.swa_pattern[li];
        const int nrot = swa ? d->m.rope_swa_dims : d->m.rope_full_dims;
        const double theta = swa ? d->m.rope_swa_theta : d->m.rope_full_theta;

        rmsnorm_batch_kernel<<<R, 256>>>(d->X, L->norm, d->XN, h, eps);
        if (draft_gemm(d, L->q.d, qdim, h, d->XN, R, d->Q)) return -1;
        if (draft_gemm(d, L->k.d, kvdim, h, d->XN, R, d->K)) return -1;
        if (L->has_v) {
            if (draft_gemm(d, L->v.d, kvdim, h, d->XN, R, d->V)) return -1;
        }
        if (L->has_q_norm)
            draft_qk_norm_kernel<<<R * heads, 256>>>(d->Q, L->q_norm, R, heads, hd, eps);
        if (L->has_k_norm)
            draft_qk_norm_kernel<<<R * kv_heads, 256>>>(d->K, L->k_norm, R, kv_heads, hd, eps);
        if (!L->has_v)
            draft_v_from_k_kernel<<<R * kv_heads, 256>>>(d->V, d->K, R, kv_heads, hd, eps);
        draft_rope_kernel<<<R * heads, 128>>>(d->Q, R, heads, hd, nrot, theta, pos0, n_block);
        draft_rope_kernel<<<R * kv_heads, 128>>>(d->K, R, kv_heads, hd, nrot, theta, pos0, n_block);
        if (L->has_gate) {
            if (draft_gemm(d, L->gate.d, heads, h, d->XN, R, d->G)) return -1;
        }
        draft_attn_kernel<<<dim3(heads, R), 256, sh>>>(
            d->Q, L->has_gate ? d->G : NULL, d->inj_k[li], d->inj_v[li], pos0,
            d->K, d->V, S, n_block, pos0, heads, kv_heads, hd, nswa, d->causal, d->ATTN);
        if (draft_gemm(d, L->out.d, h, qdim, d->ATTN, R, d->O)) return -1;
        if (L->has_attn_post)
            rmsnorm_batch_kernel<<<R, 256>>>(d->O, L->attn_post, d->O, h, eps);
        draft_add_kernel<<<(R * h + 255) / 256, 256>>>(d->X, d->O, R * h);

        rmsnorm_batch_kernel<<<R, 256>>>(d->X, L->ffn_norm, d->XN, h, eps);
        if (draft_gemm(d, L->ffn_gate.d, ffn, h, d->XN, R, d->FG)) return -1;
        if (draft_gemm(d, L->ffn_up.d, ffn, h, d->XN, R, d->FU)) return -1;
        draft_act_kernel<<<(R * ffn + 255) / 256, 256>>>(d->FG, d->FU, d->ACT, R * ffn, act_kind);
        if (draft_gemm(d, L->ffn_down.d, h, ffn, d->ACT, R, d->O)) return -1;
        draft_add_kernel<<<(R * h + 255) / 256, 256>>>(d->X, d->O, R * h);
        if (L->has_ffn_post)
            rmsnorm_batch_kernel<<<R, 256>>>(d->X, L->ffn_post, d->X, h, eps);
        if (L->has_out_scale)
            draft_scale_kernel<<<(R * h + 255) / 256, 256>>>(d->X, L->out_scale, R * h);
    }

    rmsnorm_batch_kernel<<<R, 256>>>(d->X, d->out_norm, d->H, h, eps);
    for (int r = 0; r < R; r++) {
        if (ck(cudaMemsetAsync(d->LOGITS + (size_t) r * vocab, 0, (size_t) vocab * 4),
               "memset draft logits"))
            return -1;
        if (gemv(&tgt->embed, d->H + (size_t) r * h, d->LOGITS + (size_t) r * vocab)) return -1;
    }
    draft_argmax_batch_kernel<<<R, 1024>>>(d->LOGITS, vocab, d->ROWARG);

    const int i_beg = d->anchor_mode ? 0 : 1;
    if (d->has_markov) {
        const int rank = d->markov_w1.cols;
        for (int s = 0; s < S; s++) {
            uint32_t prev = anchors[s];
            for (int i = i_beg; i < n_block; i++) {
                const int r = s * n_block + i;
                if (ck(cudaMemcpyAsync(d->W1ROW,
                                       d->markov_w1.d + (size_t) prev * rank,
                                       (size_t) rank * 4, cudaMemcpyDeviceToDevice),
                       "markov w1 row"))
                    return -1;
                if (draft_gemm(d, d->markov_w2.d, vocab, rank, d->W1ROW, 1, d->BIAS)) return -1;
                draft_markov_add_kernel<<<(vocab + 255) / 256, 256>>>(
                    d->LOGITS + (size_t) r * vocab, d->BIAS, vocab);
                if (d->has_conf)
                    draft_conf_kernel<<<1, 256>>>(
                        d->H + (size_t) r * h, d->W1ROW, d->conf_proj.d, d->conf_bias,
                        h, rank, d->CONF + r);
                argmax_kernel<<<1, 1024>>>(d->LOGITS + (size_t) r * vocab, vocab,
                                           d->ROWARG + r);
                int best = -1;
                if (ck(cudaMemcpy(&best, d->ROWARG + r, 4, cudaMemcpyDeviceToHost),
                       "markov argmax"))
                    return -1;
                prev = (uint32_t) best;
            }
        }
    }

    int * args = (int *) malloc((size_t) R * sizeof(int));
    if (!args) return -1;
    if (ck(cudaMemcpy(args, d->ROWARG, (size_t) R * sizeof(int), cudaMemcpyDeviceToHost),
           "draft argmax read")) {
        free(args);
        return -1;
    }
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < n_draft; i++) {
            const int r = s * n_block + i_beg + i;
            proposals[(size_t) s * n_draft + i] = (uint32_t) args[r];
            if (conf_out) conf_out[(size_t) s * n_draft + i] = 0.0f;
        }
    }
    if (conf_out && d->has_conf) {
        float * cf = (float *) malloc((size_t) R * sizeof(float));
        if (cf) {
            if (cudaMemcpy(cf, d->CONF, (size_t) R * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                for (int s = 0; s < S; s++)
                    for (int i = 0; i < n_draft; i++)
                        conf_out[(size_t) s * n_draft + i] =
                            cf[(size_t) s * n_block + i_beg + i];
            }
            free(cf);
        }
    }
    if (logits_host) {
        if (ck(cudaMemcpy(logits_host, d->LOGITS, (size_t) R * vocab * 4,
                          cudaMemcpyDeviceToHost), "draft logits read")) {
            free(args);
            return -1;
        }
    }
    free(args);
    return 0;
}

/* Gate 1: batch permutation invariance. Four blocks with different anchors
 * run in one batch, then in a permuted batch. Each block shares the injected
 * target cache but owns its rows, so its logits must not depend on the order.
 * The comparison is bitwise: a f32 GEMM row is a dot product, not a reduction
 * across rows. */
static int draft_gate1(Draft * d, Model * tgt, const uint32_t * ids, int nids) {
    if (nids < 5) {
        fprintf(stderr, "gate1: needs at least five prompt ids\n");
        return -1;
    }
    const int pos0 = nids - 1;
    if (enable_capture(tgt, d->target_layers, d->n_target)) return -1;
    for (int t0 = 0; t0 < pos0; t0 += tgt->chunk) {
        int ntok = pos0 - t0;
        if (ntok > tgt->chunk) ntok = tgt->chunk;
        if (forward_batch(tgt, ids + t0, t0, ntok)) return -1;
    }
    if (ck(cudaDeviceSynchronize(), "gate1 prompt sync")) return -1;
    if (draft_encode(d, tgt, 0, pos0)) return -1;

    const int S = 4;
    const int n_draft = d->block_size;
    const int n_block = d->is_dspark ? (d->anchor_mode ? n_draft : n_draft + 1)
                                     : n_draft + 1;
    const int R = S * n_block;
    if (n_draft > 64) {
        fprintf(stderr, "gate1: block_size %d exceeds the 64-slot proposal buffer\n", n_draft);
        return -1;
    }
    const uint32_t a[4] = { ids[0], ids[1], ids[2], ids[3] };
    const uint32_t b[4] = { ids[3], ids[1], ids[0], ids[2] };
    uint32_t pa[4 * 64], pb[4 * 64];
    float * la = (float *) malloc((size_t) R * d->vocab * 4);
    float * lb = (float *) malloc((size_t) R * d->vocab * 4);
    if (!la || !lb) {
        free(la); free(lb);
        fprintf(stderr, "gate1: cannot allocate the logit copies\n");
        return -1;
    }
    if (draft_block_batch(d, tgt, a, S, pos0, n_draft, pa, NULL, la) ||
        draft_block_batch(d, tgt, b, S, pos0, n_draft, pb, NULL, lb)) {
        free(la); free(lb);
        return -1;
    }

    double maxabs = 0.0;
    int bad_arg = 0, bad_prop = 0;
    for (int i = 0; i < S; i++) {
        int ib = -1;
        for (int j = 0; j < S; j++) if (b[j] == a[i]) ib = j;
        if (ib < 0) continue;
        for (int r = 0; r < n_block; r++) {
            const float * ra = la + ((size_t) i * n_block + r) * d->vocab;
            const float * rb = lb + ((size_t) ib * n_block + r) * d->vocab;
            int ai = 0, bi = 0;
            float av = -INFINITY, bv = -INFINITY;
            for (int v = 0; v < d->vocab; v++) {
                const double diff = fabs((double) ra[v] - (double) rb[v]);
                if (diff > maxabs) maxabs = diff;
                if (ra[v] > av) { av = ra[v]; ai = v; }
                if (rb[v] > bv) { bv = rb[v]; bi = v; }
            }
            if (ai != bi) bad_arg++;
        }
        for (int k = 0; k < n_draft; k++)
            if (pa[i * n_draft + k] != pb[ib * n_draft + k]) bad_prop++;
    }
    printf("gate1       %d blocks, %d rows, maxabs %.9g, argmax mismatches %d, proposal mismatches %d\n",
           S, R, maxabs, bad_arg, bad_prop);
    printf("gate1       %s\n", (maxabs == 0.0 && bad_arg == 0 && bad_prop == 0)
                                  ? "PASS (order-independent, bitwise)"
                                  : "FAIL");
    free(la);
    free(lb);
    return (maxabs == 0.0 && bad_arg == 0 && bad_prop == 0) ? 0 : 1;
}

typedef struct {
    int rounds, drafted, accepted, ng;
    double prompt_s, decode_s, draft_s, verify_s;
} DraftStats;

/* The full propose/verify loop on the GPU target. Greedy only, as phase C's
 * gate: the same ids the CPU draft path produced at temperature 0. */
static int draft_spec_run(Draft * d, Model * tgt, const uint32_t * ids, int nids,
                          int gen, int n_max, int trace, const char * out_path,
                          DraftStats * st) {
    if (nids < 2) {
        fprintf(stderr, "draft: the prompt needs at least two ids\n");
        return -1;
    }
    const int pos0_0 = nids - 1;
    if (pos0_0 >= tgt->ctx) {
        fprintf(stderr, "draft: the prompt is longer than the %d-token cache\n", tgt->ctx);
        return -1;
    }
    if (enable_capture(tgt, d->target_layers, d->n_target)) return -1;

    const double t_prompt = now_s();
    for (int t0 = 0; t0 < pos0_0; t0 += tgt->chunk) {
        int ntok = pos0_0 - t0;
        if (ntok > tgt->chunk) ntok = tgt->chunk;
        if (forward_batch(tgt, ids + t0, t0, ntok)) return -1;
    }
    if (ck(cudaDeviceSynchronize(), "draft prompt sync")) return -1;
    const double prompt_s = now_s() - t_prompt;
    if (draft_encode(d, tgt, 0, pos0_0)) return -1;

    const int n_draft_max = d->anchor_mode ? d->block_size : d->block_size - 1;
    if (n_max <= 0 || n_max > n_draft_max) n_max = n_draft_max;
    if (n_max > 64) {
        fprintf(stderr, "draft: block_size %d exceeds the 64-slot round buffer\n", n_max);
        return -1;
    }
    uint32_t * gen_ids = (uint32_t *) malloc((size_t) (gen > 0 ? gen : 1) * sizeof(uint32_t));
    if (!gen_ids) return -1;

    int pos0 = pos0_0;
    uint32_t anchor = ids[pos0];
    int ng = 0, rounds = 0, drafted = 0, accepted = 0;
    double draft_s = 0.0, verify_s = 0.0;
    const double t_decode = now_s();

    while (ng < gen) {
        int n_draft = n_max;
        if (pos0 + n_draft >= tgt->ctx) {
            n_draft = tgt->ctx - 1 - pos0;
            if (n_draft < 1) break;
            if (n_draft > n_max) n_draft = n_max;
        }
        uint32_t proposals[64];
        const double t0 = now_s();
        if (draft_block_batch(d, tgt, &anchor, 1, pos0, n_draft, proposals, NULL, NULL)) {
            free(gen_ids);
            return -1;
        }
        if (ck(cudaDeviceSynchronize(), "draft block sync")) { free(gen_ids); return -1; }
        draft_s += now_s() - t0;

        uint32_t vids[65];
        vids[0] = anchor;
        for (int i = 0; i < n_draft; i++) vids[i + 1] = proposals[i];
        const double t1 = now_s();
        if (verify_logits(tgt, vids, pos0, n_draft + 1, d->LOGITS)) {
            free(gen_ids);
            return -1;
        }
        draft_argmax_batch_kernel<<<n_draft + 1, 1024>>>(d->LOGITS, d->vocab, d->ROWARG);
        int targ[65];
        if (ck(cudaMemcpy(targ, d->ROWARG, (size_t) (n_draft + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "verify argmax read")) {
            free(gen_ids);
            return -1;
        }
        if (ck(cudaDeviceSynchronize(), "verify sync")) { free(gen_ids); return -1; }
        verify_s += now_s() - t1;

        int k = 0;
        while (k < n_draft && targ[k] == (int) proposals[k]) k++;
        const uint32_t bonus = (uint32_t) targ[k];
        for (int i = 0; i < k && ng < gen; i++) gen_ids[ng++] = proposals[i];
        if (ng < gen) gen_ids[ng++] = bonus;
        drafted += n_draft;
        accepted += k;
        rounds++;
        if (trace) {
            printf("round %d: pos0 %d, draft [", rounds, pos0);
            for (int i = 0; i < n_draft; i++)
                printf("%u%s", proposals[i], i + 1 < n_draft ? " " : "");
            printf("] target [");
            for (int i = 0; i <= n_draft; i++)
                printf("%d%s", targ[i], i < n_draft ? " " : "");
            printf("] accepted %d, bonus %u\n", k, bonus);
        }
        if (draft_encode(d, tgt, pos0, pos0 + k + 1)) { free(gen_ids); return -1; }
        pos0 += k + 1;
        anchor = bonus;
    }
    const double decode_s = now_s() - t_decode;
    if (ng > gen) ng = gen;
    if (st) {
        st->rounds = rounds;
        st->drafted = drafted;
        st->accepted = accepted;
        st->ng = ng;
        st->prompt_s = prompt_s;
        st->decode_s = decode_s;
        st->draft_s = draft_s;
        st->verify_s = verify_s;
    }

    printf("prompt      %d tokens in %.2f s (%.2f tok/s, capture on)\n",
           pos0_0, prompt_s, (double) pos0_0 / prompt_s);
    printf("generated   [");
    for (int i = 0; i < ng; i++) printf("%u%s", gen_ids[i], i + 1 < ng ? ", " : "");
    printf("]\n");
    printf("spec        %d rounds, %d drafted, %d accepted, acceptance %.3f%%\n",
           rounds, drafted, accepted,
           100.0 * (double) accepted / (double) (drafted > 0 ? drafted : 1));
    printf("decode      %.2f s (%.2f tok/s over %d committed tokens)\n",
           decode_s, (double) (accepted + rounds) / decode_s, accepted + rounds);
    printf("timing      draft %.1f ms, verify %.1f ms, total %.2f s\n",
           1000.0 * draft_s, 1000.0 * verify_s, prompt_s + decode_s);
    if (out_path) {
        FILE * f = fopen(out_path, "w");
        if (!f) {
            fprintf(stderr, "%s: cannot write\n", out_path);
            free(gen_ids);
            return -1;
        }
        for (int i = 0; i < ng; i++) fprintf(f, "%u%s", gen_ids[i], i + 1 < ng ? " " : "");
        fprintf(f, "\n");
        fclose(f);
        printf("out         %d ids written to %s\n", ng, out_path);
    }
    free(gen_ids);
    return 0;
}

/* Phase D: the spec path and the plain greedy path in one process, so the
 * break-even is computed from the same weights, prompt and device state. The
 * baseline re-runs the prompt and decodes one token at a time, which is the
 * 70.8 tok/s reference in the handoff. */
static int draft_bench(Draft * d, Model * tgt, const uint32_t * ids, int nids,
                       int gen, int n_max) {
    if (gen <= 0) {
        fprintf(stderr, "bench: --gen must be positive\n");
        return -1;
    }
    DraftStats st;
    memset(&st, 0, sizeof st);
    if (draft_spec_run(d, tgt, ids, nids, gen, n_max, 0, NULL, &st)) return -1;
    if (st.rounds <= 0 || st.decode_s <= 0.0) {
        fprintf(stderr, "bench: the spec path produced no rounds\n");
        return -1;
    }
    tgt->cap_n = 0;  /* the baseline must not pay for capture copies */

    const double t_prompt = now_s();
    int last_ntok = 0;
    for (int t0 = 0; t0 < nids; t0 += tgt->chunk) {
        int ntok = nids - t0;
        if (ntok > tgt->chunk) ntok = tgt->chunk;
        if (forward_batch(tgt, ids + t0, t0, ntok)) return -1;
        last_ntok = ntok;
    }
    if (tail_logits(tgt, tgt->hb + (size_t) (last_ntok - 1) * d->hidden)) return -1;
    if (ck(cudaDeviceSynchronize(), "bench prompt sync")) return -1;
    const double prompt_s = now_s() - t_prompt;
    int next = argmax_dev(tgt);
    if (next < 0) return -1;

    uint32_t * out = (uint32_t *) malloc((size_t) (gen > 0 ? gen : 1) * sizeof(uint32_t));
    if (!out) return -1;
    const double t_base = now_s();
    for (int k = 0; k < gen; k++) {
        out[k] = (uint32_t) next;
        if (forward(tgt, (uint32_t) next, nids + k)) { free(out); return -1; }
        next = argmax_dev(tgt);
        if (next < 0) { free(out); return -1; }
    }
    if (ck(cudaDeviceSynchronize(), "bench decode sync")) { free(out); return -1; }
    const double base_s = now_s() - t_base;
    free(out);

    const double spec_tps = (double) st.ng / st.decode_s;
    const double base_tps = (double) gen / base_s;
    const double round_s = st.decode_s / (double) st.rounds;
    const double t_target = base_s / (double) gen;
    const double avg_draft = (double) st.drafted / (double) st.rounds;
    const double measured = (double) st.accepted / (double) (st.drafted > 0 ? st.drafted : 1);
    double be = (avg_draft > 0.0) ? (((round_s / t_target) - 1.0) / avg_draft) : 0.0;
    if (be < 0.0) be = 0.0;

    printf("bench       acceptance %.3f%% (%d/%d), %.2f accepted and %.2f committed per round\n",
           100.0 * measured, st.accepted, st.drafted, (double) st.accepted / st.rounds,
           (double) st.accepted / st.rounds + 1.0);
    printf("bench       round %.1f ms (draft %.1f ms, verify %.1f ms), spec %.2f tok/s\n",
           1000.0 * round_s, 1000.0 * st.draft_s / st.rounds,
           1000.0 * st.verify_s / st.rounds, spec_tps);
    printf("bench       baseline %.2f tok/s (%.1f ms/token, prompt %.2f s), speedup %.2fx\n",
           base_tps, 1000.0 * t_target, prompt_s, spec_tps / base_tps);
    printf("bench       break-even acceptance %.1f%% per drafted token (measured %.1f%%)\n",
           100.0 * be, 100.0 * measured);
    return 0;
}

#endif /* DSPARK_GPU_CUH */
