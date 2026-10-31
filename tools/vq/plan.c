/* plan.c — artifact size and VRAM planner for a Spark encoding choice.
 *
 * Answers one question: if this model is converted with this encoding, how much
 * disk and how much VRAM will it take? It reads the checkpoint's tensor
 * inventory, computes exact stored bytes per tensor, adds the KV cache at the
 * chosen context and the compute buffers measured on this card, and prints the
 * total against the card's memory.
 *
 * VQ byte accounting uses the same layout the encoder writes
 * ([16B header][codebook nc*dim f16][rows f16 gains][index stream]), so a VQ
 * row is what a real conversion would produce, not an average.
 *
 *   plan <hf-dir>                       table of standard encodings
 *   plan <hf-dir> --vq 4 4096 2         add a custom VQ row (dim nc stages)
 *   plan <hf-dir> --vq 4 4096 2 --vq-embed   encode the tied embedding too
 *   plan <hf-dir> --draft models/dspark/Spark-X2.6-4B-DSpark.gguf
 *   plan <hf-dir> --ctx 65536 --kv q8_0 --budget 9000
 *   plan <hf-dir> --json                machine-readable rows
 *
 * Build: tools/vq/build.sh.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include "vendor/ds4_st41.h"

#define MIB 1048576.0

typedef struct {
    char   name[48];
    int    vq;
    int    dim, nc, stages;
    int    block, block_bytes;
    int    emb;        /* 0 = q8_0, 1 = f16, 2 = f32, 3 = VQ (vq rows only) */
} enc_t;

typedef struct {
    double artifact_gb, bpw, weights_mib, kv_mib, buf_mib, total_mib, headroom_mib;
    double relL2;      /* -1 when not measured */
} row_t;

static double measured_of(const char * name) {
    /* relL2 measured on model.layers.0.mlp.down_proj.weight on 2026-10-31 with
     * the tools in this repository (eg2_vq, rvq, quant_err). */
    if (!strcmp(name, "Q8_0"))         return 0.5403;
    if (!strcmp(name, "Q6_K"))         return 1.7869;
    if (!strcmp(name, "Q5_K"))         return 3.6312;
    if (!strcmp(name, "Q4_K"))         return 7.1736;
    if (!strcmp(name, "VQ 4x2048 x2")) return 4.8828;
    if (!strcmp(name, "VQ 4x4096 x1")) return 16.8933;
    if (!strcmp(name, "VQ 4x4096 x2")) return 3.9258;
    return -1.0;
}

static uint64_t vq_bytes(uint64_t rows, uint64_t cols, int dim, int nc) {
    const uint64_t nidx = rows * cols / (uint64_t) dim;
    int bits = 1; while ((1 << bits) < nc) bits++;
    const uint64_t ib = (bits == 8) ? nidx : (nidx * (uint64_t) bits + 7) / 8 + 1;
    return 16 + (uint64_t) nc * (uint64_t) dim * 2 + rows * 2 + ib;
}

static uint64_t tensor_bytes(const v41_st_ent * E, const enc_t * e) {
    uint64_t n = 1;
    for (int i = 0; i < E->nd; i++) n *= (uint64_t) E->shape[i];

    if (E->nd == 1) return n * 4;                 /* 1-D norms stay f32 */

    const uint64_t rows = (uint64_t) E->shape[0];
    const uint64_t cols = (uint64_t) E->shape[1];
    const int is_emb = strstr(E->name, "embedding") != NULL;

    if (is_emb) {
        if (e->vq && e->emb == 3) return vq_bytes(rows, cols, e->dim, e->nc) * (uint64_t) e->stages;
        if (e->emb == 0) return (rows * cols / 32) * 34;
        if (e->emb == 1) return rows * cols * 2;
        return rows * cols * 4;
    }
    if (e->vq) return vq_bytes(rows, cols, e->dim, e->nc) * (uint64_t) e->stages;
    if (e->block == 0) return 0;
    return (rows * cols / (uint64_t) e->block) * (uint64_t) e->block_bytes;
}

static void print_row(const char * label, const row_t * r, int json_first) {    if (json_first) {
        printf("{\"encoding\": \"%s\", \"artifact_gb\": %.3f, \"bpw\": %.3f, "
               "\"weights_mib\": %.0f, \"kv_mib\": %.0f, \"buffers_mib\": %.0f, "
               "\"total_mib\": %.0f, \"headroom_mib\": %.0f, \"relL2\": %.4f}",
               label, r->artifact_gb, r->bpw, r->weights_mib, r->kv_mib,
               r->buf_mib, r->total_mib, r->headroom_mib, r->relL2);
        return;
    }
    printf("%-16s %10.2f %8.3f %9.0f %9.0f %9.0f %10.0f %9.0f",
           label, r->artifact_gb, r->bpw, r->weights_mib, r->kv_mib,
           r->buf_mib, r->total_mib, r->headroom_mib);
    if (r->relL2 >= 0) printf("   relL2 %.2f%%", r->relL2);
    printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <hf-dir> [options]\n"
            "  --vq DIM NC STAGES   add a custom VQ row\n"
            "  --vq-embed           encode the tied embedding with VQ too\n"
            "  --emb q8_0|f16|f32   embedding storage for the k-quant rows (default q8_0)\n"
            "  --ctx N              context tokens (default 131072)\n"
            "  --kv f16|q8_0        KV cache type (default f16)\n"
            "  --kv-bytes N         override KV with N bytes per token, whole model\n"
            "                       (ds4 V4.1 uses about 890, A.X-K2 about 47,000)\n"
            "  --layers N --full-layers N --kv-heads N --head-dim N\n"
            "  --draft PATH         also load this draft model\n"
            "  --gpu-mib N          card memory (default 12282)\n"
            "  --budget N           print the best measured encoding that fits\n"
            "  --json               machine-readable output\n", argv[0]);
        return 2;
    }
    const char * dir = argv[1];
    /* Default context is the checkpoint's own limit, read from config.json
     * (max_position_embeddings). Spark-X2.6-4B declares 1,048,576; serving it at
     * a smaller window is a VRAM choice, not a model limit. */
    uint64_t ctx = 0;
    {
        char path[4096];
        snprintf(path, sizeof path, "%s/config.json", dir);
        FILE * cf = fopen(path, "rb");
        if (cf) {
            char buf[65536];
            size_t n = fread(buf, 1, sizeof buf - 1, cf);
            buf[n] = 0;
            fclose(cf);
            const char * p = strstr(buf, "\"max_position_embeddings\"");
            if (p) {
                p = strchr(p, ':');
                if (p) ctx = strtoull(p + 1, NULL, 10);
            }
        }
    }
    if (ctx == 0) ctx = 131072;   /* fallback when config.json is absent */
    int kv_q8 = 0, layers = 36, full_layers = 9, kv_heads = 4, head_dim = 256;
    double kv_bytes_per_token = 0.0;   /* >0 overrides the computed KV */
    int vq_n = 0, vq_embed = 0, json = 0;
    int vq_dim[8] = {0}, vq_nc[8] = {0}, vq_stages[8] = {0};
    int emb = 0;
    double gpu_mib = 12282.0, budget_mib = 0.0, draft_mib = 0.0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) kv_q8 = !strcmp(argv[++i], "q8_0");
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--full-layers") && i + 1 < argc) full_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-heads") && i + 1 < argc) kv_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--head-dim") && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-bytes") && i + 1 < argc) kv_bytes_per_token = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gpu-mib") && i + 1 < argc) gpu_mib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) budget_mib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--emb") && i + 1 < argc) {
            const char * v = argv[++i];
            if (!strcmp(v, "q8_0")) emb = 0; else if (!strcmp(v, "f16")) emb = 1;
            else if (!strcmp(v, "f32")) emb = 2; else { fprintf(stderr, "bad --emb\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--draft") && i + 1 < argc) {
            const char * p = argv[++i];
            FILE * f = fopen(p, "rb");
            if (!f) { fprintf(stderr, "cannot open draft %s\n", p); return 1; }
            fseek(f, 0, SEEK_END);
            draft_mib = (double) ftell(f) / MIB + 514.0;  /* + measured draft KV/buffers at 128K */
            fclose(f);
        }
        else if (!strcmp(argv[i], "--vq") && i + 3 < argc) {
            if (vq_n >= 8) { fprintf(stderr, "at most 8 --vq specs\n"); return 2; }
            vq_dim[vq_n] = atoi(argv[++i]); vq_nc[vq_n] = atoi(argv[++i]); vq_stages[vq_n] = atoi(argv[++i]);
            vq_n++;
        }
        else if (!strcmp(argv[i], "--vq-embed")) vq_embed = 1;
        else if (!strcmp(argv[i], "--json")) json = 1;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    for (int i = 0; i < vq_n; i++) {
        if (vq_dim[i] <= 0 || vq_nc[i] <= 1 || vq_stages[i] < 1) {
            fprintf(stderr, "--vq needs dim nc stages, all positive\n");
            return 2;
        }
    }

    v41_st S;
    if (v41_st_open(&S, dir)) { fprintf(stderr, "cannot open %s\n", dir); return 1; }
    const size_t ntensors = (size_t) S.ne;
    uint64_t total_elems = 0;
    for (size_t t = 0; t < ntensors; t++) {
        uint64_t n = 1;
        for (int i = 0; i < S.e[t].nd; i++) n *= (uint64_t) S.e[t].shape[i];
        total_elems += n;
    }

    /* KV: full-attention layers hold ctx tokens; the sliding-window layers hold
     * their 512-token window. One sequence. --kv-bytes N overrides this with a
     * measured per-token figure from a runtime that packs its KV (for example a
     * ds4 family), because that cost is a property of the runtime, not of the
     * weight encoding. */
    const double elem = kv_q8 ? 34.0 / 32.0 : 2.0;
    double kv_mib;
    if (kv_bytes_per_token > 0.0) {
        kv_mib = (double) ctx * kv_bytes_per_token / MIB;
    } else {
        kv_mib = ((double) ctx * (double) full_layers * 2.0 * (double) kv_heads * (double) head_dim * elem
                + 512.0 * (double) (layers - full_layers) * 2.0 * (double) kv_heads * (double) head_dim * elem) / MIB;
    }
    /* Compute buffers + CUDA context, measured on this card: 1,079 MiB at 32K and
     * 1,434 MiB at 128K on the Q6_K build. */
    const double buf_mib = 1300.0;

    enc_t encs[16];
    int nenc = 0;
    for (int i = 0; i < vq_n; i++) {
        snprintf(encs[nenc].name, sizeof encs[nenc].name, "VQ %dx%d x%d%s",
                 vq_dim[i], vq_nc[i], vq_stages[i], vq_embed ? " emb" : "");
        encs[nenc].vq = 1; encs[nenc].dim = vq_dim[i]; encs[nenc].nc = vq_nc[i];
        encs[nenc].stages = vq_stages[i]; encs[nenc].block = 0; encs[nenc].block_bytes = 0;
        encs[nenc].emb = vq_embed ? 3 : emb;
        nenc++;
    }
    struct { const char * n; int block, bytes; } kq[] = {
        { "Q8_0", 32, 34 }, { "Q6_K", 256, 210 }, { "Q5_K", 256, 176 }, { "Q4_K", 256, 144 },
    };
    for (size_t i = 0; i < sizeof kq / sizeof kq[0]; i++) {
        snprintf(encs[nenc].name, sizeof encs[nenc].name, "%s", kq[i].n);
        encs[nenc].vq = 0; encs[nenc].dim = encs[nenc].nc = encs[nenc].stages = 0;
        encs[nenc].block = kq[i].block; encs[nenc].block_bytes = kq[i].bytes; encs[nenc].emb = emb;
        nenc++;
    }

    double best_total = 0.0, best_q = 1e9;
    const char * best = NULL;

    if (!json) {
        printf("\n%-16s %10s %8s %9s %9s %9s %10s %9s\n",
               "encoding", "artifact", "bpw", "weights", "KV", "buffers", "total", "headroom");
        printf("%-16s %10s %8s %9s %9s %9s %10s %9s\n",
               "", "GB", "", "MiB", "MiB", "MiB", "MiB", "MiB");
        printf("---------------------------------------------------------------------------------------------\n");
    } else {
        printf("[");
    }

    for (int k = 0; k < nenc; k++) {
        const enc_t * e = &encs[k];
        uint64_t bytes = 0;
        for (size_t t = 0; t < ntensors; t++) bytes += tensor_bytes(&S.e[t], e);

        row_t r;
        r.artifact_gb = (double) bytes / 1e9;
        r.bpw         = (double) bytes * 8.0 / (double) total_elems;
        r.weights_mib = (double) bytes / MIB + draft_mib;
        r.kv_mib      = kv_mib + (draft_mib > 0.0 ? 0.0 : 0.0); /* draft KV is in draft_mib */
        r.buf_mib     = buf_mib;
        r.total_mib   = r.weights_mib + r.kv_mib + r.buf_mib;
        r.headroom_mib = gpu_mib - r.total_mib;
        r.relL2       = measured_of(e->name);

        print_row(e->name, &r, json);
        if (json) printf(k + 1 < nenc ? ", " : "");

        if (budget_mib > 0.0 && r.total_mib <= budget_mib && r.relL2 >= 0.0 && r.relL2 < best_q) {
            best_q = r.relL2; best_total = r.total_mib; best = e->name;
        }
    }

    if (json) {
        printf("]\n");
        return 0;
    }

    printf("\nKV at ctx %llu (%s): %.0f MiB   buffers: %.0f MiB   card: %.0f MiB",
           (unsigned long long) ctx, kv_q8 ? "q8_0" : "f16", kv_mib, buf_mib, gpu_mib);
    if (draft_mib > 0.0) printf("   draft: %.0f MiB", draft_mib);
    printf("\nrelL2 is measured on model.layers.0.mlp.down_proj.weight (one matrix), not model-wide.\n");
    printf("k-quant rows store the tied embedding as %s; a VQ row encodes 2-D tensors unless --vq-embed.\n",
           emb == 0 ? "q8_0" : emb == 1 ? "f16" : "f32");

    if (budget_mib > 0.0) {
        if (best) printf("\nBudget %.0f MiB: best measured encoding that fits is %s (total %.0f MiB, relL2 %.2f%%).\n",
                         budget_mib, best, best_total, best_q);
        else printf("\nBudget %.0f MiB: nothing with a measured relL2 fits.\n", budget_mib);
    }
    return 0;
}
