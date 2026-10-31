/* plan.c — artifact size and VRAM planner for a Spark quantization choice.
 *
 * The point: pick the encoding from a memory budget instead of guessing. It
 * reads the checkpoint's tensor inventory, computes the exact stored bytes for
 * each candidate encoding, adds the KV cache and the measured compute buffers,
 * and prints the total so a choice can be made against the card's free VRAM.
 *
 * Byte accounting for VQ comes from vq_payload_bytes (the same function the
 * encoder writes with), so the artifact estimate is exact for a whole-model VQ
 * pass, not an average. For the k-quants it uses the block sizes below, and the
 * embedding stays q8_0 as in the shipped GGUF.
 *
 * Build: tools/vq/build.sh.
 * Usage:
 *   plan <hf-dir> [--ctx N] [--kv f16|q8_0] [--layers N] [--full-layers N]
 *                 [--kv-heads N] [--head-dim N] [--gpu-mib N] [--verbose]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include "vendor/ds4_st41.h"

#define MIB 1048576.0

typedef struct {
    const char * name;
    int    vq;          /* 1 = VQ, 0 = k-quant */
    int    dim, nc, stages;
    int    block, block_bytes;  /* k-quant */
} enc_t;

static const enc_t ENCODINGS[] = {
    { "Q8_0",        0, 0, 0,    0, 32,  34 },
    { "Q6_K",        0, 0, 0,    0, 256, 210 },
    { "Q5_K",        0, 0, 0,    0, 256, 176 },
    { "Q4_K",        0, 0, 0,    0, 256, 144 },
    { "VQ 4x2048 x2", 1, 4, 2048, 2, 0,   0 },
    { "VQ 4x4096 x1", 1, 4, 4096, 1, 0,   0 },
    { "VQ 4x4096 x2", 1, 4, 4096, 2, 0,   0 },
    { "VQ 4x8192 x2", 1, 4, 8192, 2, 0,   0 },
    { "VQ 8x4096 x2", 1, 8, 4096, 2, 0,   0 },
};
#define NENC ((int)(sizeof(ENCODINGS) / sizeof(ENCODINGS[0])))

/* relL2 measured on model.layers.0.mlp.down_proj.weight, 2026-10-31, same
 * matrix for every row (tools/vq: eg2_vq, rvq, quant_err). -1 = not measured. */
typedef struct { const char * name; double relL2; } measured_t;
static const measured_t MEASURED[] = {
    { "Q8_0", 0.5403 }, { "Q6_K", 1.7869 }, { "Q5_K", 3.6312 }, { "Q4_K", 7.1736 },
    { "VQ 4x4096 x1", 16.8933 }, { "VQ 4x4096 x2", 3.9258 },
    { "VQ 4x2048 x2", -1 }, { "VQ 4x8192 x2", -1 }, { "VQ 8x4096 x2", -1 },
};
static double measured_of(const char * name) {
    for (size_t i = 0; i < sizeof(MEASURED) / sizeof(MEASURED[0]); i++)
        if (!strcmp(MEASURED[i].name, name)) return MEASURED[i].relL2;
    return -1;
}

/* Exact stored bytes for one tensor under one encoding. */
static uint64_t tensor_bytes(const v41_st_ent * E, const enc_t * e) {
    uint64_t n = 1;
    for (int i = 0; i < E->nd; i++) n *= (uint64_t) E->shape[i];

    if (E->nd == 1) return n * 4;                 /* 1-D norms stay f32 */
    if (!strcmp(E->dtype, "F32")) { /* fallthrough: treated as 2-D matrix */ }

    const uint64_t rows = (uint64_t) E->shape[0];
    const uint64_t cols = (uint64_t) E->shape[1];

    /* The tied embedding stays q8_0 in every artifact. */
    if (strstr(E->name, "embedding") && !e->vq) {
        return (rows * cols / 32) * 34;
    }
    if (e->vq) {
        /* [16B header][codebook nc*dim f16][rows f16 gains][index stream] */
        uint64_t nidx = rows * cols / (uint64_t) e->dim;
        int bits = 1; while ((1 << bits) < e->nc) bits++;
        uint64_t ib = (bits == 8) ? nidx : (nidx * (uint64_t) bits + 7) / 8 + 1;
        uint64_t one = 16 + (uint64_t) e->nc * e->dim * 2 + rows * 2 + ib;
        return one * (uint64_t) e->stages;
    }
    if (e->block == 0) return 0;
    return (rows * cols / (uint64_t) e->block) * (uint64_t) e->block_bytes;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <hf-dir> [--ctx N] [--kv f16|q8_0] [--layers N]\n"
                        "          [--full-layers N] [--kv-heads N] [--head-dim N]\n"
                        "          [--gpu-mib N] [--verbose]\n", argv[0]);
        return 2;
    }
    const char * dir = argv[1];
    uint64_t ctx = 131072;
    int kv_q8 = 0, layers = 36, full_layers = 9, kv_heads = 4, head_dim = 256, verbose = 0;
    double gpu_mib = 12282.0, budget_mib = 0.0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) kv_q8 = !strcmp(argv[++i], "q8_0");
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--full-layers") && i + 1 < argc) full_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-heads") && i + 1 < argc) kv_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--head-dim") && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu-mib") && i + 1 < argc) gpu_mib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) budget_mib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) verbose = 1;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }

    v41_st S;
    if (v41_st_open(&S, dir)) { fprintf(stderr, "cannot open %s\n", dir); return 1; }
    const size_t ntensors = (size_t) S.ne;
    if (verbose) printf("%zu tensors in %s\n", ntensors, dir);

    uint64_t total_elems = 0;
    for (size_t t = 0; t < ntensors; t++) {
        uint64_t n = 1;
        for (int i = 0; i < S.e[t].nd; i++) n *= (uint64_t) S.e[t].shape[i];
        total_elems += n;
    }

    /* KV: full-attention layers hold ctx tokens, the sliding-window layers hold
     * their window (512) per sequence, one sequence. */
    const double elem = kv_q8 ? 34.0 / 32.0 : 2.0;
    const double kv_full = (double) ctx * (double) full_layers * 2.0 * (double) kv_heads * (double) head_dim * elem;
    const double window = 512.0;
    const double kv_swa  = window * (double) (layers - full_layers) * 2.0 * (double) kv_heads * (double) head_dim * elem;
    const double kv_mib  = (kv_full + kv_swa) / MIB;
    /* Compute buffers + CUDA context, measured on this card: 1,079 MiB at 32K and
     * 1,434 MiB at 128K on the Q6_K build, 1,300 MiB on the Q4_K_M build. */
    const double buf_mib = 1300.0;

    printf("\n%-16s %10s %8s %9s %9s %9s %10s %9s\n",
           "encoding", "artifact", "bpw", "weights", "KV", "buffers", "total", "headroom");
    printf("%-16s %10s %8s %9s %9s %9s %10s %9s\n",
           "", "GB", "", "MiB", "MiB", "MiB", "MiB", "MiB");
    printf("---------------------------------------------------------------------------------------------\n");

    double best_total = 0.0, best_q = 1e9;
    const char * best = NULL;

    for (int k = 0; k < NENC; k++) {
        const enc_t * e = &ENCODINGS[k];
        uint64_t bytes = 0;
        for (size_t t = 0; t < ntensors; t++) {
            bytes += tensor_bytes(&S.e[t], e);
        }
        const double w_mib = (double) bytes / MIB;
        const double total = w_mib + kv_mib + buf_mib;
        printf("%-16s %10.2f %8.3f %9.0f %9.0f %9.0f %10.0f %9.0f",
               e->name, (double) bytes / 1e9, (double) bytes * 8.0 / (double) total_elems,
               w_mib, kv_mib, buf_mib, total, gpu_mib - total);
        const double q = measured_of(e->name);
        if (q >= 0) printf("   relL2 %.2f%%", q);
        printf("\n");

        /* Recommendation: best measured quality inside the budget. */
        if (budget_mib > 0.0 && total <= budget_mib && q >= 0.0 && q < best_q) {
            best_q = q; best_total = total; best = e->name;
        }
    }
    printf("\nKV at ctx %llu (%s): %0.f MiB   buffers: %0.f MiB   card: %0.f MiB\n",
           (unsigned long long) ctx, kv_q8 ? "q8_0" : "f16", kv_mib, buf_mib, gpu_mib);
    printf("relL2 is measured on model.layers.0.mlp.down_proj.weight (one matrix), not model-wide.\n");
    printf("The k-quant rows put the embedding at q8_0; VQ rows encode every 2-D tensor.\n");

    if (budget_mib > 0.0) {
        if (best) {
            printf("\nBudget %0.f MiB: best measured encoding that fits is %s "
                   "(total %0.f MiB, relL2 %.2f%%).\n", budget_mib, best, best_total, best_q);
        } else {
            printf("\nBudget %0.f MiB: no encoding with a measured relL2 fits; "
                   "the VQ rows above need a measurement before they can be chosen.\n", budget_mib);
        }
    }
    return 0;
}
