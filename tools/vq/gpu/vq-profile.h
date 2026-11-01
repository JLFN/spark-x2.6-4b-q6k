/* vq-profile.h — the per-artifact tuning profile produced by vq-tune.
 *
 * A profile is the measured answer to "which kernel, at which shape, for this
 * matrix geometry on this device". It is optional by construction: the model
 * runtime and the benchmark harness keep their readiness heuristic and only
 * substitute a profile entry when the file exists, parses, carries the same
 * format version, names the same device, and matches the artifact's
 * fingerprint.
 *
 * Format, one directive per line, plain ASCII, no quoting:
 *
 *   VQPROF 1
 *   artifact <16 lowercase hex digits>
 *   gpu <device name, to end of line>
 *   driver <driver text, to end of line>
 *   gemv <dim> <nc> <rows> <cols> <v3> <variant> <threads> <gbs>
 *   prefill <dim> <nc> <rows> <cols> <v3> <variant> <bn> <bk> <tps>
 *
 * The artifact fingerprint is FNV-1a 64 over, in this order:
 *   the manifest.txt bytes as stored,
 *   then for every matrix line of the manifest in file order (stages > 0):
 *     the tensor name bytes, the file size as 8 little-endian bytes,
 *     the first min(4096, size) bytes of the container, and
 *     the last  min(4096, size) bytes of the container.
 * The tuner and this reader must hash exactly that sequence; the gate compares
 * both values before a profile is trusted. This is a staleness check, not a
 * cryptographic digest.
 *
 * The device name is matched literally: tile and launch choices are properties
 * of the GPU that measured them, so a profile from another device is refused
 * rather than applied blind.
 */
#ifndef VQ_PROFILE_H
#define VQ_PROFILE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VQPROF_VERSION 1
#define VQPROF_MAX_ENTRIES 512

typedef struct {
    int kind;      /* 0 = gemv, 1 = prefill */
    int dim, nc, rows, cols, v3;
    int variant;
    int threads;   /* gemv only */
    int bn, bk;    /* prefill only */
} vqprof_entry;

typedef struct {
    unsigned long long artifact;
    char gpu[128];
    char driver[64];
    int n;
    vqprof_entry e[VQPROF_MAX_ENTRIES];
} vqprof;

static inline uint64_t vqprof_fnv(uint64_t h, const void * p, size_t n) {
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; i++) { h ^= (uint64_t) b[i]; h *= 1099511628211ull; }
    return h;
}

/* FNV offset basis, exposed so both sides start from the same constant. */
static inline uint64_t vqprof_fnv_init(void) { return 1469598103934665603ull; }

static inline int vqprof_read_all(const char * path, unsigned char ** out, size_t * n) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) || (*n = (size_t) ftell(f), fseek(f, 0, SEEK_SET))) { fclose(f); return -1; }
    unsigned char * b = (unsigned char *) malloc(*n ? *n : 1);
    if (!b) { fclose(f); return -1; }
    const size_t got = fread(b, 1, *n, f);
    fclose(f);
    if (got != *n) { free(b); return -1; }
    *out = b;
    return 0;
}

/* Hash one container file: name, size, and the two 4 KB edges. Returns 0 on
 * success; a missing file is an error, because a manifest entry that cannot be
 * read makes the fingerprint meaningless. */
static inline int vqprof_hash_file(uint64_t * h, const char * dir, const char * name) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return -1; }
    const long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return -1; }
    const size_t size = (size_t) sz;
    *h = vqprof_fnv(*h, name, strlen(name));
    unsigned char le[8];
    for (int i = 0; i < 8; i++) le[i] = (unsigned char) ((uint64_t) size >> (8 * i));
    *h = vqprof_fnv(*h, le, 8);
    const size_t edge = size < 4096 ? size : 4096;
    unsigned char * buf = (unsigned char *) malloc(edge ? edge : 1);
    if (!buf) { fclose(f); return -1; }
    if (edge) {
        if (fread(buf, 1, edge, f) != edge) { free(buf); fclose(f); return -1; }
        *h = vqprof_fnv(*h, buf, edge);
        if (size > 4096 && fseek(f, (long) (size - 4096), SEEK_SET)) { free(buf); fclose(f); return -1; }
        if (size > 4096 && fread(buf, 1, 4096, f) != 4096) { free(buf); fclose(f); return -1; }
        *h = vqprof_fnv(*h, buf, edge);
    }
    free(buf);
    fclose(f);
    return 0;
}

/* Compute the fingerprint of the artifact directory. Reads manifest.txt, then
 * every matrix container it lists. */
static inline int vqprof_artifact(const char * dir, unsigned long long * out) {
    char path[4096];
    snprintf(path, sizeof path, "%s/manifest.txt", dir);
    unsigned char * man = NULL;
    size_t mn = 0;
    if (vqprof_read_all(path, &man, &mn)) return -1;
    uint64_t h = vqprof_fnv_init();
    h = vqprof_fnv(h, man, mn);

    /* The manifest is a text file: walk it in place, one line at a time. */
    char * p = (char *) man;
    char * end = p + mn;
    while (p < end) {
        char * nl = (char *) memchr(p, '\n', (size_t) (end - p));
        char * line = p;
        if (nl) *nl = '\0';
        if (line[0] != '#' && line[0] != '\0') {
            char name[1024];
            unsigned long long rows, cols, stages, bytes;
            double rel;
            if (sscanf(line, "%1023s %llu %llu %llu %llu %lf", name, &rows, &cols, &stages, &bytes, &rel) == 6
                && stages > 0) {
                if (vqprof_hash_file(&h, dir, name)) { free(man); return -1; }
            }
        }
        if (!nl) break;
        p = nl + 1;
    }
    free(man);
    *out = h;
    return 0;
}

static inline char * vqprof_trim(char * s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) s[--n] = '\0';
    return s;
}

/* Load a profile. Returns 0 when the file parsed and its version matched;
 * every later trust decision (artifact, device) is the caller's. */
static inline int vqprof_load(const char * path, vqprof * pf) {
    memset(pf, 0, sizeof *pf);
    FILE * f = fopen(path, "r");
    if (!f) return -1;
    char line[512];
    int version = 0;
    while (fgets(line, sizeof line, f)) {
        vqprof_trim(line);
        if (!strncmp(line, "VQPROF ", 7)) {
            version = atoi(line + 7);
            continue;
        }
        if (!strncmp(line, "artifact ", 9)) {
            unsigned long long v = 0;
            if (sscanf(line + 9, "%llx", &v) == 1) pf->artifact = v;
            continue;
        }
        if (!strncmp(line, "gpu ", 4)) { snprintf(pf->gpu, sizeof pf->gpu, "%s", line + 4); continue; }
        if (!strncmp(line, "driver ", 7)) { snprintf(pf->driver, sizeof pf->driver, "%s", line + 7); continue; }
        vqprof_entry e;
        memset(&e, 0, sizeof e);
        if (sscanf(line, "gemv %d %d %d %d %d %d %d",
                   &e.dim, &e.nc, &e.rows, &e.cols, &e.v3, &e.variant, &e.threads) == 7) {
            e.kind = 0;
        } else if (sscanf(line, "prefill %d %d %d %d %d %d %d %d",
                          &e.dim, &e.nc, &e.rows, &e.cols, &e.v3, &e.variant, &e.bn, &e.bk) == 8) {
            e.kind = 1;
        } else {
            continue;
        }
        if (pf->n < VQPROF_MAX_ENTRIES) pf->e[pf->n++] = e;
    }
    fclose(f);
    return version == VQPROF_VERSION ? 0 : -1;
}

static inline const vqprof_entry * vqprof_find(const vqprof * pf, int kind,
                                               int dim, int nc, int rows, int cols, int v3) {
    for (int i = 0; i < pf->n; i++) {
        const vqprof_entry * e = &pf->e[i];
        if (e->kind == kind && e->dim == dim && e->nc == nc && e->rows == rows
            && e->cols == cols && e->v3 == v3) return e;
    }
    return NULL;
}

#endif /* VQ_PROFILE_H */
