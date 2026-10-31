/* stubs.c — the four ggml entry points that ggml-quants.c expects from the rest
 * of ggml, so the vendored quantizer links without libggml.
 *
 * Only the types this repository measures are covered; anything else aborts by
 * name rather than silently returning a wrong size.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ggml_abort at %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    abort();
}

size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_Q8_K: return 292;
        default: ggml_abort(__FILE__, __LINE__, "type_size: unsupported type %d", (int) type);
    }
    return 0;
}

int64_t ggml_blck_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return 256;
        default: ggml_abort(__FILE__, __LINE__, "blck_size: unsupported type %d", (int) type);
    }
    return 0;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    return ggml_type_size(type) * (size_t) (ne / ggml_blck_size(type));
}

const char * ggml_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q2_K: return "q2_K";
        case GGML_TYPE_Q3_K: return "q3_K";
        case GGML_TYPE_Q4_K: return "q4_K";
        case GGML_TYPE_Q5_K: return "q5_K";
        case GGML_TYPE_Q6_K: return "q6_K";
        case GGML_TYPE_Q8_K: return "q8_K";
        default: return "unknown";
    }
}
