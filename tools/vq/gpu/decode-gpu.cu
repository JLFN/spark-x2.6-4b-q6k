/* decode-gpu.cu — CUDA decode and GEMV for one RQVQ VQ container.
 *
 * Two kernels, both reading the payload exactly as vq_unpack_dequant reads it
 * (vendor/vq_qc.h): 16-byte DQVQ header, f16 codebook nc*dim, f16 gain per row,
 * then ceil(log2 nc)-bit little-endian indices (one byte each at 8 bits).
 *
 *   vqg_dequant  rebuilds the matrix in f32, one thread per dim-wide group
 *   vqg_gemv     fuses the gather and the dot product, one block per row
 *
 * The point of the pair is to price the two shapes a runtime can choose: pay
 * once to expand the matrix (then read 4x the bytes per token), or pay the
 * gather on every token. The host driver times both with CUDA events.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static char g_err[256];

extern "C" const char * vqg_decode_error(void) { return g_err; }

static int fail(const char * what, cudaError_t e) {
    snprintf(g_err, sizeof g_err, "%s: %s", what, cudaGetErrorString(e));
    return -1;
}

/* Payload fields as the kernels need them. The caller parses the stage header on
 * the host; nothing here may dereference `payload` on the host, because it is a
 * device pointer. */
struct layout {
    int dim, nc, rows, cols, nbit, v3;
    unsigned mask;
    size_t cb_off, gain_off, stream_off, nidx;
    size_t mrow, prow;   /* DQV3: bytes per row of the 12-bit main stream and of the plane */
};

static layout make_layout(int dim, int nc, int rows, int cols, int v3) {
    layout L;
    L.dim = dim; L.nc = nc; L.rows = rows; L.cols = cols; L.v3 = v3;
    L.nbit = 1;
    while ((1u << L.nbit) < (unsigned) nc) L.nbit++;
    L.mask = (1u << L.nbit) - 1u;
    const size_t nidx_row = (size_t) cols / dim;
    const int mb = L.nbit > 12 ? 12 : L.nbit;
    L.cb_off = v3 ? 24 : 16;
    L.gain_off = L.cb_off + (size_t) nc * dim * 2;
    L.stream_off = L.gain_off + (size_t) rows * 2;
    L.nidx = (size_t) rows * cols / dim;
    L.mrow = v3 ? (nidx_row * (size_t) mb + 7) / 8 : 0;
    L.prow = (v3 && L.nbit > 12) ? (nidx_row + 7) / 8 : 0;
    return L;
}

__device__ __forceinline__ uint32_t rd_idx(const uint8_t * s, size_t i, int nbit, uint32_t mask) {
    if (nbit == 8) return s[i];
    const size_t bit = i * (size_t) nbit;
    const uint32_t w = (uint32_t) s[bit >> 3]
                     | ((uint32_t) s[(bit >> 3) + 1] << 8)
                     | ((uint32_t) s[(bit >> 3) + 2] << 16);
    return (w >> (bit & 7)) & mask;
}

/* Format-aware index read: DQVQ numbers bits across the whole matrix, DQV3
 * restarts the 12-bit stream at every row and carries the 13th bit in a plane
 * (ds41_vq_row.cuh:7-20, :122-137). */
__device__ __forceinline__ uint32_t rd_idx_fmt(const uint8_t * st, const layout & L, size_t r, size_t i) {
    if (!L.v3) return rd_idx(st, r * (size_t) (L.cols / L.dim) + i, L.nbit, L.mask);
    const uint8_t * base = st + r * L.mrow;
    const size_t bit = i * 12;
    const uint8_t * q = base + (bit >> 3);
    const uint32_t w = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16) | ((uint32_t) q[3] << 24);
    uint32_t v = (w >> (bit & 7)) & 0xFFFu;
    if (L.prow) {
        const uint8_t * pl = st + L.mrow * (size_t) L.rows + r * L.prow;
        v |= (uint32_t) ((pl[i >> 3] >> (i & 7)) & 1u) << 12;
    }
    return v;
}

/* One thread per group of `dim` columns: write the dequantised values. */
__global__ void dequant_kernel(const uint8_t * p, const layout L, float * out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L.nidx) return;
    const __half * cb = (const __half *) (p + L.cb_off);
    const __half * gr = (const __half *) (p + L.gain_off);
    const uint8_t * st = p + L.stream_off;
    const uint32_t v = rd_idx_fmt(st, L, (i * (size_t) L.dim) / (size_t) L.cols,
                                  i - ((i * (size_t) L.dim) / (size_t) L.cols) * (size_t) (L.cols / L.dim));
    const size_t r = (i * (size_t) L.dim) / (size_t) L.cols;
    const float g = __half2float(gr[r]);
    const __half * c = cb + (size_t) v * L.dim;
    float * o = out + i * (size_t) L.dim;
    if (L.dim == 4) {
        float4 q;
        q.x = __half2float(c[0]) * g;
        q.y = __half2float(c[1]) * g;
        q.z = __half2float(c[2]) * g;
        q.w = __half2float(c[3]) * g;
        float4 * dst = (float4 *) o;
        dst->x += q.x;
        dst->y += q.y;
        dst->z += q.z;
        dst->w += q.w;
    } else {
        for (int d = 0; d < L.dim; d++) o[d] += __half2float(c[d]) * g;
    }
}

/* dim-4 fast path: one 8-byte codebook load and one half2 pair of conversions
 * per index, unrolled to hide L1 gather latency. */
#define GEMV_UNROLL 4
__global__ void gemv_warp4_kernel(const uint8_t * p, const layout L, const float * x, float * y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= L.rows) return;
    const int r = warp;
    const int nidx_row = L.cols / 4;
    const uint2 * cb = (const uint2 *) (p + L.cb_off);
    const __half * gr = (const __half *) (p + L.gain_off);
    const uint8_t * st = p + L.stream_off;
    const float4 * x4 = (const float4 *) x;

    float acc = 0.0f;
    int i = lane;
    for (; i + 32 * (GEMV_UNROLL - 1) < nidx_row; i += 32 * GEMV_UNROLL) {
        for (int k = 0; k < GEMV_UNROLL; k++) {
            const int ii = i + k * 32;
            const uint32_t v = rd_idx_fmt(st, L, r, ii);
            const uint2 c = __ldg(cb + v);
            const float4 xv = x4[ii];
            const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&c.x));
            const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&c.y));
            acc += a.x * xv.x + a.y * xv.y + b.x * xv.z + b.y * xv.w;
        }
    }
    for (; i < nidx_row; i += 32) {
        const uint32_t v = rd_idx_fmt(st, L, r, i);
        const uint2 c = __ldg(cb + v);
        const float4 xv = x4[i];
        const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&c.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&c.y));
        acc += a.x * xv.x + a.y * xv.y + b.x * xv.z + b.y * xv.w;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[r] += acc * __half2float(gr[r]);
}

/* Prefill: one thread per (row, token). The row's indices are staged in shared
 * memory once by the block, then every thread walks them, reading the codebook
 * (the same address across the block, so L1 broadcasts it) and its own token's
 * activations. xT is [col][token], so the block's reads at a fixed (i, d) are
 * consecutive addresses: one coalesced transaction per warp. There is no
 * cross-thread reduction, because each thread owns one output element.
 *
 * The first version of this kernel indexed the activation buffer per row and
 * uncoalesced the reads by 8 KB; it ran at 6 tok/s. This shape is what the ds4
 * group kernel does with codeword-once-dot-per-token, transposed onto threads. */
#define PRE_THREADS 256

__global__ void gemm_prefill_kernel(const uint8_t * p, const layout L, const float * xT,
                                    int ntok, float * y) {
    extern __shared__ uint16_t sidx[];
    const int r = blockIdx.x;
    const int nidx_row = L.cols / L.dim;
    const uint8_t * st = p + L.stream_off;

    for (int i = threadIdx.x; i < nidx_row; i += blockDim.x)
        sidx[i] = (uint16_t) rd_idx_fmt(st, L, r, i);
    __syncthreads();

    const int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= ntok) return;

    const __half * cb = (const __half *) (p + L.cb_off);
    const float * xt = xT + t;
    float acc = 0.0f;
    for (int i = 0; i < nidx_row; i++) {
        const __half * c = cb + (size_t) sidx[i] * L.dim;
        const float * xg = xt + (size_t) i * L.dim * ntok;
        for (int d = 0; d < L.dim; d++) acc += __half2float(c[d]) * xg[(size_t) d * ntok];
    }
    const __half gh = ((const __half *) (p + L.gain_off))[r];
    y[(size_t) t * L.rows + r] += acc * __half2float(gh);
}

/* One warp per row, no shared codebook: the 64 KB table stays in L1 and the
 * warp-shuffle reduction removes every block-wide barrier. This is the shape
 * that hides the gather latency; the shared-memory version below is kept for
 * comparison. */
__global__ void gemv_warp_kernel(const uint8_t * p, const layout L, const float * x, float * y) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= L.rows) return;
    const int r = warp;
    const int nidx_row = L.cols / L.dim;
    const __half * cb = (const __half *) (p + L.cb_off);
    const __half * gr = (const __half *) (p + L.gain_off);
    const uint8_t * st = p + L.stream_off;

    float acc = 0.0f;
    for (int i = lane; i < nidx_row; i += 32) {
        const uint32_t v = rd_idx_fmt(st, L, r, i);
        const __half * c = cb + (size_t) v * L.dim;
        const float * xg = x + (size_t) i * L.dim;
        for (int d = 0; d < L.dim; d++) acc += __half2float(c[d]) * xg[d];
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[r] += acc * __half2float(gr[r]);
}

/* Eight rows per block: the codebook is staged in shared memory once and reused
 * across the rows, which is what makes the gather cheap enough to compete with
 * reading an expanded matrix. Threads split a row's indices; the block reduces
 * per row. */
#define ROWS_PER_BLOCK 8

__global__ void gemv_kernel(const uint8_t * p, const layout L, const float * x, float * y) {
    extern __shared__ __half scb[];
    const __half * cb = (const __half *) (p + L.cb_off);
    const __half * gr = (const __half *) (p + L.gain_off);
    const uint8_t * st = p + L.stream_off;
    const int nidx_row = L.cols / L.dim;

    for (int i = threadIdx.x; i < L.nc * L.dim; i += blockDim.x) scb[i] = cb[i];
    __syncthreads();

    const int r0 = blockIdx.x * ROWS_PER_BLOCK;
    __shared__ float red[256];
    for (int rr = 0; rr < ROWS_PER_BLOCK; rr++) {
        const int r = r0 + rr;
        if (r >= L.rows) break;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < nidx_row; i += blockDim.x) {
            const uint32_t v = rd_idx_fmt(st, L, r, i);
            const __half * c = scb + (size_t) v * L.dim;
            const float * xg = x + (size_t) i * L.dim;
            for (int d = 0; d < L.dim; d++) acc += __half2float(c[d]) * xg[d];
        }
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[r] += red[0] * __half2float(gr[r]);
        __syncthreads();
    }
}

/* Plain f32 GEMV, the "already expanded" baseline: one block per row. */
__global__ void gemv_f32_kernel(const float * w, const float * x, float * y, int rows, int cols) {
    const int r = blockIdx.x;
    const float * wr = w + (size_t) r * cols;
    const float * xr = x;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) acc += wr[c] * xr[c];
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[r] += red[0];
}

extern "C" {

/* Decode one stage into out (device pointer), then optionally time `iters`
 * repeated launches. Returns the elapsed milliseconds of the best iteration. */
int vqg_dequant(const void * payload, int dim, int nc, int rows, int cols, int v3,
                float * out, int iters, double * ms) {
    const layout L = make_layout(dim, nc, rows, cols, v3);
    const int threads = 256;
    const int blocks = (int) ((L.nidx + threads - 1) / threads);
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess)
        return fail("event create", cudaGetLastError());
    float best = 1e30f;
    for (int it = 0; it < iters; it++) {
        cudaEventRecord(a);
        dequant_kernel<<<blocks, threads>>>((const uint8_t *) payload, L, out);
        cudaEventRecord(b);
        if (cudaEventSynchronize(b) != cudaSuccess) return fail("dequant launch", cudaGetLastError());
        float t = 0;
        cudaEventElapsedTime(&t, a, b);
        if (t < best) best = t;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    *ms = best;
    return 0;
}

/* Fused gather+dot per stage, accumulated into y (device pointer, rows).
 * variant 0 = shared-memory codebook, 1 = warp per row with the table in L1. */
int vqg_gemv(const void * payload, int dim, int nc, int rows, int cols, int v3,
             const float * x, float * y, int variant, int iters, double * ms) {
    const layout L = make_layout(dim, nc, rows, cols, v3);
    const int threads = 256;
    const size_t smem = (variant == 0) ? (size_t) L.nc * L.dim * 2 : 0;
    const int blocks = (variant == 0)
        ? (L.rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
        : (L.rows * 32 + threads - 1) / threads;
    if (smem > 48 * 1024 &&
        cudaFuncSetAttribute(gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) != cudaSuccess)
        return fail("shared memory opt-in", cudaGetLastError());
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess)
        return fail("event create", cudaGetLastError());
    float best = 1e30f;
    for (int it = 0; it < iters; it++) {
        cudaEventRecord(a);
        if (variant == 0)      gemv_kernel<<<blocks, threads, smem>>>((const uint8_t *) payload, L, x, y);
        else if (variant == 2 && L.dim == 4) gemv_warp4_kernel<<<blocks, threads>>>((const uint8_t *) payload, L, x, y);
        else                   gemv_warp_kernel<<<blocks, threads>>>((const uint8_t *) payload, L, x, y);
        cudaEventRecord(b);
        if (cudaEventSynchronize(b) != cudaSuccess) return fail("gemv launch", cudaGetLastError());
        float t = 0;
        cudaEventElapsedTime(&t, a, b);
        if (t < best) best = t;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    *ms = best;
    return 0;
}

/* Prefill one stage: xT is [cols][ntok], y is [ntok][rows], both device-side. */
int vqg_prefill(const void * payload, int dim, int nc, int rows, int cols, int v3,
                const float * xT, int ntok, float * y, int iters, double * ms) {
    const layout L = make_layout(dim, nc, rows, cols, v3);
    const int smem = /* shared index staging */ (int) ((size_t) (cols / dim) * 2);
    if (smem > 48 * 1024 &&
        cudaFuncSetAttribute(gemm_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
        return fail("prefill shared memory opt-in", cudaGetLastError());
    const dim3 grid(L.rows, (ntok + PRE_THREADS - 1) / PRE_THREADS);
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess)
        return fail("event create", cudaGetLastError());
    float best = 1e30f;
    for (int it = 0; it < iters; it++) {
        cudaEventRecord(a);
        gemm_prefill_kernel<<<grid, PRE_THREADS, smem>>>((const uint8_t *) payload, L, xT, ntok, y);
        cudaEventRecord(b);
        if (cudaEventSynchronize(b) != cudaSuccess) return fail("prefill launch", cudaGetLastError());
        float t = 0;
        cudaEventElapsedTime(&t, a, b);
        if (t < best) best = t;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    *ms = best;
    return 0;
}

int vqg_gemv_f32(const float * w, const float * x, float * y, int rows, int cols, int iters, double * ms) {
    const int threads = 128;
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess)
        return fail("event create", cudaGetLastError());
    float best = 1e30f;
    for (int it = 0; it < iters; it++) {
        cudaEventRecord(a);
        gemv_f32_kernel<<<rows, threads>>>(w, x, y, rows, cols);
        cudaEventRecord(b);
        if (cudaEventSynchronize(b) != cudaSuccess) return fail("gemv f32 launch", cudaGetLastError());
        float t = 0;
        cudaEventElapsedTime(&t, a, b);
        if (t < best) best = t;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    *ms = best;
    return 0;
}

} /* extern "C" */
