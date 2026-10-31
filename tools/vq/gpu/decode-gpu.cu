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
    size_t cb_off, gain_off, stream_off, nidx, nidx_row;
    size_t mrow, prow;   /* DQV3: bytes per row of the main stream (min(bits,12)) and of the plane */
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
    L.nidx_row = nidx_row;
    L.mrow = v3 ? (nidx_row * (size_t) mb + 7) / 8 : 0;
    L.prow = (v3 && L.nbit > 12) ? (nidx_row + 7) / 8 : 0;
    return L;
}

/* Preconditions of the DQV3 block kernel: a 12-bit main stream (nc >= 4096;
 * below that the stream is narrower and this kernel's 96-bit round map does
 * not describe it), a v3 payload whose index stream starts on a 4-byte
 * boundary in every row, and no partial 32-index group. */
static int v3_blk_ok(const layout & L) {
    return L.v3 && L.nbit >= 12 && L.nidx_row % 32 == 0 && L.mrow % 4 == 0 && L.stream_off % 4 == 0;
}

/* DQV3 carries 12 main-stream bits plus one plane bit, so nc above 8192 would
 * be decoded from truncated indices with no error. Refuse it in every entry
 * point instead (build-vq-gpu.c refuses to write such a container). */
static int v3_width_ok(int nc) {
    int b = 1;
    while ((1 << b) < nc) b++;
    return b <= 13;
}

static int bad_width(int nc) {
    int b = 1;
    while ((1 << b) < nc) b++;
    snprintf(g_err, sizeof g_err, "DQV3 carries 13 index bits, nc %d needs %d", nc, b);
    return -1;
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
    /* The v3 main stream is min(bit_width, 12) bits wide (build-vq-gpu.c:59
     * packs mb = bits > 12 ? 12 : bits), not always 12: a 4096-codebook
     * payload is 12-bit, a 1024 one is 10-bit and its rows are correspondingly
     * shorter. Reading a fixed 12 here decoded the wrong weights for every nc
     * below 4096 and, at nc 1024, produced indices past the codebook. */
    const uint32_t mb = L.nbit > 12 ? 12u : (uint32_t) L.nbit;
    const uint8_t * base = st + r * L.mrow;
    const size_t bit = i * mb;
    const uint8_t * q = base + (bit >> 3);
    const uint32_t w = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16) | ((uint32_t) q[3] << 24);
    uint32_t v = (w >> (bit & 7)) & ((1u << mb) - 1u);
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

/* ---- DQV3 block-load decode ---------------------------------------------
 *
 * The v3 stream is one 12-bit index after another, row by row, so a warp can
 * read it as whole 96-word blocks (three 128-byte lines) instead of one index
 * at a time: lane j takes words j / j+32 / j+64, and a round's index word is
 * fetched from the lane that holds it by a shuffle plus a funnel shift. The
 * block's 13th bits are 8 words for 256 indices, so lane j<8 holds word j and
 * round k takes bit `lane` of the word held by lane k: one more shuffle, no
 * extra memory traffic.
 *
 * Ported from ds4-dfm-rs cuda/ds41_vq_row.cuh:122-137 (the 3-line block and
 * whose-lane-holds-what map) and :200-220 (the per-round funnelshift and the
 * plane bit by shfl). The index numbering is unchanged: index i of a row sits
 * at bit 12*i, so round k's lane j is index 32(b+k)+j, word 12k+floor(3j/8),
 * shift 12j%32. The kernel is gated against the host f64 matvec, not against
 * the other kernels' accumulation order. */

struct v3blk { uint32_t w0, w1, w2, ex; };

__device__ __forceinline__ uint32_t v3_sel3(uint32_t r, uint32_t w0, uint32_t w1, uint32_t w2) {
    return r == 0u ? w0 : (r == 1u ? w1 : w2);
}

/* Only the words that exist are read: a tail block has fewer than 96, and its
 * plane fewer than 8. The payload's tail pad is not relied on. */
__device__ __forceinline__ v3blk v3_blk_load(const uint32_t * blk, uint32_t nw,
                                             const uint32_t * bex, uint32_t nwx) {
    const uint32_t lane = threadIdx.x & 31u;
    v3blk b;
    b.w0 = b.w1 = b.w2 = b.ex = 0u;
    if (lane < nw)          b.w0 = blk[lane];
    if (lane + 32u < nw)    b.w1 = blk[lane + 32u];
    if (lane + 64u < nw)    b.w2 = blk[lane + 64u];
    if (bex && lane < nwx)  b.ex = bex[lane];
    return b;
}

/* One block's rounds. DIM 4 has the uint2 codebook load and half2 conversion;
 * DIM 0 is the generic path at L.dim. EXT 1 reads the 13th bit from the plane,
 * EXT 0 is a 12-bit layer (nc <= 4096) with no plane at all. */
template <int DIM, int EXT>
__device__ __forceinline__ void v3_rounds(const v3blk & cur, uint32_t rounds, uint32_t a, uint32_t sh,
                                          const float * xb, const __half * cb, int dim, float & acc) {
    const uint32_t lane = threadIdx.x & 31u;
    #pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
        if (k >= rounds) break;
        const uint32_t f = k * 12u;                                      /* first word of this round */
        const uint32_t lo_r = ((f & 31u) > 32u - 12u) ? ((f + 31u - lane) >> 5) : (f >> 5);
        const uint32_t hi_r = (((f + 1u) & 31u) > 32u - 12u) ? ((f + 32u - lane) >> 5) : ((f + 1u) >> 5);
        const uint32_t lo = __shfl_sync(0xffffffffu, v3_sel3(lo_r, cur.w0, cur.w1, cur.w2), (int) (f + a));
        const uint32_t hi = __shfl_sync(0xffffffffu, v3_sel3(hi_r, cur.w0, cur.w1, cur.w2), (int) (f + a + 1u));
        uint32_t v = __funnelshift_r(lo, hi, sh) & 0xFFFu;
        if (EXT) {
            const uint32_t exw = __shfl_sync(0xffffffffu, cur.ex, (int) k);
            v |= ((exw >> lane) & 1u) << 12;
        }
        const size_t g = (size_t) (k * 32u + lane);
        if (DIM == 4) {
            const float4 xv = *(const float4 *) (xb + g * 4u);
            const uint2 c = __ldg((const uint2 *) cb + v);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *> (&c.x));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *> (&c.y));
            acc += f0.x * xv.x + f0.y * xv.y + f1.x * xv.z + f1.y * xv.w;
        } else {
            const __half * cw = cb + (size_t) v * (size_t) dim;
            const float * xg = xb + g * (size_t) dim;
            for (int d = 0; d < dim; d++) acc += __half2float(cw[d]) * xg[d];
        }
    }
}

/* One warp per row. The next block's three words are issued before this block
 * computes, which is the reference's cross-block pipeline (cuda_vq_row.inc.cu:
 * 163-175): with 32 warps in lockstep, every warp waits on DRAM together
 * otherwise. */
template <int DIM, int EXT>
__global__ void gemv_v3blk_kernel(const uint8_t * p, const layout L, const float * x, float * y) {
    const uint32_t warp = (uint32_t) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const uint32_t lane = threadIdx.x & 31u;
    if (warp >= (uint32_t) L.rows) return;
    const int r = (int) warp;
    const __half * cb = (const __half *) (p + L.cb_off);
    const __half * gr = (const __half *) (p + L.gain_off);
    const uint32_t * row = (const uint32_t *) (p + L.stream_off + (size_t) r * L.mrow);
    const uint32_t * rex = EXT
        ? (const uint32_t *) (p + L.stream_off + L.mrow * (size_t) L.rows + (size_t) r * L.prow)
        : NULL;
    const uint32_t R = (uint32_t) (L.nidx_row >> 5);          /* groups of 32 indices */
    const uint32_t nblk = (R + 7u) >> 3;
    const uint32_t a = (lane * 12u) >> 5, sh = (lane * 12u) & 31u;

    v3blk cur = v3_blk_load(row, R >= 8u ? 96u : R * 12u, rex, R >= 8u ? 8u : R);
    float acc = 0.0f;
    for (uint32_t b = 0; b < nblk; b++) {
        v3blk nxt;
        if (b + 1u < nblk) {
            const uint32_t nrem = R - (b + 1u) * 8u, nr = nrem < 8u ? nrem : 8u;
            nxt = v3_blk_load(row + (size_t) (b + 1u) * 96u, nr * 12u,
                              EXT ? rex + (b + 1u) * 8u : NULL, nr);
        } else {
            nxt.w0 = nxt.w1 = nxt.w2 = nxt.ex = 0u;
        }
        const uint32_t rem = R - b * 8u;
        v3_rounds<DIM, EXT>(cur, rem < 8u ? rem : 8u, a, sh,
                            x + (size_t) b * 256u * (DIM > 0 ? DIM : L.dim), cb, L.dim, acc);
        cur = nxt;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[r] += acc * __half2float(gr[r]);
}

/* ==== Tiled prefill: f16 tensor cores =====================================
 *
 * One block computes 128 weight rows x 128 tokens over all K, stepping BK 64
 * columns per round: 16 warps = 8 row tiles of 16 rows x 2 token halves; a
 * warp owns 16 rows and 8 n8 slices, and a round is 4 k16 mma groups. The tile
 * shape, the 16-warp split, the ldmatrix addressing and the 16-byte-chunk XOR
 * swizzle are ds4-dfm-rs cuda/ds41_vq_prefill_mma.cuh:52-58 and :82-94; the
 * per-round tile fill and the k16 loop follow :167-201 there.
 *
 * Named deviation from that reference: ds4 runs bf16 products because its
 * codebook is E4M3 (3 mantissa bits, exact in bf16). Ours is f16, 10 mantissa
 * bits, and bf16 would round every learned codeword by 2^-8 - a quality
 * change, not a layout choice. So the tile is f16: a code word is a straight
 * 8- or 16-byte copy out of the codebook with no conversion at all, the
 * activations are rounded f32 -> f16 (10 bits, not 7), and the mma is
 * f32.f16.f16.f32. An f16 x f16 product is exact in f32.
 *
 * The mma starts every k16 from zero and adds back with a plain FADD (ds4's
 * 2026-09-24 finding at :188-192: letting K accumulate inside the tensor
 * core's own f32 adder truncates low bits on alignment instead of rounding).
 */
#define PRE_BM      128
#define PRE_BN      128
#define PRE_BK      64
#define PRE_THREADS 512

/* A tile row is 64 f16 = 8 chunks of 16 B; chunk ^ (row & 7) spreads the 8
 * rows ldmatrix reads at once over 8 bank groups (ds4 :83). */
__device__ __forceinline__ uint32_t pre_swz(uint32_t row, uint32_t chunk) {
    return row * 128u + ((chunk ^ (row & 7u)) << 4);
}

__device__ __forceinline__ void pre_ldsm4(uint32_t * r, const uint8_t * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void pre_mma(float * c, const uint32_t * a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pre_h2u(__half2 h) { return *reinterpret_cast<uint32_t *>(&h); }

/* CB 0 keeps the codebook in global (it is 32-64 KB: it stays in L1 under
 * three resident blocks), CB 1 stages it in shared behind the two tiles. */
template <int CB, int DIM>
__global__ __launch_bounds__(PRE_THREADS, 1) void gemm_tiled_kernel(
        const uint8_t * p, const layout L, const float * xT, int ntok, float * y) {
    constexpr uint32_t CW  = PRE_BK / (uint32_t) DIM;    /* code words per row-round: 16 (dim 4), 8 (dim 8) */
    constexpr uint32_t TPR = 4u, CPT = CW / TPR;         /* 4 threads per row, 4 / 2 code words each */
    constexpr uint32_t NFW = PRE_BN / 16u;               /* n8 slices per warp (8) */
    constexpr uint32_t NCH = PRE_BN * 8u / PRE_THREADS;  /* B-tile 16 B chunks per thread (2) */
    extern __shared__ __align__(16) uint8_t sh[];
    uint8_t * As = sh, * Bs = sh + PRE_BM * PRE_BK * 2u;
    uint8_t * cbs = Bs + PRE_BN * PRE_BK * 2u;
    const __half * cb = (const __half *) (p + L.cb_off);
    const __half * gr = (const __half *) (p + L.gain_off);
    const uint8_t * st = p + L.stream_off;
    const uint8_t * pl = L.prow ? st + L.mrow * (size_t) L.rows : NULL;

    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t wr = warp & 7u, th = warp >> 3u;
    if (CB) {   /* the codebook bytes are already the tile format: copy, no convert */
        const uint2 * src = (const uint2 *) cb;
        uint2 * dst = (uint2 *) cbs;
        for (uint32_t i = tid; i < (uint32_t) L.nc * (uint32_t) L.dim * 2u / 8u; i += PRE_THREADS)
            dst[i] = __ldg(src + i);
    }

    const uint32_t ar = tid / TPR, sub = tid % TPR;      /* A-tile row this thread fills */
    const int r0 = (int) blockIdx.x * PRE_BM, t0 = (int) blockIdx.y * PRE_BN;
    const int nt = min(PRE_BN, ntok - t0);
    const int nrnd = (int) (L.cols / PRE_BK);
    const bool rv = r0 + (int) ar < L.rows;
    const uint32_t rsrc = (uint32_t) (rv ? r0 + (int) ar : 0);
    const uint8_t * rowp = st + (size_t) rsrc * L.mrow;
    const uint8_t * rowq = pl ? pl + (size_t) rsrc * L.prow : NULL;

    float acc[NFW][4];
    #pragma unroll
    for (uint32_t f = 0; f < NFW; f++) acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.f;
    __syncthreads();

    for (int i = 0; i < nrnd; i++) {
        /* This thread's code words of round i sit at bit 12*(i*CW + sub*CPT).
         * Two aligned words are read and shifted; the offset inside them is
         * 0 or 16 for dim 4 and in {0,8,16,24} for dim 8, and CPT*12 <= 48, so
         * every index of the round stays inside the 64-bit window. */
        const uint32_t bit = (uint32_t) i * 12u * CW + sub * 12u * CPT;
        const uint32_t aoff = (bit >> 5) << 2, sh64 = bit & 31u;
        uint32_t w0 = 0, w1 = 0, eb = 0;
        if (rv) {
            w0 = __ldg((const uint32_t *) (rowp + aoff));
            w1 = __ldg((const uint32_t *) (rowp + aoff + 4u));
            if (rowq) eb = (uint32_t) __ldg(rowq + ((uint32_t) i * CW + sub * CPT) / 8u);
        }
        /* The next round's activations, converted here so the global latency
         * overlaps the previous round's mma (ds4 :163-175). */
        uint4 bx[NCH];
        #pragma unroll
        for (uint32_t c = 0; c < NCH; c++) {
            const uint32_t ch = tid + c * PRE_THREADS, t = ch >> 3, cc = ch & 7u;
            bx[c] = make_uint4(0, 0, 0, 0);
            if ((int) t < nt) {
                const float * s = xT + (size_t) ((uint32_t) i * PRE_BK + cc * 8u) * (size_t) ntok
                                + (size_t) (t0 + (int) t);
                const size_t S = (size_t) ntok;
                bx[c].x = pre_h2u(__floats2half2_rn(s[0], s[S]));
                bx[c].y = pre_h2u(__floats2half2_rn(s[2 * S], s[3 * S]));
                bx[c].z = pre_h2u(__floats2half2_rn(s[4 * S], s[5 * S]));
                bx[c].w = pre_h2u(__floats2half2_rn(s[6 * S], s[7 * S]));
            }
        }
        __syncthreads();   /* the previous round's tiles are done being read */

        const uint64_t u = ((uint64_t) w1 << 32) | (uint64_t) w0;
        #pragma unroll
        for (uint32_t q = 0; q < CPT; q++) {
            const uint32_t j = sub * CPT + q;                     /* code word of this round */
            uint32_t v = (uint32_t) (u >> (sh64 + 12u * q)) & 0xFFFu;
            if (rowq) v |= ((eb >> (j & 7u)) & 1u) << 12;
            if (DIM == 4) {   /* 8-byte code word: half of a 16 B chunk */
                const uint8_t * src = CB ? cbs + (size_t) v * 8u : (const uint8_t *) cb + (size_t) v * 8u;
                const uint2 cw = CB ? *(const uint2 *) src : __ldg((const uint2 *) src);
                *(uint2 *) (As + pre_swz(ar, j >> 1) + ((j & 1u) << 3)) = cw;
            } else {   /* dim 8: one code word is one 16 B chunk, two aligned 8 B halves */
                const uint8_t * src = CB ? cbs + (size_t) v * 16u : (const uint8_t *) cb + (size_t) v * 16u;
                const uint2 lo = CB ? *(const uint2 *) src : __ldg((const uint2 *) src);
                const uint2 hi = CB ? *(const uint2 *) (src + 8u) : __ldg((const uint2 *) (src + 8u));
                uint8_t * dst = As + pre_swz(ar, j);
                *(uint2 *) dst = lo;
                *(uint2 *) (dst + 8u) = hi;
            }
        }
        #pragma unroll
        for (uint32_t c = 0; c < NCH; c++) {
            const uint32_t ch = tid + c * PRE_THREADS;
            *(uint4 *) (Bs + pre_swz(ch >> 3, ch & 7u)) = bx[c];
        }
        __syncthreads();

        /* The warp's 16 rows over this half's n8 slices. */
        #pragma unroll
        for (uint32_t kk = 0; kk < PRE_BK / 16u; kk++) {
            const uint32_t mt = lane >> 3;
            uint32_t a[4];
            pre_ldsm4(a, As + pre_swz(wr * 16u + (lane & 7u) + (mt & 1u) * 8u, kk * 2u + (mt >> 1)));
            #pragma unroll
            for (uint32_t np = 0; np < NFW / 2u; np++) {
                const uint32_t f0 = th * NFW + 2u * np;
                if ((int) (f0 * 8u) >= nt) break;
                uint32_t b[4];
                pre_ldsm4(b, Bs + pre_swz((f0 + (mt >> 1)) * 8u + (lane & 7u), kk * 2u + (mt & 1u)));
                float t0v[4] = {0.f, 0.f, 0.f, 0.f};
                pre_mma(t0v, a, b[0], b[1]);
                #pragma unroll
                for (uint32_t e = 0; e < 4u; e++) acc[2 * np][e] += t0v[e];
                if ((int) ((f0 + 1u) * 8u) < nt) {
                    float t1v[4] = {0.f, 0.f, 0.f, 0.f};
                    pre_mma(t1v, a, b[2], b[3]);
                    #pragma unroll
                    for (uint32_t e = 0; e < 4u; e++) acc[2 * np + 1][e] += t1v[e];
                }
            }
        }
    }

    /* Exit: c0,c1 = row lane/4, token (lane&3)*2+{0,1}; c2,c3 = row + 8. */
    #pragma unroll
    for (uint32_t h = 0; h < 2u; h++) {
        const uint32_t r = (uint32_t) r0 + wr * 16u + (lane >> 2) + h * 8u;
        if (r >= (uint32_t) L.rows) continue;
        const float g = __half2float(__ldg(gr + r));
        #pragma unroll
        for (uint32_t f = 0; f < NFW; f++) {
            const uint32_t gf = th * NFW + f;
            if ((int) (gf * 8u) >= nt) break;
            #pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t t = gf * 8u + (lane & 3u) * 2u + e;
                if ((int) t >= nt) continue;
                y[(size_t) (t0 + (int) t) * L.rows + r] += acc[f][h * 2u + e] * g;
            }
        }
    }
}

/* Shared bytes a tiled-prefill launch needs: the two 16 KB tiles, plus the
 * codebook when it is staged (CB 1). */
static size_t pre_smem(const layout & L, int cb) {
    return (size_t) (PRE_BM + PRE_BN) * PRE_BK * 2u + (cb ? (size_t) L.nc * L.dim * 2u : 0u);
}

/* Geometry the tiled kernel can take: a DQV3 payload (the round map reads the
 * row-local stream; DQVQ numbers its bits across the whole matrix), dim 4 or 8
 * (the 4-thread-per-row split assumes 2 or 4 code words each), a 12-bit main
 * stream (nc >= 4096), whole 64-column rounds, and a row stream whose 4-byte
 * loads are aligned (cols/64 whole implies it; the check is on the derived
 * width, not on the caller's promise). */
static int pre_ok(const layout & L) {
    return L.v3 && (L.dim == 4 || L.dim == 8) && L.rows > 0 && L.cols > 0 && L.nbit >= 12
        && L.cols % PRE_BK == 0 && L.mrow % 4 == 0 && L.stream_off % 4 == 0;
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
#define PRE_LEGACY_THREADS 256

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

template <int CB, int DIM>
static int pre_tiled(const uint8_t * p, const layout & L, const float * xT, int ntok, float * y,
                     int iters, double * ms) {
    const size_t smem = pre_smem(L, CB);
    if (cudaFuncSetAttribute(gemm_tiled_kernel<CB, DIM>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) smem) != cudaSuccess)
        return fail("prefill shared memory opt-in", cudaGetLastError());
    const dim3 grid((unsigned) ((L.rows + PRE_BM - 1) / PRE_BM),
                    (unsigned) ((ntok + PRE_BN - 1) / PRE_BN));
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess)
        return fail("event create", cudaGetLastError());
    float best = 1e30f;
    for (int it = 0; it < iters; it++) {
        cudaEventRecord(a);
        gemm_tiled_kernel<CB, DIM><<<grid, PRE_THREADS, smem>>>(p, L, xT, ntok, y);
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

extern "C" {

/* Largest dynamic shared-memory window the device accepts; the shared-codebook
 * variant needs nc*dim f16 in one block, which a wide codebook cannot fit. */
static size_t smem_max(void) {
    static size_t v = 0;
    if (!v) {
        int x = 0;
        v = (cudaDeviceGetAttribute(&x, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) == cudaSuccess && x > 0)
            ? (size_t) x : (size_t) (48 * 1024);
    }
    return v;
}

/* Whether a variant can run on this geometry; variant 3 (the DQV3 block
 * kernel) needs 4-byte-aligned row streams and no partial 32-index group, so
 * a caller must ask before timing it. */
int vqg_gemv_ready(int dim, int nc, int rows, int cols, int v3, int variant) {
    const layout L = make_layout(dim, nc, rows, cols, v3);
    switch (variant) {
        case 3: return v3_blk_ok(L);
        case 2: return L.dim == 4;
        case 1: return 1;   /* generic warp, any dim */
        case 0: return (size_t) L.nc * L.dim * 2 <= smem_max();
        default: return 0;
    }
}

/* One fused GEMV launch, no events and no synchronization. The model runtime
 * issues a chain of these per token and synchronizes once, so the chain costs
 * one sync, not one per matrix. variant 0 picks the fastest kernel the
 * geometry supports; the shared-codebook variant is not offered here because
 * it needs its dynamic shared-memory opt-in before the loop starts. */
static int gemv_launch(const uint8_t * p, const layout & L, const float * x, float * y, int variant) {
    const int threads = 256;
    if (variant == 0) variant = v3_blk_ok(L) ? 3 : (L.dim == 4 ? 2 : 1);
    const int blocks = (L.rows * 32 + threads - 1) / threads;
    if (variant == 3) {
        if (!v3_blk_ok(L)) { snprintf(g_err, sizeof g_err, "the DQV3 block kernel does not take this geometry"); return -1; }
        if (L.dim == 4) {
            if (L.prow) gemv_v3blk_kernel<4, 1><<<blocks, threads>>>(p, L, x, y);
            else        gemv_v3blk_kernel<4, 0><<<blocks, threads>>>(p, L, x, y);
        } else {
            if (L.prow) gemv_v3blk_kernel<0, 1><<<blocks, threads>>>(p, L, x, y);
            else        gemv_v3blk_kernel<0, 0><<<blocks, threads>>>(p, L, x, y);
        }
    } else if (variant == 2 && L.dim == 4) {
        gemv_warp4_kernel<<<blocks, threads>>>(p, L, x, y);
    } else {
        gemv_warp_kernel<<<blocks, threads>>>(p, L, x, y);
    }
    return 0;
}

/* Decode one stage into out (device pointer), then optionally time `iters`
 * repeated launches. Returns the elapsed milliseconds of the best iteration. */
int vqg_dequant(const void * payload, int dim, int nc, int rows, int cols, int v3,
                float * out, int iters, double * ms) {
    if (v3 && !v3_width_ok(nc)) return bad_width(nc);
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
    if (v3 && !v3_width_ok(nc)) return bad_width(nc);
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
        else if (gemv_launch((const uint8_t *) payload, L, x, y, variant)) return -1;
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

/* One GEMV launch for a caller that already checked the geometry, with no
 * timing and no synchronization (see gemv_launch). variant 0 = auto. */
int vqg_gemv_launch(const void * payload, int dim, int nc, int rows, int cols, int v3,
                    const float * x, float * y, int variant) {
    if (v3 && !v3_width_ok(nc)) return bad_width(nc);
    if (rows <= 0 || cols <= 0) { snprintf(g_err, sizeof g_err, "gemv needs a non-empty matrix"); return -1; }
    return gemv_launch((const uint8_t *) payload, make_layout(dim, nc, rows, cols, v3), x, y, variant);
}

/* Rebuild one row of one stage into out (device pointer, accumulated). The
 * model runtime needs exactly this for the token embedding: the tied
 * embedding matrix is a VQ container too, and expanding it to f32 would cost
 * 1.3 GB of device memory for rows it will never read. */
__global__ void row_dequant_kernel(const uint8_t * p, const layout L, int row, float * out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L.cols / L.dim) return;
    const __half * cb = (const __half *) (p + L.cb_off);
    const __half * gr = (const __half *) (p + L.gain_off);
    const uint8_t * st = p + L.stream_off;
    const uint32_t v = rd_idx_fmt(st, L, (size_t) row, (size_t) i);
    const float g = __half2float(gr[row]);
    const __half * c = cb + (size_t) v * L.dim;
    float * o = out + (size_t) i * L.dim;
    for (int d = 0; d < L.dim; d++) o[d] += __half2float(c[d]) * g;
}

int vqg_row_dequant(const void * payload, int dim, int nc, int rows, int cols, int v3,
                    int row, float * out) {
    if (v3 && !v3_width_ok(nc)) return bad_width(nc);
    const layout L = make_layout(dim, nc, rows, cols, v3);
    if (row < 0 || row >= L.rows) { snprintf(g_err, sizeof g_err, "row %d is outside %d", row, L.rows); return -1; }
    const int threads = 256;
    const int blocks = (int) (((L.cols / L.dim) + threads - 1) / threads);
    row_dequant_kernel<<<blocks, threads>>>((const uint8_t *) payload, L, row, out);
    return 0;
}

/* Prefill one stage, legacy shape: xT is [cols][ntok], y is [ntok][rows], both
 * device-side. One thread per (row, token), no cross-thread reduction. */
static int pre_legacy(const void * payload, int dim, int nc, int rows, int cols, int v3,
                      const float * xT, int ntok, float * y, int iters, double * ms) {
    const layout L = make_layout(dim, nc, rows, cols, v3);
    const int smem = /* shared index staging */ (int) ((size_t) (cols / dim) * 2);
    if (smem > 48 * 1024 &&
        cudaFuncSetAttribute(gemm_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
        return fail("prefill shared memory opt-in", cudaGetLastError());
    const dim3 grid(L.rows, (ntok + PRE_LEGACY_THREADS - 1) / PRE_LEGACY_THREADS);
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess)
        return fail("event create", cudaGetLastError());
    float best = 1e30f;
    for (int it = 0; it < iters; it++) {
        cudaEventRecord(a);
        gemm_prefill_kernel<<<grid, PRE_LEGACY_THREADS, smem>>>((const uint8_t *) payload, L, xT, ntok, y);
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

/* Whether a prefill path can run on this geometry:
 *   0 = auto (the fastest one this geometry supports)
 *   1 = tiled f16 mma, codebook through L1
 *   2 = tiled f16 mma, codebook staged in shared (needs the opt-in window)
 *   3 = legacy one-thread-per-(row, token) kernel (any geometry)
 */
int vqg_prefill_ready(int dim, int nc, int rows, int cols, int v3, int variant) {
    const layout L = make_layout(dim, nc, rows, cols, v3);
    switch (variant) {
        case 0: return 1;
        case 1: return pre_ok(L);
        case 2: return pre_ok(L) && pre_smem(L, 1) <= smem_max();
        case 3: return 1;
        default: return 0;
    }
}

/* Prefill one stage: xT is [cols][ntok], y is [ntok][rows], both device-side.
 * `variant` selects the path (see vqg_prefill_ready); 0 picks the tiled kernel
 * when the geometry allows it and the legacy kernel otherwise. */
int vqg_prefill(const void * payload, int dim, int nc, int rows, int cols, int v3,
                const float * xT, int ntok, float * y, int iters, double * ms, int variant) {
    if (v3 && !v3_width_ok(nc)) return bad_width(nc);
    if (ntok <= 0) { snprintf(g_err, sizeof g_err, "prefill needs at least one token"); return -1; }
    const layout L = make_layout(dim, nc, rows, cols, v3);
    const uint8_t * p = (const uint8_t *) payload;
    if (variant == 0) variant = pre_ok(L) ? 1 : 3;
    if (variant == 3 || !pre_ok(L)) return pre_legacy(payload, dim, nc, rows, cols, v3, xT, ntok, y, iters, ms);
    if (variant == 2) {
        if (pre_smem(L, 1) > smem_max()) { snprintf(g_err, sizeof g_err, "prefill codebook does not fit shared"); return -1; }
        return L.dim == 4 ? pre_tiled<1, 4>(p, L, xT, ntok, y, iters, ms)
                          : pre_tiled<1, 8>(p, L, xT, ntok, y, iters, ms);
    }
    if (variant == 1)
        return L.dim == 4 ? pre_tiled<0, 4>(p, L, xT, ntok, y, iters, ms)
                          : pre_tiled<0, 8>(p, L, xT, ntok, y, iters, ms);
    snprintf(g_err, sizeof g_err, "unknown prefill variant %d", variant);
    return -1;
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
