// vq_gpu.cu — device-resident Lloyd training and nearest assignment for the
// dense VQ encoder in this repository.
//
// The ds4 reference kernels (tools/vq_calib/cuda/axk2_vq_convert.cu) take host
// pointers directly and rely on unified memory; this card reports
// pageableMemoryAccess = 0.  These kernels instead take device pointers, so the
// weight block is uploaded once and nothing crosses PCIe inside the loop.
//
// Codeword tiles live in shared memory at 1024 codewords per pass, so the
// 48 KiB per-block limit does not cap the codebook size the way the reference's
// single-tile kernels do.
//
// Determinism: the centroid update accumulates with FP64 atomics, so the
// summation order varies between runs.  Repeat runs can differ in the last bits;
// the CPU path stays the reference when byte-identical output is required.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE 1024

__global__ void assign_kernel(const float * __restrict__ X, int nvec, int dim,
                              const float * __restrict__ C, int nc,
                              int * __restrict__ idx) {
    extern __shared__ float sh[];
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvec) return;

    const float * x = X + (size_t) v * dim;
    float best = 3.4e38f;
    int bi = 0;

    for (int base = 0; base < nc; base += TILE) {
        const int n = min(TILE, nc - base);
        for (int i = threadIdx.x; i < n * dim; i += blockDim.x) sh[i] = C[(size_t) base * dim + i];
        __syncthreads();
        for (int c = 0; c < n; c++) {
            const float * cb = sh + (size_t) c * dim;
            float d = 0.0f;
            for (int k = 0; k < dim; k++) { const float t = x[k] - cb[k]; d += t * t; }
            if (d < best) { best = d; bi = base + c; }
        }
        __syncthreads();
    }
    idx[v] = bi;
}

__global__ void accumulate_kernel(const float * __restrict__ X, const int * __restrict__ idx,
                                  int nvec, int dim, double * __restrict__ sums,
                                  int * __restrict__ cnt) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvec) return;
    const int c = idx[v];
    atomicAdd(&cnt[c], 1);
    const float * x = X + (size_t) v * dim;
    double * s = sums + (size_t) c * dim;
    for (int k = 0; k < dim; k++) atomicAdd(&s[k], (double) x[k]);
}

__global__ void update_kernel(double * __restrict__ sums, int * __restrict__ cnt,
                              float * __restrict__ C, int nc, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc * dim) return;
    const int c = i / dim;
    if (cnt[c] > 0) C[i] = (float) (sums[i] / (double) cnt[c]);
}

extern "C" int vqg_train(const float * X, int nvec, int dim, int nc, int iters, float * C) {
    float * dX = NULL, * dC = NULL;
    double * dS = NULL;
    int * dIdx = NULL, * dCnt = NULL;

    if (cudaMalloc(&dX, (size_t) nvec * dim * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&dC, (size_t) nc * dim * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&dS, (size_t) nc * dim * sizeof(double)) != cudaSuccess) return 1;
    if (cudaMalloc(&dIdx, (size_t) nvec * sizeof(int)) != cudaSuccess) return 1;
    if (cudaMalloc(&dCnt, (size_t) nc * sizeof(int)) != cudaSuccess) return 1;

    if (cudaMemcpy(dX, X, (size_t) nvec * dim * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpy(dC, C, (size_t) nc * dim * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    const int threads = 256;
    const int blocks = (nvec + threads - 1) / threads;
    const size_t shbytes = (size_t) TILE * dim * sizeof(float);

    for (int it = 0; it < iters; it++) {
        assign_kernel<<<blocks, threads, shbytes>>>(dX, nvec, dim, dC, nc, dIdx);
        if (cudaGetLastError() != cudaSuccess) return 2;
        if (cudaMemset(dS, 0, (size_t) nc * dim * sizeof(double)) != cudaSuccess) return 1;
        if (cudaMemset(dCnt, 0, (size_t) nc * sizeof(int)) != cudaSuccess) return 1;
        accumulate_kernel<<<blocks, threads>>>(dX, dIdx, nvec, dim, dS, dCnt);
        if (cudaGetLastError() != cudaSuccess) return 3;
        const int ub = (nc * dim + threads - 1) / threads;
        update_kernel<<<ub, threads>>>(dS, dCnt, dC, nc, dim);
        if (cudaGetLastError() != cudaSuccess) return 4;
    }

    if (cudaMemcpy(C, dC, (size_t) nc * dim * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    cudaFree(dX); cudaFree(dC); cudaFree(dS); cudaFree(dIdx); cudaFree(dCnt);
    return 0;
}

extern "C" int vqg_assign(const float * X, int nvec, int dim, const float * C, int nc, int * idx) {
    float * dX = NULL, * dC = NULL;
    int * dIdx = NULL;
    if (cudaMalloc(&dX, (size_t) nvec * dim * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&dC, (size_t) nc * dim * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&dIdx, (size_t) nvec * sizeof(int)) != cudaSuccess) return 1;
    if (cudaMemcpy(dX, X, (size_t) nvec * dim * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpy(dC, C, (size_t) nc * dim * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 1;

    const int threads = 256;
    const int blocks = (nvec + threads - 1) / threads;
    const size_t shbytes = (size_t) TILE * dim * sizeof(float);
    assign_kernel<<<blocks, threads, shbytes>>>(dX, nvec, dim, dC, nc, dIdx);
    if (cudaGetLastError() != cudaSuccess) return 2;
    if (cudaMemcpy(idx, dIdx, (size_t) nvec * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    cudaFree(dX); cudaFree(dC); cudaFree(dIdx);
    return 0;
}

extern "C" const char * vqg_last_error(void) { return cudaGetErrorString(cudaGetLastError()); }
