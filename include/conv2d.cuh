#pragma once
#include <cuda_runtime.h>
#include <cstdio>

// ── Tile dimensions ───────────────────────────────────────────────────────────
#define TILE_W 16
#define TILE_H 16
#define GEMM_TILE 16   // square tile for the im2col GEMM kernel

// ── CUDA error-check macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA error] %s  (file %s, line %d)\n",           \
                    cudaGetErrorString(_err), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── ConvParams ────────────────────────────────────────────────────────────────
struct ConvParams {
    int N;          // batch size
    int C_in;       // input channels
    int C_out;      // output channels
    int H, W;       // input  height, width
    int KH, KW;     // kernel height, width
    int pad;
    int stride;

    // Both __host__ and __device__ so they can be called from kernels too
    __host__ __device__ int OH() const { return (H + 2*pad - KH) / stride + 1; }
    __host__ __device__ int OW() const { return (W + 2*pad - KW) / stride + 1; }

    // Flat sizes — handy for allocation
    __host__ __device__ int input_size()  const { return N * C_in  * H  * W;  }
    __host__ __device__ int kernel_size() const { return C_out * C_in * KH * KW; }
    __host__ __device__ int output_size() const { return N * C_out * OH() * OW(); }
    // im2col column matrix: each sample expands to [C_in*KH*KW, OH*OW]
    __host__ __device__ int col_size()    const { return N * (C_in*KH*KW) * (OH()*OW()); }
};

// ── Phase 1 — Naive kernel ────────────────────────────────────────────────────
// One thread → one output element.
// Grid:  (ceil(OW/TILE_W), ceil(OH/TILE_H), N*C_out)
// Block: (TILE_W, TILE_H, 1)
__global__ void conv2d_naive(
    const float* __restrict__ input,    // [N, C_in,  H,  W ]
    const float* __restrict__ weight,   // [C_out, C_in, KH, KW]
    float*       __restrict__ output,   // [N, C_out, OH, OW]
    ConvParams p
);

// ── Phase 2 — Shared-memory tiled kernel ─────────────────────────────────────
// Each block loads an input patch (including halo) into shared memory,
// then accumulates over all input channels.
// Grid / block same shape as naive.
__global__ void conv2d_shared(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    ConvParams p
);

// ── Phase 3 — im2col + GEMM ───────────────────────────────────────────────────

// Step 3a: unroll input patches → column matrix
// col layout: [N, C_in*KH*KW, OH*OW]
// One thread writes one element of col.
__global__ void im2col_kernel(
    const float* __restrict__ input,   // [N, C_in, H, W]
    float*       __restrict__ col,     // [N, C_in*KH*KW, OH*OW]
    ConvParams p
);

// Step 3b: tiled GEMM  C = A × B
//   A: [M, K]  (the weight matrix, reshaped to [C_out, C_in*KH*KW])
//   B: [K, N_]  (one sample's col matrix  [C_in*KH*KW, OH*OW])
//   C: [M, N_]  (one sample's output      [C_out, OH*OW])
// Call once per sample in the batch.
__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N_cols
);

// ── Host launchers ────────────────────────────────────────────────────────────
// All pointers are device pointers; caller is responsible for allocation.

void launch_naive(
    const float* d_input,
    const float* d_weight,
    float*       d_output,
    ConvParams   p
);

void launch_shared(
    const float* d_input,
    const float* d_weight,
    float*       d_output,
    ConvParams   p
);

// d_col must be pre-allocated with at least p.col_size() floats.
void launch_im2col(
    const float* d_input,
    const float* d_weight,
    float*       d_output,
    float*       d_col,
    ConvParams   p
);