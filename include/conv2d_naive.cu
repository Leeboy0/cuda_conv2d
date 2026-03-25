#include "conv2d.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Naive conv2d: one thread computes one output element output[n, c_out, oh, ow]
//
// Memory layout (all NCHW / row-major):
//   input  : [N, C_in,  H,  W]
//   weight : [C_out, C_in, KH, KW]
//   output : [N, C_out, OH, OW]
// ─────────────────────────────────────────────────────────────────────────────

__global__ void conv2d_naive(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    ConvParams p
) {
    // --- Decode thread → (n, c_out, oh, ow) ----------------------------------
    const int ow    = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh    = blockIdx.y * blockDim.y + threadIdx.y;
    const int co_n  = blockIdx.z;               // flattened (n * C_out + c_out)
    const int n     = co_n / p.C_out;
    const int c_out = co_n % p.C_out;

    if (ow >= p.OW() || oh >= p.OH() || n >= p.N) return;

    // --- Accumulate over input channels and kernel window --------------------
    float acc = 0.0f;

    for (int c_in = 0; c_in < p.C_in; ++c_in) {
        for (int kh = 0; kh < p.KH; ++kh) {
            for (int kw = 0; kw < p.KW; ++kw) {

                // Map output position back to input position
                const int ih = oh * p.stride - p.pad + kh;
                const int iw = ow * p.stride - p.pad + kw;

                // Zero-pad: skip out-of-bounds input coordinates
                float in_val = 0.0f;
                if (ih >= 0 && ih < p.H && iw >= 0 && iw < p.W) {
                    in_val = input[((n * p.C_in + c_in) * p.H + ih) * p.W + iw];
                }

                const float w_val =
                    weight[((c_out * p.C_in + c_in) * p.KH + kh) * p.KW + kw];

                acc += in_val * w_val;
            }
        }
    }

    // --- Write result --------------------------------------------------------
    output[((n * p.C_out + c_out) * p.OH() + oh) * p.OW() + ow] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launcher
// ─────────────────────────────────────────────────────────────────────────────
void launch_naive(
    const float* d_input,
    const float* d_weight,
    float*       d_output,
    ConvParams   p
) {
    const dim3 block(TILE_W, TILE_H, 1);
    const dim3 grid(
        (p.OW()  + TILE_W - 1) / TILE_W,
        (p.OH()  + TILE_H - 1) / TILE_H,
        p.N * p.C_out
    );

    conv2d_naive<<<grid, block>>>(d_input, d_weight, d_output, p);
    CUDA_CHECK(cudaGetLastError());
}
