#include "conv2d.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    // ── Small config — easy to verify by hand ────────────────────────────────
    ConvParams p;
    p.N      = 1;
    p.C_in   = 1;
    p.C_out  = 1;
    p.H      = 4;
    p.W      = 4;
    p.KH     = 3;
    p.KW     = 3;
    p.pad    = 0;
    p.stride = 1;
    // OH = OW = (4 + 0 - 3)/1 + 1 = 2  →  output is [1,1,2,2]

    // ── Input: all 1s ────────────────────────────────────────────────────────
    float h_input[16];
    for (int i = 0; i < p.input_size(); ++i) h_input[i] = 1.0f;

    // ── Kernel: all 1s  → each output = sum of 3×3 = 9 ─────────────────────
    float h_weight[9];
    for (int i = 0; i < p.kernel_size(); ++i) h_weight[i] = 1.0f;

    float h_output[4] = {0};

    // ── Device alloc + copy ───────────────────────────────────────────────────
    float *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  p.input_size()  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, p.kernel_size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, p.output_size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input,  h_input,  p.input_size()  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, p.kernel_size() * sizeof(float), cudaMemcpyHostToDevice));

    // ── Launch ────────────────────────────────────────────────────────────────
    launch_naive(d_input, d_weight, d_output, p);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Copy back + print ─────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_output, d_output, p.output_size() * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Output (expected all 9.0):\n");
    for (int i = 0; i < p.output_size(); ++i)
        printf("  out[%d] = %.1f %s\n", i, h_output[i], fabsf(h_output[i] - 9.0f) < 1e-4f ? "✓" : "✗ WRONG");

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    return 0;
}
