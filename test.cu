// ---------------------------------------------------------------------------
// test.cu — validates conv2d_naive against PyTorch ground truth.
//
// Prereq: run `python gen_testdata.py` once to create ./testdata/*.bin files.
//
// Build:
//   nvcc -O2 -Iinclude test.cu src/conv2d_naive.cu -o test_conv2d
// Run:
//   ./test_conv2d
// ---------------------------------------------------------------------------

#include "conv2d.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(_e));  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Read a raw float32 binary file into a vector.
static std::vector<float> load_bin(const std::string& path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}

static bool run_case(int case_idx) {
    char tag[16];
    std::snprintf(tag, sizeof(tag), "case%02d", case_idx);

    // --- read shape ---
    ConvParams p;
    int OH, OW;
    {
        std::string shape_path = std::string("testdata/") + tag + "_shape.txt";
        std::ifstream f(shape_path);
        if (!f) { std::fprintf(stderr, "missing %s -- run gen_testdata.py first\n",
                               shape_path.c_str()); std::exit(1); }
        f >> p.N >> p.C_in >> p.C_out >> p.H >> p.W
          >> p.KH >> p.KW >> p.pad >> p.stride >> OH >> OW;
    }

    const size_t in_n  = (size_t)p.N * p.C_in  * p.H * p.W;
    const size_t w_n   = (size_t)p.C_out * p.C_in * p.KH * p.KW;
    const size_t out_n = (size_t)p.N * p.C_out * OH * OW;

    // --- load tensors ---
    auto h_in  = load_bin(std::string("testdata/") + tag + "_input.bin",    in_n);
    auto h_w   = load_bin(std::string("testdata/") + tag + "_weight.bin",   w_n);
    auto h_ref = load_bin(std::string("testdata/") + tag + "_expected.bin", out_n);
    std::vector<float> h_gpu(out_n, 0.0f);

    // --- run kernel ---
    float *d_in, *d_w, *d_out;
    CUDA_OK(cudaMalloc(&d_in,  in_n  * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_w,   w_n   * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_out, out_n * sizeof(float)));

    CUDA_OK(cudaMemcpy(d_in, h_in.data(), in_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_w,  h_w.data(),  w_n  * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((OW + TILE_W - 1) / TILE_W,
              (OH + TILE_H - 1) / TILE_H,
              p.N * p.C_out);
    conv2d_naive<<<grid, block>>>(d_in, d_w, d_out, p);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(h_gpu.data(), d_out, out_n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);

    // --- compare ---
    float max_err = 0.0f;
    double sum_err = 0.0;
    int    first_bad = -1;
    const float tol = 1e-3f;  // loose -- PyTorch and naive accumulate in different orders
    for (size_t i = 0; i < out_n; ++i) {
        float e = std::fabs(h_ref[i] - h_gpu[i]);
        sum_err += e;
        if (e > max_err) max_err = e;
        if (e > tol && first_bad < 0) first_bad = (int)i;
    }
    bool passed = (first_bad < 0);

    std::printf("[%s] N=%d Cin=%d Cout=%d H=%d W=%d KH=%d KW=%d pad=%d stride=%d  "
                "max_err=%.3e  mean_err=%.3e  %s\n",
                tag, p.N, p.C_in, p.C_out, p.H, p.W, p.KH, p.KW, p.pad, p.stride,
                max_err, sum_err / out_n, passed ? "PASS" : "FAIL");
    if (!passed)
        std::printf("    first mismatch idx %d: ref=%.6f  gpu=%.6f\n",
                    first_bad, h_ref[first_bad], h_gpu[first_bad]);
    return passed;
}

int main() {
    const int NUM_CASES = 7;  // must match gen_testdata.py
    int passed = 0;
    for (int i = 0; i < NUM_CASES; ++i)
        if (run_case(i)) ++passed;
    std::printf("\n=== %d / %d cases passed ===\n", passed, NUM_CASES);
    return (passed == NUM_CASES) ? 0 : 1;
}
