// C ABI around the S1 kernel family, loaded with ctypes (no torch ABI dependency, static cudart).
// Launch goes on the stream the caller passes (torch's current stream), so CUDA-graph capture works.
#include <stdint.h>
#include <cuda_runtime.h>
#include "gemv_bf16.cuh"

namespace {

using LaunchFn = int (*)(const void*, const void*, void*, int, int, cudaStream_t);

template <int R, int U, int W, bool S, bool H>
int launch(const void* Wt, const void* x, void* y, int N, int K, cudaStream_t st) {
    constexpr int rows_per_block = R * W;
    const dim3 grid((N + rows_per_block - 1) / rows_per_block);
    const dim3 block(W * 32);
    paia::gemv_bf16_kernel<R, U, W, S, H><<<grid, block, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(Wt), reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return static_cast<int>(cudaGetLastError());
}

struct Variant { int R, U, W, S, H; LaunchFn fn; };

#define V_(R, U, W, S, H) {R, U, W, S, H, &launch<R, U, W, (S) != 0, (H) != 0>},
#define EXP_W(R, U, S, H) V_(R, U, 4, S, H) V_(R, U, 8, S, H)
#define EXP_H(R, U, S) EXP_W(R, U, S, 0) EXP_W(R, U, S, 1)
#define EXP_S(R, U) EXP_H(R, U, 0) EXP_H(R, U, 1)
#define EXP_U(R) EXP_S(R, 1) EXP_S(R, 2) EXP_S(R, 4) EXP_S(R, 8)

const Variant kVariants[] = { EXP_U(1) EXP_U(2) EXP_U(4) };
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

}  // namespace

extern "C" {

int paia_gemv_bf16_variant_count() { return kNumVariants; }

// Returns 0 on success.
int paia_gemv_bf16_variant_info(int v, int* R, int* U, int* W, int* S, int* H) {
    if (v < 0 || v >= kNumVariants) return -1;
    *R = kVariants[v].R; *U = kVariants[v].U; *W = kVariants[v].W; *S = kVariants[v].S; *H = kVariants[v].H;
    return 0;
}

// y[N] = W[N,K] @ x[K], bf16 in/out, fp32 accumulate. Returns 0 on success, cudaError_t (>0) on launch failure,
// -1 bad variant, -2 bad shape or alignment.
int paia_gemv_bf16(const void* W, const void* x, void* y, int N, int K, int variant, void* stream) {
    if (variant < 0 || variant >= kNumVariants) return -1;
    if (N <= 0 || K <= 0 || (K & 7) != 0) return -2;
    if ((reinterpret_cast<uintptr_t>(W) | reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(y)) & 15u) return -2;
    return kVariants[variant].fn(W, x, y, N, K, static_cast<cudaStream_t>(stream));
}

}  // extern "C"
