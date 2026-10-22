// C ABI for the quantized kernels (FP8 W8A16 and INT4 W4A16 g128): variant tables, launchers, argument checks.
// Return codes: 0 ok, >0 cudaError_t, -1 bad variant, -2 bad shape or alignment.
#include <stdint.h>
#include <cuda_runtime.h>
#include "gemv_fp8.cuh"
#include "gemv_int4.cuh"

extern "C" {
int paia_gemv_bf16_variant_count();
int paia_gemv_bf16_variant_info(int v, int* R, int* U, int* W, int* S, int* H);
int paia_gemv_bf16(const void* W, const void* x, void* y, int N, int K, int variant, void* stream);
}

namespace {
inline bool aligned(const void* p, unsigned a) { return (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0; }

using Fp8Fn = int (*)(const void*, const void*, const void*, void*, int, int, cudaStream_t);
template <int R, int U, int W, bool H>
int launch_fp8(const void* Wq, const void* sc, const void* x, void* y, int N, int K, cudaStream_t st) {
    paia::gemv_fp8_kernel<R, U, W, H><<<dim3((N + R * W - 1) / (R * W)), dim3(W * 32), 0, st>>>(
        (const unsigned char*)Wq, (const float*)sc, (const __nv_bfloat16*)x, (__nv_bfloat16*)y, N, K);
    return (int)cudaGetLastError();
}
template <int R, int U, int W, bool H>
int launch_int4(const void* Wq, const void* sc, const void* x, void* y, int N, int K, cudaStream_t st) {
    paia::gemv_int4_kernel<R, U, W, H><<<dim3((N + R * W - 1) / (R * W)), dim3(W * 32), 0, st>>>(
        (const unsigned char*)Wq, (const __half*)sc, (const __nv_bfloat16*)x, (__nv_bfloat16*)y, N, K);
    return (int)cudaGetLastError();
}
struct QVariant { int R, U, W, H; Fp8Fn fn; };

#define Q_(L, R, U, W, H) {R, U, W, H, &L<R, U, W, (H) != 0>},
#define QW_(L, R, U, H) Q_(L, R, U, 4, H) Q_(L, R, U, 8, H)
#define QH_(L, R, U) QW_(L, R, U, 0) QW_(L, R, U, 1)
#define QU4_(L, R) QH_(L, R, 1) QH_(L, R, 2) QH_(L, R, 4)
#define QU3_(L, R) QH_(L, R, 1) QH_(L, R, 2) QH_(L, R, 4)
const QVariant kFp8[] = { QU3_(launch_fp8, 1) QU3_(launch_fp8, 2) QU3_(launch_fp8, 4) };
const QVariant kInt4[] = { QU4_(launch_int4, 1) QU4_(launch_int4, 2) QU4_(launch_int4, 4) QU4_(launch_int4, 8) };
constexpr int kNumFp8 = sizeof(kFp8) / sizeof(kFp8[0]);
constexpr int kNumInt4 = sizeof(kInt4) / sizeof(kInt4[0]);

int info(const QVariant* t, int n, int v, int* R, int* U, int* W, int* H) {
    if (v < 0 || v >= n) return -1;
    *R = t[v].R; *U = t[v].U; *W = t[v].W; *H = t[v].H;
    return 0;
}
int default_bf16_variant() {   // R=2 U=4 W=4 S=0 H=1, looked up so it survives reordering
    static int v = -1;
    if (v < 0) {
        for (int i = 0; i < paia_gemv_bf16_variant_count(); ++i) {
            int R, U, W, S, H; paia_gemv_bf16_variant_info(i, &R, &U, &W, &S, &H);
            if (R == 2 && U == 4 && W == 4 && S == 0 && H == 1) { v = i; break; }
        }
    }
    return v < 0 ? 0 : v;
}
}  // namespace

extern "C" {

int paia_gemv_fp8_variant_count() { return kNumFp8; }
int paia_gemv_fp8_variant_info(int v, int* R, int* U, int* W, int* H) { return info(kFp8, kNumFp8, v, R, U, W, H); }
// Wq [N,K] e4m3 bytes, scale [N] fp32, x [K] bf16, y [N] bf16.
int paia_gemv_fp8(const void* Wq, const void* scale, const void* x, void* y, int N, int K, int variant, void* stream) {
    if (variant < 0 || variant >= kNumFp8) return -1;
    if (N <= 0 || K <= 0 || (K & 15) || !aligned(Wq, 16) || !aligned(x, 16) || !aligned(scale, 4) || !aligned(y, 2)) return -2;
    return kFp8[variant].fn(Wq, scale, x, y, N, K, static_cast<cudaStream_t>(stream));
}

int paia_gemv_int4_variant_count() { return kNumInt4; }
int paia_gemv_int4_variant_info(int v, int* R, int* U, int* W, int* H) { return info(kInt4, kNumInt4, v, R, U, W, H); }
// Wq [N,K/2] packed nibbles, scales [N,K/128] fp16, x [K] bf16, y [N] bf16. K % 128 == 0.
int paia_gemv_int4(const void* Wq, const void* scales, const void* x, void* y, int N, int K, int variant, void* stream) {
    if (variant < 0 || variant >= kNumInt4) return -1;
    if (N <= 0 || K <= 0 || (K & 127) || !aligned(Wq, 16) || !aligned(x, 16) || !aligned(scales, 2) || !aligned(y, 2)) return -2;
    return kInt4[variant].fn(Wq, scales, x, y, N, K, static_cast<cudaStream_t>(stream));
}

// K1 naive control (one row per warp, 4 warps per block).
int paia_gemv_int4_naive(const void* Wq, const void* scales, const void* x, void* y, int N, int K, void* stream) {
    if (N <= 0 || K <= 0 || (K & 127) || !aligned(Wq, 16) || !aligned(x, 16) || !aligned(scales, 2) || !aligned(y, 2)) return -2;
    paia::gemv_int4_naive_kernel<4><<<dim3((N + 3) / 4), dim3(128), 0, static_cast<cudaStream_t>(stream)>>>(
        (const unsigned char*)Wq, (const __half*)scales, (const __nv_bfloat16*)x, (__nv_bfloat16*)y, N, K);
    return (int)cudaGetLastError();
}

// Unfused control: dequantize to a bf16 scratch matrix (2*N*K bytes, 16 B aligned, caller-owned), then S1's BF16 GEMV.
int paia_gemv_int4_unfused(const void* Wq, const void* scales, const void* x, void* y, void* scratch_bf16, int N, int K, void* stream) {
    if (N <= 0 || K <= 0 || (K & 127) || !aligned(Wq, 16) || !aligned(x, 16) || !aligned(scales, 2) || !aligned(scratch_bf16, 16)) return -2;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    const size_t words = (size_t)N * (size_t)K / 8;
    paia::dequant_int4_kernel<<<dim3(2048), dim3(256), 0, st>>>((const unsigned char*)Wq, (const __half*)scales,
                                                                  (__nv_bfloat16*)scratch_bf16, words, K);
    int rc = (int)cudaGetLastError();
    if (rc) return rc;
    return paia_gemv_bf16(scratch_bf16, x, y, N, K, default_bf16_variant(), stream);
}

}  // extern "C"
