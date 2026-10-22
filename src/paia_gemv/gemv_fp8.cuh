// B3: FP8 (E4M3) weight-only GEMV, W8A16. y[N] = (scale[row] * decode(Wq[row,:])) . x, x and y bf16, fp32 accumulate.
// Wq is [N,K] uint8 row-major, scale is [N] fp32 (per output row). Bytes per weight 1 + 4/K.
//
// Same skeleton as S1 (src/README.md): one warp owns R rows, a lane loads 16 B = 16 weights per row per chunk, a warp
// covers 512 columns per chunk, R*U independent 16 B loads are issued before any is consumed.
//
// Decode is pure integer bit work, exact for every finite code, and it avoids a per-weight multiply:
//   place the 7 magnitude bits of the e4m3 byte at bits 13..7 of a half: exponent field = e (bias 15 instead of 7),
//   mantissa 3 bits at 9..7. The half then equals true_value / 256 exactly, including subnormals (e == 0), and the sign
//   bit moves to bit 15. The factor 256 is folded into the per-row scale at the end. (Codes 0x7F and 0xFF are NaN in
//   e4m3fn; the quantizer never emits them, and this decode turns them into +-480/256 instead of NaN.)
// Requirements: K % 16 == 0, Wq and x 16 B aligned.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace paia {

__device__ __forceinline__ uint4 fp8_ld_w(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint4 fp8_ld_x(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// 4 e4m3 bytes (one 32-bit word) -> 4 floats equal to true_value / 256.
__device__ __forceinline__ void fp8x4_decode(unsigned int w, float& f0, float& f1, float& f2, float& f3) {
    const unsigned int lo = __byte_perm(w, 0u, 0x4140u);   // [0 b1 0 b0]
    const unsigned int hi = __byte_perm(w, 0u, 0x4342u);   // [0 b3 0 b2]
    const unsigned int hl = ((lo << 7) & 0x3F803F80u) | ((lo << 8) & 0x80008000u);
    const unsigned int hh = ((hi << 7) & 0x3F803F80u) | ((hi << 8) & 0x80008000u);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&hl));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hh));
    f0 = a.x; f1 = a.y; f2 = b.x; f3 = b.y;
}
__device__ __forceinline__ float bf_lo_(unsigned int v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi_(unsigned int v) { return __uint_as_float(v & 0xFFFF0000u); }

// 16 weights (uint4) . 16 x values (two uint4 of bf16) -> partial dot, still scaled by 1/256.
__device__ __forceinline__ float fp8_dot16(const uint4 w, const uint4 xa, const uint4 xb) {
    float f0, f1, f2, f3, p0 = 0.f, p1 = 0.f;
    fp8x4_decode(w.x, f0, f1, f2, f3);
    p0 = fmaf(f0, bf_lo_(xa.x), p0); p1 = fmaf(f1, bf_hi_(xa.x), p1); p0 = fmaf(f2, bf_lo_(xa.y), p0); p1 = fmaf(f3, bf_hi_(xa.y), p1);
    fp8x4_decode(w.y, f0, f1, f2, f3);
    p0 = fmaf(f0, bf_lo_(xa.z), p0); p1 = fmaf(f1, bf_hi_(xa.z), p1); p0 = fmaf(f2, bf_lo_(xa.w), p0); p1 = fmaf(f3, bf_hi_(xa.w), p1);
    fp8x4_decode(w.z, f0, f1, f2, f3);
    p0 = fmaf(f0, bf_lo_(xb.x), p0); p1 = fmaf(f1, bf_hi_(xb.x), p1); p0 = fmaf(f2, bf_lo_(xb.y), p0); p1 = fmaf(f3, bf_hi_(xb.y), p1);
    fp8x4_decode(w.w, f0, f1, f2, f3);
    p0 = fmaf(f0, bf_lo_(xb.z), p0); p1 = fmaf(f1, bf_hi_(xb.z), p1); p0 = fmaf(f2, bf_lo_(xb.w), p0); p1 = fmaf(f3, bf_hi_(xb.w), p1);
    return p0 + p1;
}

template <int R, int UC>
__device__ __forceinline__ void fp8_chunk_batch(const unsigned char* const (&wrow)[R], const __nv_bfloat16* __restrict__ x,
                                                int c0, int lane, float (&acc)[R])
{
    uint4 xv[UC][2];
    uint4 wv[R][UC];
#pragma unroll
    for (int u = 0; u < UC; ++u) {
        const int col = ((c0 + u) << 9) + lane * 16;
        xv[u][0] = fp8_ld_x(x + col); xv[u][1] = fp8_ld_x(x + col + 8);
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r][u] = fp8_ld_w(wrow[r] + col);
    }
#pragma unroll
    for (int u = 0; u < UC; ++u)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += fp8_dot16(wv[r][u], xv[u][0], xv[u][1]);
}

template <int R>
__device__ __forceinline__ void fp8_chunk_partial(const unsigned char* const (&wrow)[R], const __nv_bfloat16* __restrict__ x,
                                                  int c, int K, int lane, float (&acc)[R])
{
    const int col = (c << 9) + lane * 16;
    if (col >= K) return;
    const uint4 xa = fp8_ld_x(x + col), xb = fp8_ld_x(x + col + 8);
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] += fp8_dot16(fp8_ld_w(wrow[r] + col), xa, xb);
}

template <int R, int U, int W, bool H>
__global__ void __launch_bounds__(W * 32, H ? 1 : 0)
gemv_fp8_kernel(const unsigned char* __restrict__ Wq, const float* __restrict__ scale, const __nv_bfloat16* __restrict__ x,
                __nv_bfloat16* __restrict__ y, int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * W + (threadIdx.x >> 5)) * R;
    if (row0 >= N) return;
    const unsigned char* wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wrow[r] = Wq + (size_t)min(row0 + r, N - 1) * (size_t)K;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;

    const int nfull = K >> 9;
    int c0 = 0;
    for (; c0 + U <= nfull; c0 += U) fp8_chunk_batch<R, U>(wrow, x, c0, lane, acc);
    if constexpr (U > 2) { if (nfull - c0 >= 2) { fp8_chunk_batch<R, 2>(wrow, x, c0, lane, acc); c0 += 2; } }
    if constexpr (U > 1) { if (nfull - c0 >= 1) { fp8_chunk_batch<R, 1>(wrow, x, c0, lane, acc); c0 += 1; } }
    if ((nfull << 9) < K) fp8_chunk_partial<R>(wrow, x, nfull, K, lane, acc);

#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = acc[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0 && row0 + r < N) y[row0 + r] = __float2bfloat16_rn(s * (scale[row0 + r] * 256.f));
    }
}

}  // namespace paia
