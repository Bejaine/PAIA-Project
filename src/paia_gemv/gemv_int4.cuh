// W4A16 INT4 GEMV (group 128, fp16 scale, zero point 8). Format and packing: see quant_host.h.
// Wq [N, K/2] bytes (byte j = weights 2j low nibble, 2j+1 high nibble), scales [N, K/128] fp16, x and y bf16.
//
// gemv_int4_kernel (K2, fused): one warp owns R rows; a lane loads 16 B = 32 weights per row per chunk plus the one
//   scale of its group (a lane's 32 weights lie inside one 128-weight group); a warp covers 1024 columns per chunk.
//   Natural packing needs no repack because the dequant works on the byte structure directly:
//     even = w & 0x0F0F0F0F   -> the 4 even-position weights of a 32-bit word, one per byte
//     odd  = (w >> 4) & 0x0F0F0F0F
//   and a byte is turned into the float 8388608 + n by one PRMT against 0x4B000000; subtracting 8388616 gives n - 8
//   exactly. Even-position weights pair with the low bf16 half of each x word and odd ones with the high half.
//   Per weight: PRMT + FADD + FFMA, plus amortised nibble masks and the x conversion (shared by the R rows).
// gemv_int4_naive_kernel (K1): the deliberately naive control: one row per warp, one 4 B load (8 weights) per lane per
//   iteration, scale loaded from global every iteration, shift/mask/I2F per weight, no unrolling.
// dequant_int4_kernel: writes a bf16 copy of the weights (the unfused control, then S1's BF16 GEMV reads it back).
// Requirements: K % 128 == 0 (so 16 B alignment of rows holds too), Wq and x 16 B aligned, scales 2 B aligned.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace paia {

__device__ __forceinline__ uint4 i4_ld_w(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint4 i4_ld_x(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ float i4_bf_lo(unsigned int v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float i4_bf_hi(unsigned int v) { return __uint_as_float(v & 0xFFFF0000u); }

// byte k (0..3) of m holds a nibble 0..15 -> (nibble - 8) as float, exactly.
template <int K_>
__device__ __forceinline__ float nib_to_f(unsigned int m) {
    return __uint_as_float(__byte_perm(m, 0x4B000000u, 0x7440u | K_)) - 8388616.f;
}

// 32 weights (uint4) . 32 x values (4 uint4 of bf16) -> sum of (n - 8) * x over the lane's slice.
__device__ __forceinline__ float int4_dot32(const uint4 w, const uint4 x0, const uint4 x1, const uint4 x2, const uint4 x3) {
    float p0 = 0.f, p1 = 0.f;
#define PAIA_I4_WORD(WORD, XV)                                                                              \
    {                                                                                                       \
        const unsigned int ev = (WORD) & 0x0F0F0F0Fu, od = ((WORD) >> 4) & 0x0F0F0F0Fu;                     \
        p0 = fmaf(nib_to_f<0>(ev), i4_bf_lo((XV).x), p0); p1 = fmaf(nib_to_f<0>(od), i4_bf_hi((XV).x), p1); \
        p0 = fmaf(nib_to_f<1>(ev), i4_bf_lo((XV).y), p0); p1 = fmaf(nib_to_f<1>(od), i4_bf_hi((XV).y), p1); \
        p0 = fmaf(nib_to_f<2>(ev), i4_bf_lo((XV).z), p0); p1 = fmaf(nib_to_f<2>(od), i4_bf_hi((XV).z), p1); \
        p0 = fmaf(nib_to_f<3>(ev), i4_bf_lo((XV).w), p0); p1 = fmaf(nib_to_f<3>(od), i4_bf_hi((XV).w), p1); \
    }
    PAIA_I4_WORD(w.x, x0) PAIA_I4_WORD(w.y, x1) PAIA_I4_WORD(w.z, x2) PAIA_I4_WORD(w.w, x3)
#undef PAIA_I4_WORD
    return p0 + p1;
}

template <int R, int UC>
__device__ __forceinline__ void i4_chunk_batch(const unsigned char* const (&wrow)[R], const __half* const (&srow)[R],
                                               const __nv_bfloat16* __restrict__ x, int c0, int lane, float (&acc)[R])
{
    uint4 xv[UC][4];
    uint4 wv[R][UC];
    __half sv[R][UC];
#pragma unroll
    for (int u = 0; u < UC; ++u) {
        const int col = ((c0 + u) << 10) + lane * 32;
#pragma unroll
        for (int j = 0; j < 4; ++j) xv[u][j] = i4_ld_x(x + col + 8 * j);
#pragma unroll
        for (int r = 0; r < R; ++r) { wv[r][u] = i4_ld_w(wrow[r] + (col >> 1)); sv[r][u] = srow[r][col >> 7]; }
    }
#pragma unroll
    for (int u = 0; u < UC; ++u)
#pragma unroll
        for (int r = 0; r < R; ++r)
            acc[r] = fmaf(__half2float(sv[r][u]), int4_dot32(wv[r][u], xv[u][0], xv[u][1], xv[u][2], xv[u][3]), acc[r]);
}

template <int R>
__device__ __forceinline__ void i4_chunk_partial(const unsigned char* const (&wrow)[R], const __half* const (&srow)[R],
                                                 const __nv_bfloat16* __restrict__ x, int c, int K, int lane, float (&acc)[R])
{
    const int col = (c << 10) + lane * 32;
    if (col >= K) return;
    const uint4 x0 = i4_ld_x(x + col), x1 = i4_ld_x(x + col + 8), x2 = i4_ld_x(x + col + 16), x3 = i4_ld_x(x + col + 24);
#pragma unroll
    for (int r = 0; r < R; ++r)
        acc[r] = fmaf(__half2float(srow[r][col >> 7]), int4_dot32(i4_ld_w(wrow[r] + (col >> 1)), x0, x1, x2, x3), acc[r]);
}

template <int R, int U, int W, bool H>
__global__ void __launch_bounds__(W * 32, H ? 1 : 0)
gemv_int4_kernel(const unsigned char* __restrict__ Wq, const __half* __restrict__ scales, const __nv_bfloat16* __restrict__ x,
                 __nv_bfloat16* __restrict__ y, int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * W + (threadIdx.x >> 5)) * R;
    if (row0 >= N) return;
    const unsigned char* wrow[R];
    const __half* srow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, N - 1);
        wrow[r] = Wq + (size_t)row * (size_t)(K >> 1);
        srow[r] = scales + (size_t)row * (size_t)(K >> 7);
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;

    const int nfull = K >> 10;
    int c0 = 0;
    for (; c0 + U <= nfull; c0 += U) i4_chunk_batch<R, U>(wrow, srow, x, c0, lane, acc);
    if constexpr (U > 2) { if (nfull - c0 >= 2) { i4_chunk_batch<R, 2>(wrow, srow, x, c0, lane, acc); c0 += 2; } }
    if constexpr (U > 1) { if (nfull - c0 >= 1) { i4_chunk_batch<R, 1>(wrow, srow, x, c0, lane, acc); c0 += 1; } }
    if ((nfull << 10) < K) i4_chunk_partial<R>(wrow, srow, x, nfull, K, lane, acc);

#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = acc[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0 && row0 + r < N) y[row0 + r] = __float2bfloat16_rn(s);
    }
}

// K1: naive control. One row per warp, W warps per block.
template <int W>
__global__ void __launch_bounds__(W * 32)
gemv_int4_naive_kernel(const unsigned char* __restrict__ Wq, const __half* __restrict__ scales, const __nv_bfloat16* __restrict__ x,
                       __nv_bfloat16* __restrict__ y, int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * W + (threadIdx.x >> 5);
    if (row >= N) return;
    const unsigned int* wr = reinterpret_cast<const unsigned int*>(Wq + (size_t)row * (size_t)(K >> 1));
    const __half* sr = scales + (size_t)row * (size_t)(K >> 7);
    float acc = 0.f;
#pragma unroll 1
    for (int col = lane * 8; col < K; col += 256) {
        const unsigned int w = wr[col >> 3];
        const float s = __half2float(sr[col >> 7]);
        const uint4 xv = *reinterpret_cast<const uint4*>(x + col);
        const unsigned int xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int n = (int)((w >> (4 * j)) & 0xFu);
            const float xf = (j & 1) ? __uint_as_float(xs[j >> 1] & 0xFFFF0000u) : __uint_as_float(xs[j >> 1] << 16);
            acc = fmaf((float)(n - 8) * s, xf, acc);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = __float2bfloat16_rn(acc);
}

// Unfused control, stage 1: bf16 copy of the dequantized weights, 8 weights per thread.
__global__ void dequant_int4_kernel(const unsigned char* __restrict__ Wq, const __half* __restrict__ scales,
                                    __nv_bfloat16* __restrict__ out, size_t n_words, int K)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_words; i += (size_t)gridDim.x * blockDim.x) {
        const unsigned int w = reinterpret_cast<const unsigned int*>(Wq)[i];
        const size_t elem = i * 8;
        const size_t row = elem / (size_t)K;
        const int k = (int)(elem - row * (size_t)K);
        const float s = __half2float(scales[row * (size_t)(K >> 7) + (k >> 7)]);
        uint4 o;
        unsigned int* po = reinterpret_cast<unsigned int*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a = (float)((int)((w >> (8 * j)) & 0xFu) - 8) * s, b = (float)((int)((w >> (8 * j + 4)) & 0xFu) - 8) * s;
            const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
            po[j] = *reinterpret_cast<const unsigned int*>(&v);
        }
        *reinterpret_cast<uint4*>(out + elem) = o;
    }
}

}  // namespace paia
