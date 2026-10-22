// S1: BF16 GEMV for batch-1 decode, y[N] = W[N,K] @ x[K]  (W row-major, i.e. the nn.Linear layout).
//
// Design goal (see src/README.md): keep enough bytes in flight per SM to saturate DRAM, with a small register
// budget so more warps are resident. cuBLAS gemvx/v6 loses 30% on GB10 because each warp keeps about half as many
// bytes in flight (16-lane rows); v7 (32-lane rows) reaches 87-98% of B0.
//
// Mapping: one warp owns R consecutive rows. A lane loads 16 B (8 bf16) per row per chunk, a warp covers 256 columns
// of a row per chunk, and every lane issues R*U independent 16 B loads before it consumes any of them, so a warp has
// R*U*512 B in flight. Accumulation is fp32 (two independent accumulators per row to shorten the FMA chain), reduction
// is a warp shuffle, one bf16 store per row. No split-K, no shared memory, no atomics.
//
// Requirements: K % 8 == 0 (16 B aligned row pitch), W and x 16 B aligned. K need not be a multiple of 256.
//
// Template knobs (all swept by src/tools/dev_sweep_bf16.py, then frozen per shape):
//   R  rows per warp            {1,2,4}   x is re-read per warp, so L2->SM x traffic is 1/R of weight traffic
//   U  chunks in flight per row {1,2,4,8}
//   W  warps per block          {2,4,8}
//   S  weight loads use ld.global.nc.L1::no_allocate (keeps x resident in L1) instead of the default __ldg path
//   H  hoist: __launch_bounds__(W*32, 1). With the default bounds ptxas caps registers near the full-occupancy limit
//      (about 40) and does NOT keep all R*U loads in flight (measured with ptxas -v: R=2,U=8 uses 40 regs, 123 with H).
//      H lets ptxas allocate the registers the batch needs, at lower occupancy; the sweep decides which wins.
#pragma once
#include <cuda_bf16.h>

namespace paia {

__device__ __forceinline__ uint4 ld_weights_stream(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint4 ld_weights_plain(const void* p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ uint4 ld_x(const void* p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}
// bf16 -> fp32 is a shift: the bf16 bits are the high half of the fp32 bits.
__device__ __forceinline__ float bf_lo(unsigned int v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(unsigned int v) { return __uint_as_float(v & 0xFFFF0000u); }

// Issue R*UC independent 16 B weight loads (+ UC x loads) back to back, then consume them. No predicates: the caller
// guarantees all UC chunks are full, so the loads are straight-line code the scheduler can keep in flight together.
template <int R, int UC, bool S>
__device__ __forceinline__ void chunk_batch(const __nv_bfloat16* const (&wrow)[R], const __nv_bfloat16* __restrict__ x,
                                            int c0, int lane, float (&acc)[R][2])
{
    uint4 xv[UC];
    uint4 wv[R][UC];
#pragma unroll
    for (int u = 0; u < UC; ++u) {
        const int col = ((c0 + u) << 8) + lane * 8;
        xv[u] = ld_x(x + col);
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r][u] = S ? ld_weights_stream(wrow[r] + col) : ld_weights_plain(wrow[r] + col);
    }
#pragma unroll
    for (int u = 0; u < UC; ++u) {
        const float x0 = bf_lo(xv[u].x), x1 = bf_hi(xv[u].x), x2 = bf_lo(xv[u].y), x3 = bf_hi(xv[u].y);
        const float x4 = bf_lo(xv[u].z), x5 = bf_hi(xv[u].z), x6 = bf_lo(xv[u].w), x7 = bf_hi(xv[u].w);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 w = wv[r][u];
            float a = acc[r][u & 1];
            a = fmaf(bf_lo(w.x), x0, a); a = fmaf(bf_hi(w.x), x1, a);
            a = fmaf(bf_lo(w.y), x2, a); a = fmaf(bf_hi(w.y), x3, a);
            a = fmaf(bf_lo(w.z), x4, a); a = fmaf(bf_hi(w.z), x5, a);
            a = fmaf(bf_lo(w.w), x6, a); a = fmaf(bf_hi(w.w), x7, a);
            acc[r][u & 1] = a;
        }
    }
}

// The last chunk when K % 256 != 0: only lanes whose 8 columns are inside K load (K % 8 == 0 is required).
template <int R, bool S>
__device__ __forceinline__ void chunk_partial(const __nv_bfloat16* const (&wrow)[R], const __nv_bfloat16* __restrict__ x,
                                              int c, int K, int lane, float (&acc)[R][2])
{
    const int col = (c << 8) + lane * 8;
    if (col >= K) return;
    const uint4 xv = ld_x(x + col);
    const float x0 = bf_lo(xv.x), x1 = bf_hi(xv.x), x2 = bf_lo(xv.y), x3 = bf_hi(xv.y);
    const float x4 = bf_lo(xv.z), x5 = bf_hi(xv.z), x6 = bf_lo(xv.w), x7 = bf_hi(xv.w);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const uint4 w = S ? ld_weights_stream(wrow[r] + col) : ld_weights_plain(wrow[r] + col);
        float a = acc[r][0];
        a = fmaf(bf_lo(w.x), x0, a); a = fmaf(bf_hi(w.x), x1, a);
        a = fmaf(bf_lo(w.y), x2, a); a = fmaf(bf_hi(w.y), x3, a);
        a = fmaf(bf_lo(w.z), x4, a); a = fmaf(bf_hi(w.z), x5, a);
        a = fmaf(bf_lo(w.w), x6, a); a = fmaf(bf_hi(w.w), x7, a);
        acc[r][0] = a;
    }
}

template <int R, int U, int W, bool S, bool H>
__global__ void __launch_bounds__(W * 32, H ? 1 : 0)
gemv_bf16_kernel(const __nv_bfloat16* __restrict__ Wt, const __nv_bfloat16* __restrict__ x,
                 __nv_bfloat16* __restrict__ y, int N, int K)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * W + warp) * R;
    if (row0 >= N) return;                       // warp-uniform; there is no block-level sync in this kernel

    const __nv_bfloat16* wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, N - 1);    // partial last group: clamp the loads, mask the store
        wrow[r] = Wt + (size_t)row * (size_t)K;
    }

    float acc[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) { acc[r][0] = 0.f; acc[r][1] = 0.f; }

    const int nfull = K >> 8;                    // full 256-column chunks
    int c0 = 0;
    for (; c0 + U <= nfull; c0 += U) chunk_batch<R, U, S>(wrow, x, c0, lane, acc);
    // Remainder of full chunks (< U), largest power-of-two batches first so loads still overlap.
    if constexpr (U > 4) { if (nfull - c0 >= 4) { chunk_batch<R, 4, S>(wrow, x, c0, lane, acc); c0 += 4; } }
    if constexpr (U > 2) { if (nfull - c0 >= 2) { chunk_batch<R, 2, S>(wrow, x, c0, lane, acc); c0 += 2; } }
    if constexpr (U > 1) { if (nfull - c0 >= 1) { chunk_batch<R, 1, S>(wrow, x, c0, lane, acc); c0 += 1; } }
    if ((nfull << 8) < K) chunk_partial<R, S>(wrow, x, nfull, K, lane, acc);

#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = acc[r][0] + acc[r][1];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0 && row0 + r < N) y[row0 + r] = __float2bfloat16_rn(s);
    }
}

}  // namespace paia
