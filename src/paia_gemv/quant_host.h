// Host-side quantizers and reference decoders for the two quantized formats (pure C++, no CUDA). Used by the standalone
// check (src/tests/gemv_quant_check.cu) and self-tested on a CPU (src/tests/quant_host_selftest.cpp).
//
// FP8 W8A16: weights E4M3 (OCP "fn" variant: bias 7, no infinities, S.1111.111 is NaN, max finite 448), one fp32 scale
//   per output row, scale = max|w| / 448, q = round-to-nearest-even(w / scale). Codes 0x7F and 0xFF are never produced.
// INT4 W4A16: unsigned nibbles with zero point 8, group size 128 along K, one fp16 scale per group,
//   scale = fp16(max|w| / 7), n = clamp(rint(w / scale), -7, 7) + 8, so n is in 1..15 and w ~ (n - 8) * scale.
//   Packing: byte j of a row holds weight 2j in its low nibble and weight 2j+1 in its high nibble.
//   Scales are stored [N][K/128] row-major. 4.125 bits per weight.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace paia_host {

inline float e4m3_decode(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const float v = (e == 0) ? std::ldexp((float)m / 8.f, -6) : std::ldexp(1.f + (float)m / 8.f, e - 7);
    return s ? -v : v;
}

// Round to nearest, ties to the even code (even mantissa lsb), clamped to +-448.
inline uint8_t e4m3_encode(float x) {
    static float tab[127];
    static bool init = false;
    if (!init) { for (int c = 0; c < 127; ++c) tab[c] = e4m3_decode((uint8_t)c); init = true; }
    const uint8_t sign = std::signbit(x) ? 0x80 : 0;
    const float a = std::min(std::fabs(x), 448.f);
    int lo = 0, hi = 126;
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (tab[mid] <= a) lo = mid; else hi = mid - 1; }
    int c = lo;
    if (c < 126) {
        const float d_lo = a - tab[c], d_hi = tab[c + 1] - a;
        if (d_hi < d_lo || (d_hi == d_lo && ((c + 1) & 1) == 0)) c += 1;
    }
    return sign | (uint8_t)c;
}

inline uint16_t f32_to_f16_bits(float f) {   // round to nearest even, handles subnormals and overflow to inf
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (m ? 0x200u : 0));
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        const int shift = 14 - e;
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t)(sign | r);
    }
    uint32_t r = ((uint32_t)e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t)(sign | r);
}

inline float f16_bits_to_f32(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(1.f + (float)m / 1024.f, e - 15);
    return s ? -v : v;
}

// W: N x K row-major floats. q: N x K bytes. scale: N floats.
inline void quantize_fp8_rowwise(const float* W, int N, int K, uint8_t* q, float* scale) {
    for (int n = 0; n < N; ++n) {
        float amax = 0;
        for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(W[(size_t)n * K + k]));
        const float s = amax > 0 ? amax / 448.f : 1.f;
        scale[n] = s;
        for (int k = 0; k < K; ++k) q[(size_t)n * K + k] = e4m3_encode(W[(size_t)n * K + k] / s);
    }
}
inline float dequant_fp8(const uint8_t* q, const float* scale, int K, int n, int k) {
    return e4m3_decode(q[(size_t)n * K + k]) * scale[n];
}

// K % 128 == 0. packed: N x K/2 bytes. scales: N x K/128 fp16 bit patterns.
inline void quantize_int4_g128(const float* W, int N, int K, uint8_t* packed, uint16_t* scales) {
    const int G = K / 128;
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            const float* w = W + (size_t)n * K + g * 128;
            float amax = 0;
            for (int i = 0; i < 128; ++i) amax = std::max(amax, std::fabs(w[i]));
            const uint16_t sb = f32_to_f16_bits(amax > 0 ? amax / 7.f : 1.f);
            scales[(size_t)n * G + g] = sb;
            float sf = f16_bits_to_f32(sb);
            if (!(sf > 0)) sf = 1.f;
            for (int i = 0; i < 128; i += 2) {
                int q0 = (int)std::nearbyint(w[i] / sf), q1 = (int)std::nearbyint(w[i + 1] / sf);
                q0 = std::min(7, std::max(-7, q0)) + 8; q1 = std::min(7, std::max(-7, q1)) + 8;
                packed[(size_t)n * (K / 2) + (g * 128 + i) / 2] = (uint8_t)(q0 | (q1 << 4));
            }
        }
    }
}
inline float dequant_int4(const uint8_t* packed, const uint16_t* scales, int K, int n, int k) {
    const uint8_t byte = packed[(size_t)n * (K / 2) + k / 2];
    const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
    return (float)(nib - 8) * f16_bits_to_f32(scales[(size_t)n * (K / 128) + k / 128]);
}

}  // namespace paia_host
