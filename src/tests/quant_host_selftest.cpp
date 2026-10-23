// CPU self-test of quant_host.h (g++ -std=c++17 -I src/paia_gemv src/tests/quant_host_selftest.cpp && ./a.out).
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "quant_host.h"
using namespace paia_host;
#define REQUIRE(c) do { if (!(c)) { printf("FAILED: %s (line %d)\n", #c, __LINE__); return 1; } } while (0)
int main() {
    // 1. e4m3: max finite 448, min subnormal 2^-9, NaN codes
    REQUIRE(e4m3_decode(0x7E) == 448.f); REQUIRE(e4m3_decode(0x01) == std::ldexp(1.f, -9)); REQUIRE(std::isnan(e4m3_decode(0x7F)));
    REQUIRE(e4m3_decode(0x38) == 1.f); REQUIRE(e4m3_decode(0xB8) == -1.f);
    // 2. every finite code round-trips; encode never emits the NaN codes
    for (int c = 0; c < 256; ++c) { if ((c & 0x7F) == 0x7F) continue; REQUIRE(e4m3_encode(e4m3_decode((uint8_t)c)) == (c == 0x80 ? 0x80 : c)); }
    // 3. encode picks the nearest code (brute force), ties to even, on random inputs
    std::mt19937 rng(1); std::uniform_real_distribution<float> U(-500.f, 500.f);
    for (int i = 0; i < 200000; ++i) {
        const float x = (i % 3 == 0) ? U(rng) * 0.001f : U(rng);
        const uint8_t got = e4m3_encode(x);
        float best = 1e30f; int bc = -1;
        for (int c = 0; c < 127; ++c) { const float d = std::fabs(std::min(std::fabs(x), 448.f) - e4m3_decode((uint8_t)c)); if (d < best || (d == best && c % 2 == 0 && bc % 2 == 1)) { best = d; bc = c; } }
        REQUIRE((got & 0x7F) == bc); REQUIRE(((got >> 7) != 0) == std::signbit(x));
    }
    // 4. fp16 conversion against all 65536 patterns and the rounding of random floats
    for (uint32_t h = 0; h < 65536; ++h) { const float f = f16_bits_to_f32((uint16_t)h); if (std::isnan(f)) continue; REQUIRE(f32_to_f16_bits(f) == h); }
    for (int i = 0; i < 100000; ++i) {   // RNE: result must be a nearest representable half
        const float x = U(rng) * (i % 5 == 0 ? 1e-6f : 1e-3f);
        const uint16_t h = f32_to_f16_bits(x); const float r = f16_bits_to_f32(h);
        const float up = f16_bits_to_f32((uint16_t)(h + 1)), dn = h ? f16_bits_to_f32((uint16_t)(h - 1)) : r;
        REQUIRE(std::fabs(r - x) <= std::fabs(up - x) + 1e-30f); REQUIRE(std::fabs(r - x) <= std::fabs(dn - x) + 1e-30f || (h & 0x8000));
    }
    // 5. int4 round trip: dequant error bounded by half a step, nibbles in 1..15
    const int N = 5, K = 384; std::vector<float> W((size_t)N * K); for (auto& w : W) w = U(rng) * 1e-4f;
    std::vector<uint8_t> pk((size_t)N * K / 2); std::vector<uint16_t> sc((size_t)N * K / 128);
    quantize_int4_g128(W.data(), N, K, pk.data(), sc.data());
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) {
        const float s = f16_bits_to_f32(sc[(size_t)n * (K / 128) + k / 128]);
        REQUIRE(std::fabs(dequant_int4(pk.data(), sc.data(), K, n, k) - W[(size_t)n * K + k]) <= 0.5f * s * 1.0001f + 1e-12f);
        const uint8_t b = pk[(size_t)n * (K / 2) + k / 2]; const int nib = (k & 1) ? (b >> 4) : (b & 0xF); REQUIRE(nib >= 1 && nib <= 15);
    }
    // 6. fp8 round trip within one e4m3 step (relative 2^-4 of the value, or the subnormal step)
    std::vector<uint8_t> q((size_t)N * K); std::vector<float> s8(N); quantize_fp8_rowwise(W.data(), N, K, q.data(), s8.data());
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) { const float w = W[(size_t)n * K + k], d = dequant_fp8(q.data(), s8.data(), K, n, k);
        REQUIRE(std::fabs(d - w) <= std::fabs(w) * 0.0625f + s8[n] * std::ldexp(1.f, -10)); }
    printf("quant_host self-test passed\n"); return 0;
}
