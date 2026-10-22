"""CPU emulation of the quantized kernels' arithmetic (no GPU). It is a line-by-line numpy port of the device code in
gemv_fp8.cuh and gemv_int4.cuh: __byte_perm, the half-bit decode of e4m3, the PRMT/magic-number nibble decode, the
even/odd pairing of weights with bf16 halves, lane/chunk/remainder geometry. It checks that
  1. the e4m3 decode is exact for all 254 finite codes, and the nibble decode is exact for 0..15;
  2. y computed with the kernels' dataflow equals the dequantised reference (double) to fp32 accuracy;
  3. every (row, column) is covered exactly once by the loop/remainder/partial-chunk structure.
It cannot test CUDA semantics (that is src/tests/gemv_quant_check.cu on a GPU)."""
import itertools
import numpy as np

rng = np.random.default_rng(0)


# ---------- device intrinsics ----------
def byte_perm(x, y, s):
    """__byte_perm for uint32 arrays: bytes 0..3 from x, 4..7 from y; selector nibbles pick result bytes 0..3."""
    x = np.asarray(x, dtype=np.uint32); y = np.asarray(y, dtype=np.uint32) * np.ones_like(x)
    out = np.zeros_like(x)
    for i in range(4):
        sel = (s >> (4 * i)) & 0x7
        b = ((x >> (8 * sel)) & 0xFF) if sel < 4 else ((y >> (8 * (sel - 4))) & 0xFF)
        out |= b.astype(np.uint32) << (8 * i)
    return out


def bf16_bits(f):  # float32 -> bf16 bits, round to nearest even
    u = np.asarray(f, dtype=np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint32)


def bf16_to_f32(b):
    return (np.asarray(b, dtype=np.uint32) << 16).view(np.float32)


def e4m3_ref():
    t = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        v = np.nan if (e == 15 and m == 7) else (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
        t[b] = -v if s else v
    return t


# ---------- fp8 ----------
def fp8x4_decode(w):
    lo = byte_perm(w, 0, 0x4140); hi = byte_perm(w, 0, 0x4342)
    hl = ((lo << 7) & 0x3F803F80) | ((lo << 8) & 0x80008000)
    hh = ((hi << 7) & 0x3F803F80) | ((hi << 8) & 0x80008000)
    def half2(h):
        h = np.asarray(h, dtype=np.uint32)
        return (h & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32), (h >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    a0, a1 = half2(hl); b0, b1 = half2(hh)
    return a0, a1, b0, b1


def test_fp8_decode():
    ref = e4m3_ref()
    words = np.array([sum(((c + k * 61) % 256) << (8 * k) for k in range(4)) for c in range(256)], dtype=np.uint32)
    for c in range(256):
        w = int(words[c]); bytes_ = [(w >> (8 * k)) & 255 for k in range(4)]
        f = fp8x4_decode(np.array([w], dtype=np.uint32))
        for k in range(4):
            code = bytes_[k]
            if (code & 0x7F) == 0x7F:
                continue                                       # NaN codes are never produced by the quantizer
            assert float(f[k][0]) * 256.0 == ref[code], (code, f[k], ref[code])
    print("fp8 decode: exact for all finite e4m3 codes (256 words x 4 byte positions)")


def fp8_kernel_math(Wq, scale, xb, R=2, U=2):
    """Replays gemv_fp8_kernel for all rows. Wq [N,K] uint8, xb [K] bf16 bits. Returns y float32 [N] and coverage."""
    N, K = Wq.shape
    xw = (xb[0::2] | (xb[1::2] << 16)).astype(np.uint32)       # bf16 pairs as 32-bit words, little endian
    cover = np.zeros((N, K), dtype=np.int32)
    y = np.zeros(N, dtype=np.float32)
    nfull = K >> 9

    def dot16(wrow, col):
        ww = (wrow[col:col + 16].reshape(4, 4).astype(np.uint32) << (8 * np.arange(4, dtype=np.uint32))).sum(1).astype(np.uint32)
        xa = xw[col // 2: col // 2 + 4]; xbv = xw[col // 2 + 4: col // 2 + 8]       # xa.{x,y,z,w}, xb.{x,y,z,w}
        xs = np.concatenate([xa, xbv])                                              # 8 words = 16 bf16
        lo = bf16_to_f32((xs & 0xFFFF)); hi = bf16_to_f32((xs >> 16))
        p0 = np.float32(0); p1 = np.float32(0)
        for j in range(4):                                                          # w.x, w.y, w.z, w.w
            f = fp8x4_decode(ww[j:j + 1])
            p0 = np.float32(p0 + f[0][0] * lo[2 * j]); p1 = np.float32(p1 + f[1][0] * hi[2 * j])
            p0 = np.float32(p0 + f[2][0] * lo[2 * j + 1]); p1 = np.float32(p1 + f[3][0] * hi[2 * j + 1])
        return np.float32(p0 + p1)

    for row in range(N):
        accl = np.zeros(32, dtype=np.float32)
        for lane in range(32):
            c0 = 0
            def run(c, uc):
                nonlocal accl
                for u in range(uc):
                    col = ((c + u) << 9) + lane * 16
                    accl[lane] = np.float32(accl[lane] + dot16(Wq[row], col)); cover[row, col:col + 16] += 1
            while c0 + U <= nfull: run(c0, U); c0 += U
            if U > 2 and nfull - c0 >= 2: run(c0, 2); c0 += 2
            if U > 1 and nfull - c0 >= 1: run(c0, 1); c0 += 1
            if (nfull << 9) < K:
                col = (nfull << 9) + lane * 16
                if col < K: accl[lane] = np.float32(accl[lane] + dot16(Wq[row], col)); cover[row, col:col + 16] += 1
        y[row] = np.float32(accl.sum(dtype=np.float32) * np.float32(scale[row] * np.float32(256.0)))
    return y, cover


def test_fp8_math():
    ref = e4m3_ref()
    for N, K, U in [(3, 512, 1), (5, 1024, 2), (4, 1536, 4), (3, 3584, 2), (2, 2064, 4), (2, 528, 1)]:
        codes = rng.integers(0, 255, size=(N, K), dtype=np.int64); codes[(codes & 0x7F) == 0x7F] = 0x30
        Wq = codes.astype(np.uint8); scale = (rng.random(N) * 0.01 + 0.001).astype(np.float32)
        xf = (rng.random(K) - 0.5).astype(np.float32); xb = bf16_bits(xf); xr = bf16_to_f32(xb).astype(np.float64)
        y, cover = fp8_kernel_math(Wq, scale, xb, U=U)
        yref = (ref[Wq] * scale[:, None].astype(np.float64)) @ xr
        assert (cover == 1).all(), "fp8 coverage"
        rel = np.abs(y - yref).max() / np.abs(yref).max()
        assert rel < 1e-5, (N, K, U, rel)
    print("fp8 kernel dataflow: matches the dequantised reference, every element covered once")


# ---------- int4 ----------
def nib_to_f(m, k):
    return (byte_perm(m, 0x4B000000, 0x7440 | k).view(np.float32) - np.float32(8388616.0))


def test_nibble_decode():
    for n in range(16):
        for k in range(4):
            m = np.array([n << (8 * k)], dtype=np.uint32)
            assert float(nib_to_f(m, k)[0]) == n - 8, (n, k)
    print("int4 nibble decode: exact for all 16 values in all 4 byte positions")


def int4_kernel_math(packed, scales16, xb, R=2, U=2):
    N, half = packed.shape; K = half * 2
    xw = (xb[0::2] | (xb[1::2] << 16)).astype(np.uint32)
    cover = np.zeros((N, K), dtype=np.int32)
    y = np.zeros(N, dtype=np.float32)
    nfull = K >> 10
    scales = scales16.astype(np.float16).astype(np.float32)

    def dot32(wrow, col):
        ww = (wrow[col // 2: col // 2 + 16].reshape(4, 4).astype(np.uint32) << (8 * np.arange(4, dtype=np.uint32))).sum(1).astype(np.uint32)
        xs = xw[col // 2: col // 2 + 16]                                          # 16 words = 4 uint4 = 32 bf16
        p0 = np.float32(0); p1 = np.float32(0)
        for j in range(4):                                                        # word j of w pairs with x uint4 j
            w = ww[j:j + 1]; ev = w & 0x0F0F0F0F; od = (w >> 4) & 0x0F0F0F0F
            xv = xs[4 * j: 4 * j + 4]                                              # .x .y .z .w
            for k in range(4):
                p0 = np.float32(p0 + nib_to_f(ev, k)[0] * bf16_to_f32(xv[k] & 0xFFFF))
                p1 = np.float32(p1 + nib_to_f(od, k)[0] * bf16_to_f32(xv[k] >> 16))
        return np.float32(p0 + p1)

    for row in range(N):
        accl = np.zeros(32, dtype=np.float32)
        for lane in range(32):
            c0 = 0
            def run(c, uc):
                for u in range(uc):
                    col = ((c + u) << 10) + lane * 32
                    accl[lane] = np.float32(scales[row, col >> 7] * dot32(packed[row], col) + accl[lane]); cover[row, col:col + 32] += 1
            while c0 + U <= nfull: run(c0, U); c0 += U
            if U > 2 and nfull - c0 >= 2: run(c0, 2); c0 += 2
            if U > 1 and nfull - c0 >= 1: run(c0, 1); c0 += 1
            if (nfull << 10) < K:
                col = (nfull << 10) + lane * 32
                if col < K:
                    accl[lane] = np.float32(scales[row, col >> 7] * dot32(packed[row], col) + accl[lane]); cover[row, col:col + 32] += 1
        y[row] = accl.sum(dtype=np.float32)
    return y, cover


def test_int4_math():
    for N, K, U in [(3, 1024, 1), (4, 2048, 2), (2, 4096, 4), (3, 3584, 2), (2, 1152, 4), (2, 18944 // 8, 1)]:
        K = (K // 128) * 128
        n = rng.integers(1, 16, size=(N, K), dtype=np.int64)
        packed = (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)
        s16 = (rng.random((N, K // 128)) * 0.01 + 0.001).astype(np.float16)
        xf = (rng.random(K) - 0.5).astype(np.float32); xb = bf16_bits(xf); xr = bf16_to_f32(xb).astype(np.float64)
        y, cover = int4_kernel_math(packed, s16, xb, U=U)
        s = np.repeat(s16.astype(np.float64), 128, axis=1)
        yref = ((n - 8) * s) @ xr
        assert (cover == 1).all(), "int4 coverage"
        rel = np.abs(y - yref).max() / np.abs(yref).max()
        assert rel < 2e-5, (N, K, U, rel)
    print("int4 kernel dataflow: matches the dequantised reference, every element covered once")


if __name__ == "__main__":
    test_fp8_decode(); test_nibble_decode(); test_fp8_math(); test_int4_math()
    print("ALL FORMAT EMULATION CHECKS PASSED")
