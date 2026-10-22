"""ctypes loader for libpaia_gemv.so (no torch ABI dependency). Launches go on torch's current stream, so the ops are
CUDA-graph capturable. All device pointers are plain ints; cache them per operand copy in Op.build() to keep the
per-call Python cost to one ctypes call."""
import ctypes
import os
import platform
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent


def find_lib():
    env = os.environ.get("PAIA_GEMV_LIB")
    cands = [env] if env else []
    cands += [str(_SRC / "build" / platform.machine() / "libpaia_gemv.so"), str(_SRC / "build" / "libpaia_gemv.so")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("libpaia_gemv.so not found (run src/paia_gemv/build.sh, or set PAIA_GEMV_LIB). Tried: "
                            + ", ".join(c for c in cands if c))


def current_raw_stream():
    import torch
    return torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())


class GemvLib:
    def __init__(self, path=None):
        self.path = path or find_lib()
        lib = ctypes.CDLL(self.path)
        lib.paia_gemv_bf16.restype = ctypes.c_int
        lib.paia_gemv_bf16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.paia_gemv_bf16_variant_count.restype = ctypes.c_int
        lib.paia_gemv_bf16_variant_info.restype = ctypes.c_int
        lib.paia_gemv_bf16_variant_info.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 5
        P = ctypes.c_void_p
        lib.paia_gemv_fp8.restype = ctypes.c_int
        lib.paia_gemv_fp8.argtypes = [P, P, P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int, P]
        lib.paia_gemv_int4.restype = ctypes.c_int
        lib.paia_gemv_int4.argtypes = [P, P, P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int, P]
        lib.paia_gemv_int4_naive.restype = ctypes.c_int
        lib.paia_gemv_int4_naive.argtypes = [P, P, P, P, ctypes.c_int, ctypes.c_int, P]
        lib.paia_gemv_int4_unfused.restype = ctypes.c_int
        lib.paia_gemv_int4_unfused.argtypes = [P, P, P, P, P, ctypes.c_int, ctypes.c_int, P]
        self.variants_fp8 = self._qvariants(lib, "fp8")
        self.variants_int4 = self._qvariants(lib, "int4")
        self._lib = lib
        self._fn = lib.paia_gemv_bf16
        self.n_variants = lib.paia_gemv_bf16_variant_count()
        self.variants = []
        for v in range(self.n_variants):
            R, U, W, S, H = (ctypes.c_int() for _ in range(5))
            assert lib.paia_gemv_bf16_variant_info(v, *map(ctypes.byref, (R, U, W, S, H))) == 0
            self.variants.append({"id": v, "R": R.value, "U": U.value, "W": W.value, "S": S.value, "H": H.value})

    @staticmethod
    def _qvariants(lib, kind):
        count = getattr(lib, f"paia_gemv_{kind}_variant_count")
        info = getattr(lib, f"paia_gemv_{kind}_variant_info")
        info.restype = ctypes.c_int
        info.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 4
        out = []
        for v in range(count()):
            R, U, W, H = (ctypes.c_int() for _ in range(4))
            assert info(v, *map(ctypes.byref, (R, U, W, H))) == 0
            out.append({"id": v, "R": R.value, "U": U.value, "W": W.value, "S": 0, "H": H.value})
        return out

    @staticmethod
    def _check(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} failed with code {rc} (-1 bad variant, -2 bad shape/alignment, >0 cudaError_t)")

    def fp8(self, Wq_ptr, scale_ptr, x_ptr, y_ptr, N, K, variant, stream=None):
        """Wq [N,K] e4m3 bytes, scale [N] fp32, x [K] bf16, y [N] bf16; K % 16 == 0."""
        self._check(self._lib.paia_gemv_fp8(Wq_ptr, scale_ptr, x_ptr, y_ptr, N, K, variant,
                                            current_raw_stream() if stream is None else stream), "paia_gemv_fp8")

    def int4(self, Wq_ptr, scales_ptr, x_ptr, y_ptr, N, K, variant, stream=None):
        """Wq [N,K/2] packed, scales [N,K/128] fp16; K % 128 == 0. Fused kernel (K2)."""
        self._check(self._lib.paia_gemv_int4(Wq_ptr, scales_ptr, x_ptr, y_ptr, N, K, variant,
                                             current_raw_stream() if stream is None else stream), "paia_gemv_int4")

    def int4_naive(self, Wq_ptr, scales_ptr, x_ptr, y_ptr, N, K, stream=None):
        self._check(self._lib.paia_gemv_int4_naive(Wq_ptr, scales_ptr, x_ptr, y_ptr, N, K,
                                                   current_raw_stream() if stream is None else stream), "paia_gemv_int4_naive")

    def int4_unfused(self, Wq_ptr, scales_ptr, x_ptr, y_ptr, scratch_ptr, N, K, stream=None):
        self._check(self._lib.paia_gemv_int4_unfused(Wq_ptr, scales_ptr, x_ptr, y_ptr, scratch_ptr, N, K,
                                                     current_raw_stream() if stream is None else stream), "paia_gemv_int4_unfused")

    def bf16(self, W_ptr, x_ptr, y_ptr, N, K, variant, stream=None):
        """y[N] = W[N,K] @ x[K]; pointers are ints. Raises on a nonzero return code."""
        rc = self._fn(W_ptr, x_ptr, y_ptr, N, K, variant, current_raw_stream() if stream is None else stream)
        if rc != 0:
            raise RuntimeError(f"paia_gemv_bf16 failed with code {rc} (N={N}, K={K}, variant={variant}); "
                               "-1 bad variant, -2 bad shape/alignment (K%8, 16B pointers), >0 cudaError_t")


_lib = None


def load_lib():
    global _lib
    if _lib is None:
        _lib = GemvLib()
    return _lib
