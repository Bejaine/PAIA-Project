"""B3 and INT4 kernels (src/paia_gemv) behind the harness Op interface: FP8 W8A16 and INT4 W4A16 g128.

Same conventions as ours_bf16.OursBF16Gemv (M = 1 only, preallocated outputs, variant frozen per shape from
src/tuned/<platform>.json or overridden by variant=<id> / PAIA_GEMV_VARIANT, never autotuned inside call()).
Operands: copy 0 is real quantized data from a random float matrix (same generator as the BF16 op), and its dequantized
fp32 value is the correctness reference (gate: max|y-ref|/max|ref| < 2e-2, as in the harness). Copies 1.. are random bytes
(valid for timing: the decode has no data-dependent branches) so that rotation over distinct memory defeats the L2.
bytes() includes the scale metadata, so achieved GB/s and the ideal speedup are metadata-corrected."""
import json
import os
from pathlib import Path

import torch

from bench.ops.base import Op
from bench.ops.cublas_bf16 import REF_ROWS
from bench.ops.ours_bf16 import TUNED_DIR

DEFAULTS = {"fp8": {"R": 2, "U": 2, "W": 4, "S": 0, "H": 1}, "int4": {"R": 4, "U": 2, "W": 4, "S": 0, "H": 1}}


def _pick(variants, want):
    for v in variants:
        if all(v[k] == want[k] for k in "RUWSH"):
            return v["id"]
    raise KeyError(want)


def _tuned(lib, kind, platform_tag, N, K):
    variants = lib.variants_fp8 if kind == "fp8" else lib.variants_int4
    f = TUNED_DIR / f"{platform_tag}.json"
    if f.exists():
        e = json.load(open(f)).get(f"{kind}_shapes", {}).get(f"{N}x{K}")
        if e:
            return _pick(variants, e["variant"]), "tuned"
    return _pick(variants, DEFAULTS[kind]), "default"


class _OursQuant(Op):
    kind = None   # "fp8" | "int4"

    def __init__(self, M=None, N=None, K=None, model=None, layer_name=None, variant=None, platform_tag=None):
        if M != 1:
            raise NotImplementedError("quantized ours_* ops support M=1 only")
        super().__init__(M, N, K, model, layer_name)
        from paia_gemv import load_lib
        self.lib = load_lib()
        if variant is None and os.environ.get("PAIA_GEMV_VARIANT"):
            variant = int(os.environ["PAIA_GEMV_VARIANT"])
        if variant is not None:
            self.variant, self.variant_src = int(variant), "override"
        else:
            from bench.common import default_platform_tag
            self.variant, self.variant_src = _tuned(self.lib, self.kind, platform_tag or default_platform_tag(), N, K)

    def _variants(self):
        return self.lib.variants_fp8 if self.kind == "fp8" else self.lib.variants_int4

    def impl_config(self):
        return {"variant": self._variants()[self.variant], "variant_source": self.variant_src}

    def _alloc_common(self, n_copies):
        g = torch.Generator(device="cuda").manual_seed(1000 + self.N + self.K)    # same seeds as the BF16 ops
        W0 = (torch.rand(self.N, self.K, device="cuda", generator=g, dtype=torch.float32) - 0.5).mul_(0.04)
        self._tx = [(torch.rand(self.M, self.K, device="cuda", generator=g) - 0.5).to(torch.bfloat16) for _ in range(n_copies)]
        self._ty = [torch.empty(self.M, self.N, device="cuda", dtype=torch.bfloat16) for _ in range(n_copies)]
        return W0

    def _reference(self, Wdeq):
        x = self._tx[0].float()
        return torch.cat([x @ Wdeq[r:r + REF_ROWS].t() for r in range(0, self.N, REF_ROWS)], dim=1)

    def check(self):
        self._ty[0].zero_()
        y = self.call(0).float()
        ref = self._reference(self._ref_deq)
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        return err < 2e-2, {"rel_err": err, **self.impl_config()}


class OursFP8Gemv(_OursQuant):
    op, impl, precision, kind = "linear", "ours_fp8_w8a16", "fp8_w8a16", "fp8"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if self.K % 16:
            raise ValueError("K % 16 required")

    def footprint_bytes(self):
        return self.N * self.K + 4 * self.N + 2 * self.K + 2 * self.N

    def bytes(self):
        return {"weights": self.N * self.K, "metadata": 4 * self.N, "act_in": 2 * self.M * self.K, "out": 2 * self.M * self.N}

    def build(self, n_copies):
        from paia_gemv.quant import quantize_fp8_rowwise, dequant_fp8
        W0 = self._alloc_common(n_copies)
        q0, s0 = quantize_fp8_rowwise(W0)
        self._ref_deq = dequant_fp8(q0, s0)
        del W0
        self._tq = [q0] + [torch.randint(0, 255, (self.N, self.K), device="cuda", dtype=torch.uint8) for _ in range(n_copies - 1)]
        self._ts = [s0] + [torch.rand(self.N, device="cuda", dtype=torch.float32).mul_(0.01).add_(1e-3) for _ in range(n_copies - 1)]
        self._tptrs = [(q.data_ptr(), s.data_ptr(), x.data_ptr(), y.data_ptr()) for q, s, x, y in zip(self._tq, self._ts, self._tx, self._ty)]

    def call(self, i):
        q, s, x, y = self._tptrs[i]
        self.lib.fp8(q, s, x, y, self.N, self.K, self.variant)
        return self._ty[i]


class OursInt4Gemv(_OursQuant):
    """mode: 'fused' (K2), 'naive' (K1 control), 'unfused' (dequantize to bf16 then S1's GEMV; sanity control)."""
    op, precision, kind = "linear", "int4_g128_w4a16", "int4"

    def __init__(self, M=None, N=None, K=None, model=None, layer_name=None, variant=None, platform_tag=None, mode="fused"):
        if mode not in ("fused", "naive", "unfused"):
            raise ValueError(mode)
        if K % 128:
            raise ValueError("K % 128 required")
        self.mode = mode
        self.impl = f"ours_int4_{mode}"
        super().__init__(M, N, K, model, layer_name, variant, platform_tag)

    def footprint_bytes(self):
        base = self.N * self.K // 2 + 2 * self.N * (self.K // 128) + 2 * self.K + 2 * self.N
        return base + (2 * self.N * self.K if self.mode == "unfused" else 0)

    def bytes(self):
        # Useful bytes of the quantized format. The unfused control additionally writes and re-reads 2*N*K of scratch; that
        # traffic is not counted here on purpose, so its GB/s shows how much slower the control is.
        return {"weights": self.N * self.K // 2, "metadata": 2 * self.N * (self.K // 128),
                "act_in": 2 * self.M * self.K, "out": 2 * self.M * self.N}

    def build(self, n_copies):
        from paia_gemv.quant import quantize_int4_g128, dequant_int4
        W0 = self._alloc_common(n_copies)
        p0, s0 = quantize_int4_g128(W0)
        self._ref_deq = dequant_int4(p0, s0)
        del W0
        G = self.K // 128
        self._tq = [p0] + [torch.randint(0, 255, (self.N, self.K // 2), device="cuda", dtype=torch.uint8) for _ in range(n_copies - 1)]
        self._ts = [s0] + [(torch.rand(self.N, G, device="cuda").mul_(0.01).add_(1e-3)).to(torch.float16) for _ in range(n_copies - 1)]
        self._scratch = torch.empty(self.N, self.K, device="cuda", dtype=torch.bfloat16) if self.mode == "unfused" else None
        self._tptrs = [(q.data_ptr(), s.data_ptr(), x.data_ptr(), y.data_ptr()) for q, s, x, y in zip(self._tq, self._ts, self._tx, self._ty)]

    def call(self, i):
        q, s, x, y = self._tptrs[i]
        if self.mode == "fused":
            self.lib.int4(q, s, x, y, self.N, self.K, self.variant)
        elif self.mode == "naive":
            self.lib.int4_naive(q, s, x, y, self.N, self.K)
        else:
            self.lib.int4_unfused(q, s, x, y, self._scratch.data_ptr(), self.N, self.K)
        return self._ty[i]

    def impl_config(self):
        cfg = super().impl_config() if self.mode == "fused" else {"variant": None, "variant_source": self.mode}
        return {**cfg, "mode": self.mode}
