"""S1: our BF16 GEMV (src/paia_gemv) behind the harness Op interface.

Same operands as cublas_bf16.CublasBF16Linear (same RNG seeds, same value range, same fp32 reference and 2e-2 relative
error gate), so the two ops are directly comparable cell by cell. Differences, stated so nobody has to guess:
  * M = 1 only. Other M raise NotImplementedError at construction (S1 scope).
  * The output buffer is preallocated per operand copy; F.linear allocates its output on every call (cheap through the
    caching allocator, but a host-side cost in eager mode). Graph-mode numbers are unaffected; eager numbers include
    one ctypes call (about 1 us) for us and the allocator for cuBLAS.
  * The kernel variant (rows per warp, loads in flight, ...) comes from src/tuned/<platform>.json, frozen per shape
    after a sweep, never autotuned inside call(). Override with variant=<id> or PAIA_GEMV_VARIANT=<id>.
"""
import json
import os
from pathlib import Path

import torch

from bench.ops.base import Op
from bench.ops.cublas_bf16 import REF_ROWS

TUNED_DIR = Path(__file__).resolve().parents[3] / "src" / "tuned"
DEFAULT_VARIANT = {"R": 2, "U": 4, "W": 4, "S": 0, "H": 1}   # safe default before any sweep


def _variant_id(lib, want):
    for v in lib.variants:
        if all(v[k] == want[k] for k in "RUWSH"):
            return v["id"]
    raise KeyError(want)


def tuned_variant(lib, platform_tag, N, K):
    f = TUNED_DIR / f"{platform_tag}.json"
    if f.exists():
        entry = json.load(open(f)).get("shapes", {}).get(f"{N}x{K}")
        if entry:
            return _variant_id(lib, entry["variant"]), "tuned"
    return _variant_id(lib, DEFAULT_VARIANT), "default"


class OursBF16Gemv(Op):
    op, impl, precision = "linear", "ours_bf16_gemv", "bf16"

    def __init__(self, M=None, N=None, K=None, model=None, layer_name=None, variant=None, platform_tag=None):
        if M != 1:
            raise NotImplementedError("ours_bf16_gemv supports M=1 only (S1)")
        if K % 8:
            raise ValueError("K must be a multiple of 8")
        super().__init__(M, N, K, model, layer_name)
        from paia_gemv import load_lib
        self.lib = load_lib()
        if variant is None and os.environ.get("PAIA_GEMV_VARIANT"):
            variant = int(os.environ["PAIA_GEMV_VARIANT"])
        if variant is not None:
            self.variant, self.variant_src = int(variant), "override"
        else:
            from bench.common import default_platform_tag
            self.variant, self.variant_src = tuned_variant(self.lib, platform_tag or default_platform_tag(), N, K)

    def impl_config(self):
        return {"variant": self.lib.variants[self.variant], "variant_source": self.variant_src}

    def footprint_bytes(self):
        return 2 * (self.N * self.K + self.M * self.K + self.M * self.N)

    def bytes(self):
        return {"weights": 2 * self.N * self.K, "metadata": 0, "act_in": 2 * self.M * self.K, "out": 2 * self.M * self.N}

    def build(self, n_copies):
        g = torch.Generator(device="cuda").manual_seed(1000 + self.N + self.K)   # same as CublasBF16Linear
        self._tW = [(torch.rand(self.N, self.K, device="cuda", generator=g, dtype=torch.float32) - 0.5)
                    .mul_(0.04).to(torch.bfloat16) for _ in range(n_copies)]
        self._tx = [(torch.rand(self.M, self.K, device="cuda", generator=g) - 0.5).to(torch.bfloat16)
                    for _ in range(n_copies)]
        self._ty = [torch.empty(self.M, self.N, device="cuda", dtype=torch.bfloat16) for _ in range(n_copies)]
        self._tptrs = [(w.data_ptr(), x.data_ptr(), y.data_ptr()) for w, x, y in zip(self._tW, self._tx, self._ty)]

    def call(self, i):
        w, x, y = self._tptrs[i]
        self.lib.bf16(w, x, y, self.N, self.K, self.variant)
        return self._ty[i]

    def check(self):
        self._ty[0].zero_()
        y = self.call(0).float()
        W, x = self._tW[0], self._tx[0].float()
        ref = torch.cat([x @ W[r:r + REF_ROWS].float().t() for r in range(0, self.N, REF_ROWS)], dim=1)
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        return err < 2e-2, {"rel_err": err, **self.impl_config()}
