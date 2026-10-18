"""E2 (B1): production BF16 GEMV/GEMM via torch.nn.functional.linear (cuBLAS / cuBLASLt).

y[M, N] = x[M, K] @ W[N, K].T. Weights are random in a small range (timing does not depend on values).
The output is allocated per call, exactly as F.linear does in a model.
"""
import torch
import torch.nn.functional as F

from bench.ops.base import Op

REF_ROWS = 8192  # rows of W per chunk for the fp32 reference (bounds transient memory on lm_head)


class CublasBF16Linear(Op):
    op, impl, precision = "linear", "cublas_f_linear", "bf16"

    def footprint_bytes(self):
        return 2 * (self.N * self.K + self.M * self.K)

    def bytes(self):
        return {"weights": 2 * self.N * self.K, "metadata": 0, "act_in": 2 * self.M * self.K, "out": 2 * self.M * self.N}

    def build(self, n_copies):
        g = torch.Generator(device="cuda").manual_seed(1000 + self.N + self.K)
        self._tW = [(torch.rand(self.N, self.K, device="cuda", generator=g, dtype=torch.float32) - 0.5)
                    .mul_(0.04).to(torch.bfloat16) for _ in range(n_copies)]
        self._tx = [(torch.rand(self.M, self.K, device="cuda", generator=g) - 0.5).to(torch.bfloat16)
                    for _ in range(n_copies)]

    def call(self, i):
        return F.linear(self._tx[i], self._tW[i])

    def check(self):
        y = self.call(0).float()
        W, x = self._tW[0], self._tx[0].float()
        ref = torch.cat([x @ W[r:r + REF_ROWS].float().t() for r in range(0, self.N, REF_ROWS)], dim=1)
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        return err < 2e-2, {"rel_err": err}
