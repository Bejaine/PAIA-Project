"""E4: library FP8 and INT4 linear ops behind the harness Op interface.

y[M, N] = x[M, K] @ W[N, K].T with W stored quantized. Every op is the production call a serving stack makes:
  torch_scaled_mm_fp8_pt   torch._scaled_mm, e4m3 weights and activations, per-tensor fp32 scales (cuBLASLt)
  vllm_cutlass_fp8_rw      vLLM cutlass_scaled_mm, e4m3, per-row weight scales, per-token activation scales
                           (torch's row-wise _scaled_mm returns garbage on sm_121, see findings.md)
  *_full                   the same GEMM preceded by vLLM's dynamic activation quantization (scaled_fp8_quant),
                           i.e. what a W8A8 layer costs per call when x arrives in bf16
  tinygemm_int4_g128       torch _weight_int4pack_mm, uint4 with bf16 scale and zero per group of 128 (4.25 bits)
  marlin_int4_g128         vLLM gptq_marlin_gemm (apply_gptq_marlin_linear), uint4b8, bf16 scale per group of 128,
                           no zero point (4.125 bits), weights repacked on the GPU by gptq_marlin_repack as vLLM does

Operands: copy 0 is real quantized data from the same random float matrix as the BF16 ops (seed 1000 + N + K), and its
dequantized fp32 value is the correctness reference (gate: max|y-ref|/max|ref| < 2e-2, as for every op). Copies 1.. hold
random codes in the same layout (valid for timing: these kernels have no data-dependent control flow; FP8 codes avoid
NaN), so rotating over distinct memory defeats the L2 without quantizing a large matrix n_copies times.
bytes() counts the stored format, metadata included, so achieved GB/s and the theoretical speedup are metadata-corrected.
For *_full ops act_in is the bf16 x read once; the fp8 copy of x and its scale (M*K + 4M bytes, written then read) are
intermediate traffic and are not counted, so the per-call cost of activation quantization shows up as time.
Outputs are allocated per call, as in a model.
"""
import torch

from bench.ops.base import Op
from bench.ops.cublas_bf16 import REF_ROWS

FP8_MAX = 448.0
G = 128   # quantization group size along K (INT4)


def _float_weights(N, K, M):
    """Same generator, seed and value range as CublasBF16Linear, so all E2/E4 ops see the same W and x."""
    g = torch.Generator(device="cuda").manual_seed(1000 + N + K)
    W = (torch.rand(N, K, device="cuda", generator=g, dtype=torch.float32) - 0.5).mul_(0.04)
    return W, g


def _random_fp8(N, K):
    """Random e4m3fn codes without NaN (0x7F / 0xFF): bytes with all 7 magnitude bits set are cleared to 0x7E."""
    b = torch.randint(0, 256, (N, K), device="cuda", dtype=torch.uint8)
    b[(b & 0x7F) == 0x7F] = 0x7E
    return b.view(torch.float8_e4m3fn)


def _reference(x, Wdeq_fn, N):
    """fp32 x @ Wdeq.T in row chunks; Wdeq_fn(r0, r1) returns the dequantized fp32 rows [r0, r1)."""
    return torch.cat([x @ Wdeq_fn(r, min(r + REF_ROWS, N)).t() for r in range(0, N, REF_ROWS)], dim=1)


def _rel_err(y, ref):
    return ((y.float() - ref).abs().max() / ref.abs().max()).item()


# ----------------------------------------------------------------------------- FP8

class _FP8(Op):
    rowwise = False
    full = False

    @staticmethod
    def supports(N, K):
        return N % 16 == 0 and K % 16 == 0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if not self.supports(self.N, self.K):
            raise ValueError(f"{self.impl}: unsupported shape N={self.N} K={self.K}")

    def footprint_bytes(self):
        return self.N * self.K + 4 * self.N + 2 * self.M * self.K + self.M * self.K + 4 * self.M + 8

    def bytes(self):
        w_meta = 4 * self.N if self.rowwise else 4
        if self.full:
            return {"weights": self.N * self.K, "metadata": w_meta, "act_in": 2 * self.M * self.K, "out": 2 * self.M * self.N}
        x_meta = 4 * self.M if self.rowwise else 4
        return {"weights": self.N * self.K, "metadata": w_meta + x_meta, "act_in": self.M * self.K, "out": 2 * self.M * self.N}

    def _quant_x(self, x):
        """Activation quantization as the reference sees it (and as the GEMM-only op stores x)."""
        if self.rowwise:
            sx = (x.float().abs().amax(1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
        else:
            sx = (x.float().abs().amax() / FP8_MAX).clamp(min=1e-12)
        return (x.float() / sx).to(torch.float8_e4m3fn), sx.float()

    def build(self, n_copies):
        W, g = _float_weights(self.N, self.K, self.M)
        if self.rowwise:
            sw = (W.abs().amax(1, keepdim=True) / FP8_MAX).clamp(min=1e-12)      # [N, 1]
        else:
            sw = (W.abs().amax() / FP8_MAX).clamp(min=1e-12).reshape(())          # scalar
        Wq0 = (W / sw).to(torch.float8_e4m3fn)
        del W
        self._tWq = [Wq0] + [_random_fp8(self.N, self.K) for _ in range(n_copies - 1)]
        self._tsw = [sw.clone() for _ in range(n_copies)]
        self._tx = [(torch.rand(self.M, self.K, device="cuda", generator=g) - 0.5).to(torch.bfloat16)
                    for _ in range(n_copies)]
        if not self.full:
            q = [self._quant_x(x) for x in self._tx]
            self._txq = [a for a, _ in q]
            self._tsx = [b for _, b in q]

    def _gemm(self, xq, sx, Wq, sw):
        raise NotImplementedError

    def _dyn_quant(self, x):
        from vllm import _custom_ops as ops
        return ops.scaled_fp8_quant(x, use_per_token_if_dynamic=self.rowwise)

    def call(self, i):
        if self.full:
            xq, sx = self._dyn_quant(self._tx[i])
            return self._gemm(xq, sx, self._tWq[i], self._tsw[i])
        return self._gemm(self._txq[i], self._tsx[i], self._tWq[i], self._tsw[i])

    def check(self):
        y = self.call(0)
        if self.full:
            xq, sx = self._dyn_quant(self._tx[0])   # the op's own quantized x is the reference input
        else:
            xq, sx = self._txq[0], self._tsx[0]
        x = xq.float() * sx
        Wq, sw = self._tWq[0], self._tsw[0]
        if self.rowwise:
            deq = lambda r0, r1: Wq[r0:r1].float() * sw[r0:r1]
        else:
            deq = lambda r0, r1: Wq[r0:r1].float() * sw
        err = _rel_err(y, _reference(x, deq, self.N))
        return err < 2e-2, {"rel_err": err}


class ScaledMMFP8PerTensor(_FP8):
    op, impl, precision = "linear", "torch_scaled_mm_fp8_pt", "fp8_e4m3_w8a8_pt"

    def _gemm(self, xq, sx, Wq, sw):
        return torch._scaled_mm(xq, Wq.t(), scale_a=sx.reshape(()), scale_b=sw, out_dtype=torch.bfloat16)


class ScaledMMFP8PerTensorFull(ScaledMMFP8PerTensor):
    impl, full = "torch_scaled_mm_fp8_pt_full", True


class CutlassFP8RowWise(_FP8):
    op, impl, precision = "linear", "vllm_cutlass_fp8_rw", "fp8_e4m3_w8a8_rw"
    rowwise = True

    def _gemm(self, xq, sx, Wq, sw):
        from vllm import _custom_ops as ops
        return ops.cutlass_scaled_mm(xq, Wq.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)


class CutlassFP8RowWiseFull(CutlassFP8RowWise):
    impl, full = "vllm_cutlass_fp8_rw_full", True


# ----------------------------------------------------------------------------- INT4

def _quant_int4_asym(W):
    """Per group of G along K: q in 0..15, w ~ (q - 8) * scale + zero (tinygemm's convention)."""
    N, K = W.shape
    Wg = W.view(N, K // G, G)
    mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    scale = (mx - mn).clamp(min=1e-6) / 15.0
    q = ((Wg - mn) / scale).round().clamp(0, 15).to(torch.uint8).view(N, K)
    zero = mn + 8 * scale
    return q, scale.squeeze(-1), zero.squeeze(-1)          # [N, K], [N, K/G], [N, K/G]


def _quant_int4_sym(W):
    """Per group of G along K: q = clamp(rint(w / scale), -8, 7) + 8, scale = amax / 7, w ~ (q - 8) * scale (uint4b8)."""
    N, K = W.shape
    Wg = W.view(N, K // G, G)
    scale = (Wg.abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-8)
    q = ((Wg / scale).round().clamp(-8, 7) + 8).to(torch.uint8).view(N, K)
    return q, scale.squeeze(-1)                              # [N, K], [N, K/G]


class TinygemmInt4(Op):
    op, impl, precision = "linear", "tinygemm_int4_g128", "int4_g128_asym_w4a16"
    INNER_K_TILES = 8

    @staticmethod
    def supports(N, K):
        return N % 8 == 0 and K % (TinygemmInt4.INNER_K_TILES * 16) == 0 and K % G == 0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if not self.supports(self.N, self.K):
            raise ValueError(f"{self.impl}: unsupported shape N={self.N} K={self.K}")

    def footprint_bytes(self):
        return self.N * self.K // 2 + 4 * self.N * (self.K // G) + 2 * self.M * self.K

    def bytes(self):
        return {"weights": self.N * self.K // 2, "metadata": 4 * self.N * (self.K // G),
                "act_in": 2 * self.M * self.K, "out": 2 * self.M * self.N}

    def _pack(self, q):
        """uint8 [N, K/2], even k in the high nibble (the layout _convert_weight_to_int4pack expects; verified in
        selfcheck), then tinygemm's tiled int32 layout."""
        return torch.ops.aten._convert_weight_to_int4pack((q[:, ::2] << 4 | q[:, 1::2]).contiguous(), self.INNER_K_TILES)

    def build(self, n_copies):
        W, g = _float_weights(self.N, self.K, self.M)
        q, scale, zero = _quant_int4_asym(W)
        del W
        self._ref = (q, scale, zero)
        sz0 = torch.stack([scale, zero], -1).transpose(0, 1).contiguous().to(torch.bfloat16)   # [K/G, N, 2]
        self._tpk = [self._pack(q)]
        self._tsz = [sz0]
        for _ in range(n_copies - 1):
            qr = torch.randint(0, 16, (self.N, self.K), device="cuda", dtype=torch.uint8)
            self._tpk.append(self._pack(qr))
            del qr
            self._tsz.append(sz0.clone())
        self._tx = [(torch.rand(self.M, self.K, device="cuda", generator=g) - 0.5).to(torch.bfloat16)
                    for _ in range(n_copies)]

    def call(self, i):
        return torch.ops.aten._weight_int4pack_mm(self._tx[i], self._tpk[i], G, self._tsz[i])

    def check(self):
        y = self.call(0)
        q, scale, zero = self._ref
        # the reference uses the bf16 scale and zero the kernel reads
        s, z = scale.to(torch.bfloat16).float(), zero.to(torch.bfloat16).float()

        def deq(r0, r1):
            qq = q[r0:r1].view(r1 - r0, self.K // G, G).float()
            return ((qq - 8) * s[r0:r1, :, None] + z[r0:r1, :, None]).view(r1 - r0, self.K)
        err = _rel_err(y, _reference(self._tx[0].float(), deq, self.N))
        return err < 2e-2, {"rel_err": err}

    def free(self):
        self._ref = None
        super().free()


class MarlinInt4(Op):
    op, impl, precision = "linear", "marlin_int4_g128", "int4_g128_sym_w4a16"

    @staticmethod
    def supports(N, K):
        return N % 64 == 0 and K % 128 == 0 and K % G == 0     # GPTQ_MARLIN_MIN_THREAD_N / _K in vLLM 0.11

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if not self.supports(self.N, self.K):
            raise ValueError(f"{self.impl}: unsupported shape N={self.N} K={self.K}")

    def footprint_bytes(self):
        return self.N * self.K // 2 + 2 * self.N * (self.K // G) + 2 * self.M * self.K

    def bytes(self):
        return {"weights": self.N * self.K // 2, "metadata": 2 * self.N * (self.K // G),
                "act_in": 2 * self.M * self.K, "out": 2 * self.M * self.N}

    def _marlin(self, q, scale):
        """q uint8 [N, K] (0..15), scale [N, K/G] -> (marlin weight int32, marlin scales bf16), via the GPTQ layout
        (qweight [K/8, N] int32, nibble j of word i holds row k = 8i + j) and vLLM's on-GPU repack, as vLLM loads GPTQ."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
        qt = q.t().contiguous().view(self.K // 8, 8, self.N).to(torch.int32)
        packed = torch.zeros(self.K // 8, self.N, device="cuda", dtype=torch.int32)
        for j in range(8):
            packed |= qt[:, j, :] << (4 * j)
        del qt
        perm = torch.empty(0, dtype=torch.int32, device="cuda")
        mq = ops.gptq_marlin_repack(packed, perm, self.K, self.N, 4)
        ms = marlin_permute_scales(scale.t().contiguous().to(torch.bfloat16), self.K, self.N, G)
        return mq, ms

    def build(self, n_copies):
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
        W, g = _float_weights(self.N, self.K, self.M)
        q, scale = _quant_int4_sym(W)
        del W
        self._ref = (q, scale)
        mq0, ms0 = self._marlin(q, scale)
        self._tq = [mq0] + [torch.randint(-2**31, 2**31 - 1, mq0.shape, device="cuda", dtype=torch.int32)
                            for _ in range(n_copies - 1)]
        self._ts = [ms0] + [ms0.clone() for _ in range(n_copies - 1)]
        self._tx = [(torch.rand(self.M, self.K, device="cuda", generator=g) - 0.5).to(torch.bfloat16)
                    for _ in range(n_copies)]
        dev = torch.device("cuda")
        self._tws = mu.marlin_make_workspace_new(dev)
        self._tg = mu.marlin_make_empty_g_idx(dev)
        self._mu = mu

    def call(self, i):
        from vllm.scalar_type import scalar_types
        return self._mu.apply_gptq_marlin_linear(self._tx[i], self._tq[i], self._ts[i], self._tg, self._tg, self._tg,
                                                 self._tws, scalar_types.uint4b8, self.N, self.K, is_k_full=True)

    def check(self):
        y = self.call(0)
        q, scale = self._ref
        s = scale.to(torch.bfloat16).float()

        def deq(r0, r1):
            qq = q[r0:r1].view(r1 - r0, self.K // G, G).float()
            return ((qq - 8) * s[r0:r1, :, None]).view(r1 - r0, self.K)
        err = _rel_err(y, _reference(self._tx[0].float(), deq, self.N))
        return err < 2e-2, {"rel_err": err}

    def free(self):
        self._ref = None
        self._mu = None
        super().free()


E4_OPS = {c.impl: c for c in (ScaledMMFP8PerTensor, ScaledMMFP8PerTensorFull, CutlassFP8RowWise, CutlassFP8RowWiseFull,
                              TinygemmInt4, MarlinInt4)}
