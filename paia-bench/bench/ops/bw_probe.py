"""E1 (B0) bandwidth probes: read / copy / write with torch ops, plus a single-kernel Triton read probe.

Buffers hold random data (not constants) so no data-dependent memory effect can inflate the numbers.
These are measurement tools, not project kernels.
"""
import json

import torch

from bench.ops.base import Op

CHUNK = 1 << 28  # elements per chunk for float64 reference sums


def _rand_f32(n_elems, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(n_elems, device="cuda", generator=g)


def _ref_sum(t):
    return sum(t[i:i + CHUNK].double().sum().item() for i in range(0, t.numel(), CHUNK))


class TorchRead(Op):
    """x.sum() over a float32 buffer. Note: launches a Memset of the output plus a reduce kernel."""
    op, impl, precision = "read", "torch_sum", "fp32"

    def __init__(self, size_bytes):
        super().__init__(N=size_bytes)
        self.size = size_bytes

    def footprint_bytes(self):
        return self.size

    def bytes(self):
        return {"weights": 0, "metadata": 0, "act_in": self.size, "out": 4}

    def build(self, n_copies):
        self._tbufs = [_rand_f32(self.size // 4, 100 + i) for i in range(n_copies)]

    def call(self, i):
        return self._tbufs[i].sum()

    def check(self):
        got, ref = self.call(0).item(), _ref_sum(self._tbufs[0])
        err = abs(got - ref) / abs(ref)
        return err < 1e-3, {"rel_err": err}


class TorchCopy(Op):
    op, impl, precision = "copy", "torch_copy_", "fp32"

    def __init__(self, size_bytes):
        super().__init__(N=size_bytes)
        self.size = size_bytes

    def footprint_bytes(self):
        return 2 * self.size

    def bytes(self):
        return {"weights": 0, "metadata": 0, "act_in": self.size, "out": self.size}

    def build(self, n_copies):
        self._tsrc = [_rand_f32(self.size // 4, 200 + i) for i in range(n_copies)]
        self._tdst = [torch.empty_like(s) for s in self._tsrc]

    def call(self, i):
        return self._tdst[i].copy_(self._tsrc[i])

    def check(self):
        self._tdst[0].zero_()
        self.call(0)
        ok = torch.equal(self._tdst[0], self._tsrc[0])
        return ok, {"equal": ok}


class TorchWrite(Op):
    op, impl, precision = "write", "torch_fill_", "fp32"
    VALUE = 0.7071067811865476

    def __init__(self, size_bytes):
        super().__init__(N=size_bytes)
        self.size = size_bytes

    def footprint_bytes(self):
        return self.size

    def bytes(self):
        return {"weights": 0, "metadata": 0, "act_in": 0, "out": self.size}

    def build(self, n_copies):
        self._tbufs = [_rand_f32(self.size // 4, 300 + i) for i in range(n_copies)]

    def call(self, i):
        return self._tbufs[i].fill_(self.VALUE)

    def check(self):
        self.call(0)
        ok = bool((self._tbufs[0] == torch.tensor(self.VALUE, dtype=torch.float32, device="cuda")).all())
        return ok, {"all_equal": ok}


# ----------------------------------------------------------------------------- Triton read probe

_TRITON = {}


def _triton_kernel():
    if "k" not in _TRITON:
        import triton
        import triton.language as tl

        @triton.jit
        def read_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            nprog = tl.num_programs(0)
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for start in range(pid * BLOCK, n, nprog * BLOCK):
                offs = start + tl.arange(0, BLOCK)
                acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + pid, tl.sum(acc, axis=0))

        _TRITON["k"] = read_kernel
    return _TRITON["k"]


TRITON_CONFIGS = [(b, w, waves) for b in (1024, 2048, 4096) for w in (4, 8) for waves in (2, 4, 8)]


def tune_triton_read(cache_path, sm_count, tune_bytes=1 << 30):
    """Pick (BLOCK, num_warps, waves) once per session on a cold 1 GiB buffer, then persist it so every
    cell (and every resume) uses the same probe configuration."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    k = _triton_kernel()
    bufs = [_rand_f32(tune_bytes // 4, 900 + i) for i in range(2)]
    res = []
    for block, warps, waves in TRITON_CONFIGS:
        grid = sm_count * waves
        out = torch.empty(grid, device="cuda")
        for i in range(3):
            k[(grid,)](bufs[i % 2], out, bufs[0].numel(), BLOCK=block, num_warps=warps)
        torch.cuda.synchronize()
        ts = []
        for i in range(10):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); k[(grid,)](bufs[i % 2], out, bufs[0].numel(), BLOCK=block, num_warps=warps); e.record()
            torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ts.sort()
        res.append({"BLOCK": block, "num_warps": warps, "waves": waves, "median_ms": ts[len(ts) // 2]})
    best = min(res, key=lambda r: r["median_ms"])
    cfg = {**{k2: best[k2] for k2 in ("BLOCK", "num_warps", "waves")}, "sm_count": sm_count, "tuning": res,
           "note": "tuned once per session on a cold 1 GiB buffer; timings here are for selection only"}
    del bufs
    torch.cuda.empty_cache()
    cache_path.write_text(json.dumps(cfg, indent=1))
    return cfg


class TritonRead(Op):
    """Single-kernel grid-stride vectorized read; writes one float per program."""
    op, impl, precision = "read", "triton_read", "fp32"

    def __init__(self, size_bytes, tcfg):
        super().__init__(N=size_bytes)
        self.size = size_bytes
        self.tcfg = tcfg
        self.grid = tcfg["sm_count"] * tcfg["waves"]

    def config(self):
        c = super().config()
        c["triton_cfg"] = {k: self.tcfg[k] for k in ("BLOCK", "num_warps", "waves")}
        return c

    def footprint_bytes(self):
        return self.size

    def bytes(self):
        return {"weights": 0, "metadata": 0, "act_in": self.size, "out": 4 * self.grid}

    def build(self, n_copies):
        self._k = _triton_kernel()
        self._tbufs = [_rand_f32(self.size // 4, 400 + i) for i in range(n_copies)]
        self._tout = torch.empty(self.grid, device="cuda")

    def call(self, i):
        b = self._tbufs[i]
        self._k[(self.grid,)](b, self._tout, b.numel(), BLOCK=self.tcfg["BLOCK"], num_warps=self.tcfg["num_warps"])
        return self._tout

    def check(self):
        got, ref = self.call(0).double().sum().item(), _ref_sum(self._tbufs[0])
        err = abs(got - ref) / abs(ref)
        return err < 1e-3, {"rel_err": err}
