"""Timing modes, L2 rotation and statistics. CUDA events only; wall clock is used only to bound warmup.

isolated : per sample: sync, record start, 1 op, record end, sync
eager    : events around a Python loop of R back-to-back ops; per-op = elapsed / R
graph    : R back-to-back ops captured in one CUDA graph; events around one replay; per-op = elapsed / R
"""
import math
import statistics
import time

import torch


def l2_bytes():
    return torch.cuda.get_device_properties(0).L2_cache_size


def plan_rotation(footprint, hot, mem_cap_bytes, l2, factor):
    """n_copies = max(2, ceil(factor * L2 / footprint)), capped by memory. Cold safety is judged by reuse
    distance: bytes touched between two uses of the same copy must be >= factor * L2."""
    if hot:
        return {"n_copies": 1, "rotation_bytes": footprint, "l2_bytes": l2, "l2_safe": None,
                "l2_coverage": footprint / l2}
    want = max(2, math.ceil(factor * l2 / footprint))
    fit = max(1, mem_cap_bytes // footprint)
    n = min(want, fit)
    rot = n * footprint
    return {"n_copies": n, "rotation_bytes": rot, "l2_bytes": l2, "l2_safe": rot >= factor * l2,
            "l2_coverage": rot / l2, "n_copies_wanted": want}


R_MAX = 20000


def _events(n):
    return [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]


class Runner:
    """Builds the per-mode unit of work for an op: one 'unit' is what one sample times."""

    def __init__(self, op, n_copies, mode, R=1):
        self.op, self.n, self.mode, self.R = op, n_copies, mode, R
        self.i = 0
        self.graph = None
        if mode == "graph":
            self._capture()

    def _next(self):
        i = self.i
        self.i = (self.i + 1) % self.n
        return i

    def _capture(self):
        op, n = self.op, self.n
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for j in range(min(3, self.R)):
                op.call(j % n)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for j in range(self.R):
                op.call(j % n)
        self.graph = g

    def unit(self):
        """Launch one unit without timing (warmup)."""
        if self.mode == "graph":
            self.graph.replay()
        elif self.mode == "eager":
            for _ in range(self.R):
                self.op.call(self._next())
        else:
            self.op.call(self._next())

    def timed(self, n_samples):
        """Per-op microseconds for n_samples units."""
        ev = _events(n_samples)
        if self.mode == "isolated":
            for s, e in ev:
                i = self._next()
                torch.cuda.synchronize()
                s.record()
                self.op.call(i)
                e.record()
                e.synchronize()
        elif self.mode == "eager":
            torch.cuda.synchronize()
            for s, e in ev:
                s.record()
                for _ in range(self.R):
                    self.op.call(self._next())
                e.record()
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()
            for s, e in ev:
                s.record()
                self.graph.replay()
                e.record()
            torch.cuda.synchronize()
        return [s.elapsed_time(e) * 1e3 / (1 if self.mode == "isolated" else self.R) for s, e in ev]

    def release(self):
        self.graph = None


def warmup(runner, min_calls, min_ms):
    """Run units until at least min_calls ops and min_ms of wall time have passed."""
    per_unit = 1 if runner.mode == "isolated" else runner.R
    calls, t0 = 0, time.perf_counter()
    while calls < min_calls or (time.perf_counter() - t0) * 1e3 < min_ms:
        runner.unit()
        calls += per_unit
        if runner.mode == "isolated" or calls % (per_unit * 8) == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return {"warmup_calls": calls, "warmup_ms": (time.perf_counter() - t0) * 1e3}


def estimate_op_us(op, n_copies, n=10):
    r = Runner(op, n_copies, "isolated")
    for _ in range(3):
        r.unit()
    return statistics.median(r.timed(n))


def choose_R(op_us, target_ms):
    return max(1, math.ceil(target_ms * 1e3 / max(op_us, 1e-3)))


def stats(samples, noisy_cv):
    s = sorted(samples)
    n = len(s)
    q = lambda p: s[min(n - 1, max(0, round(p * (n - 1))))]
    mean = statistics.fmean(s)
    std = statistics.pstdev(s) if n > 1 else 0.0
    cv = std / mean if mean > 0 else None
    return {"median_us": statistics.median(s), "p5_us": q(0.05), "p95_us": q(0.95), "mean_us": mean,
            "std_us": std, "cv": cv, "noisy": bool(cv is not None and cv > noisy_cv), "n_samples": n}


def measure(op, mode, n_copies, tcfg):
    """Full measurement of one cell's op in one mode. Returns samples, stats and parameters."""
    est = estimate_op_us(op, n_copies)
    R = 1 if mode == "isolated" else choose_R(est, tcfg["loop_target_ms"])
    runner = Runner(op, n_copies, mode, R)
    try:
        if mode != "isolated":
            # The isolated estimate includes launch/sync overhead, so back-to-back ops are faster and R is
            # too small. Re-size R from a measured unit until one unit lasts at least loop_target_ms.
            for _ in range(3):
                runner.unit(); torch.cuda.synchronize()
                unit_ms = statistics.median(runner.timed(3)) * R / 1e3
                if unit_ms >= 0.9 * tcfg["loop_target_ms"] or R >= R_MAX:
                    break
                runner.release()
                R = min(R_MAX, math.ceil(R * tcfg["loop_target_ms"] / max(unit_ms, 1e-4)))
                runner = Runner(op, n_copies, mode, R)
        w = warmup(runner, tcfg["warmup_min_calls"], tcfg["warmup_min_ms"])
        unit_ms = statistics.median(runner.timed(3)) * R / 1e3
        n = int(min(tcfg["samples_max"], max(tcfg["samples_min"], math.ceil(tcfg["sample_target_ms"] / max(unit_ms, 1e-3)))))
        samples = runner.timed(n)
    finally:
        runner.release()
    return {"samples_us": samples, "R_per_iter": R, "est_op_us": est, **w,
            **stats(samples, tcfg["noisy_cv"])}


def kernel_names(op, n_copies):
    from torch.profiler import ProfilerActivity, profile
    op.call(0)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(2):
            op.call(i % n_copies)
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
