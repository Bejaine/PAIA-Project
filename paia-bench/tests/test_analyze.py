"""Synthetic check of the E1 analysis: known tau and R must be recovered, and B0, the hot falloff and
session comparison must behave. GPU-free. Run: python -m tests.test_analyze
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TAU, R_GBPS, L2 = 5.0, 225.0, 24 * 2 ** 20


def make_session(root, sid, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    d = root / "synth" / sid / "cells"
    d.mkdir(parents=True)
    sizes = [2 ** p for p in range(18, 33)]
    n = 0
    for size in sizes:
        for op, impl, mult in (("read", "torch_sum", 1), ("read", "triton_read", 1), ("copy", "torch_copy_", 2),
                               ("write", "torch_fill_", 1)):
            for hc in ("hot", "cold"):
                for mode in ("isolated", "eager", "graph"):
                    B = size * mult
                    r = R_GBPS * (4.0 if hc == "hot" and size <= L2 // 2 else 1.0)
                    t = scale * (TAU + B / (r * 1e3)) * (1 + 0.01 * rng.standard_normal())
                    cid = f"{sid}_{n:04d}"
                    n += 1
                    (d / f"{cid}.json").write_text(json.dumps({
                        "cell_id": f"c{n:04d}", "session_id": sid, "phase": "e1", "op": op, "impl": impl,
                        "precision": "fp32", "model": None, "layer_name": None,
                        "shape": {"M": None, "N": size, "K": None}, "timing_mode": mode, "hot_or_cold": hc,
                        "bytes": {"weights": 0, "metadata": 0, "act_in": B, "out": 0, "total": B},
                        "median_us": t, "achieved_gbps": B / (t * 1e-6) / 1e9, "l2_safe": hc == "cold",
                        "l2_bytes": L2, "status": "clean", "cv": 0.01, "noisy": False,
                        "telemetry": {"sm_clock_mhz": {"min": 2400, "median": 2418, "max": 2420}, "temp_c_max": 60,
                                      "throttle_reasons_seen": []}}))
    return root / "synth" / sid


def main():
    root = Path(tempfile.mkdtemp(prefix="paia_analyze_"))
    s1 = make_session(root, "s1", 1)
    make_session(root, "s2", 2, scale=1.05)  # 5% slower everywhere: comparison must flag
    r = subprocess.run([sys.executable, "-m", "bench.analyze", "--session", "s1", "--platform-tag", "synth",
                        "--results-root", str(root), "--compare", "s2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    fits = json.loads((s1 / "summary" / "fits.json").read_text())
    f = fits["e1"]["fits"]["read/triton_read/graph"]
    assert abs(f["tau_us"] - TAU) / TAU < 0.05, f["tau_us"]
    assert abs(f["R_inf_gbps"] - R_GBPS) / R_GBPS < 0.02, f["R_inf_gbps"]
    assert f["tau_ci95_us"][0] <= TAU <= f["tau_ci95_us"][1], f["tau_ci95_us"]
    assert abs(f["theil_sen"]["R_inf_gbps"] - R_GBPS) / R_GBPS < 0.03, f["theil_sen"]
    b0 = fits["e1"]["R_inf_B0_gbps"]
    assert 0.95 * R_GBPS < b0 <= 1.02 * R_GBPS, b0  # plateau sits just under R_inf because of tau
    hot = fits["e1"]["hot"]["read/triton_read"]
    assert hot["falloff_buffer_bytes"] == 16 * 2 ** 20, hot  # synthetic L2 fast up to 12 MiB, so 16 MiB is the first slow size
    cmp_rows = (s1 / "summary" / "compare.csv").read_text().splitlines()
    assert len(cmp_rows) > 1
    for p in ("summary.md", "cells.csv", "e1_cold_bandwidth.png", "e1_hot_vs_cold.png", "e1_fit_residuals.png", "b0.json"):
        assert (s1 / "summary" / p).exists(), p
    print(f"PASS: tau {f['tau_us']:.3f} us (true {TAU}), R {f['R_inf_gbps']:.2f} GB/s (true {R_GBPS}), "
          f"CI tau {f['tau_ci95_us'][0]:.2f} to {f['tau_ci95_us'][1]:.2f}, B0 {b0:.2f}, hot falloff "
          f"{hot['falloff_buffer_bytes'] / 2**20:g} MiB; outputs in {s1 / 'summary'}")


if __name__ == "__main__":
    main()
