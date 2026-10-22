"""GPU correctness tests for the S1 BF16 GEMV. Run in the paia-bench container (or any torch+CUDA env) after build.sh:
    python src/tests/test_gemv_bf16.py            # quick: every variant on a few awkward shapes, default variant on all real shapes
    python src/tests/test_gemv_bf16.py --full     # every variant on every shape (slow)
Reference: fp32 matmul of the same bf16 operands. Gate: max|y-ref| / max|ref| < 2e-2 (the harness's gate); the observed
error should be near bf16 rounding of the output (about 4e-3), and we also print it so drift is visible.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paia_gemv import load_lib  # noqa: E402

AWKWARD = [(1, 8), (1, 256), (3, 264), (7, 520), (33, 1032), (129, 4104), (1000, 3584), (4097, 4096), (513, 14336)]
REAL = [(4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336), (3584, 3584), (512, 3584), (18944, 3584),
        (3584, 18944), (6144, 4096), (4608, 3584)]
BIG = [(128256, 4096), (152064, 3584), (28672, 4096), (37888, 3584)]


def run_one(lib, v, N, K, seed, scale=0.04, special=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = ((torch.rand(N, K, device="cuda", generator=g) - 0.5) * scale).to(torch.bfloat16)
    x = (torch.rand(1, K, device="cuda", generator=g) - 0.5).to(torch.bfloat16)
    if special == "zeros":
        W.zero_()
    elif special == "large":
        W = (W.float() * 1e4).clamp(-3e4, 3e4).to(torch.bfloat16); x = (x.float() * 1e2).to(torch.bfloat16)
    y = torch.full((1, N), float("nan"), device="cuda", dtype=torch.bfloat16)     # NaN canary: every row must be written
    lib.bf16(W.data_ptr(), x.data_ptr(), y.data_ptr(), N, K, v)
    torch.cuda.synchronize()
    ref = x.float() @ W.float().t()
    if torch.isnan(y).any():
        return float("inf")
    denom = ref.abs().max().item()
    return ((y.float() - ref).abs().max().item() / denom) if denom > 0 else y.float().abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    lib = load_lib()
    print(f"{lib.path}: {lib.n_variants} variants, device {torch.cuda.get_device_name(0)} cc {torch.cuda.get_device_capability()}")
    bad, worst = [], 0.0
    shapes_all = AWKWARD + REAL + (BIG if a.full else [])
    for v in lib.variants:
        shapes = shapes_all if a.full else AWKWARD
        for (N, K) in shapes:
            e = run_one(lib, v["id"], N, K, seed=N * 7 + K)
            worst = max(worst, e if e != float("inf") else 0)
            if not e < 2e-2:
                bad.append((v, N, K, e))
        for special in ("zeros", "large"):
            e = run_one(lib, v["id"], 1000, 3584, 5, special=special)
            if not e < 2e-2:
                bad.append((v, 1000, 3584, f"{special}:{e}"))
    if not a.full:   # default variant on all real shapes, plus the big ones
        dv = next(v["id"] for v in lib.variants if (v["R"], v["U"], v["W"], v["S"], v["H"]) == (2, 4, 4, 0, 1))
        for (N, K) in REAL + BIG:
            e = run_one(lib, dv, N, K, seed=N + K)
            worst = max(worst, e)
            if not e < 2e-2:
                bad.append((dv, N, K, e))
    # contract checks
    W = torch.zeros(4, 12, device="cuda", dtype=torch.bfloat16)
    try:
        lib.bf16(W.data_ptr(), W.data_ptr(), W.data_ptr(), 4, 12, 0); bad.append(("K%8 check missing",))
    except RuntimeError:
        pass
    print(f"worst relative error over passing cases: {worst:.2e}")
    if bad:
        print("FAILURES:", *bad[:20], sep="\n  ")
        sys.exit(1)
    print("all correctness checks passed")


if __name__ == "__main__":
    main()
