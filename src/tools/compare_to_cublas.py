"""Per-shape table: our kernel vs cuBLAS, from harness cell JSONs. Reads clean cells of two phases/impls (default:
phase e2 impl cublas_f_linear vs phase s1 impl ours_bf16_gemv), M=1, graph mode, medians across the given sessions.
    python src/tools/compare_to_cublas.py --b0 239.5 results/dgxspark_gb10/e2v2_s1 results/dgxspark_gb10/e2v2_s2 --ours results/dgxspark_gb10/s1_<id>
Output: % of B0 and speedup per shape; marks shapes whose tau-limited ceiling (--tau-us, default 1.7) is below 100%."""
import argparse, glob, json, statistics
from pathlib import Path


def load(dirs, phase, impl, mode):
    out = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / "cells" / "*.json")):
            c = json.load(open(f))
            if c.get("phase") == phase and c.get("impl") == impl and c.get("status") == "clean" \
                    and c["timing_mode"] == mode and c["shape"]["M"] == 1 and c.get("median_us"):
                out.setdefault((c["model"], c["layer_name"], c["shape"]["N"], c["shape"]["K"]), []).append(c)
    return {k: (statistics.median(x["median_us"] for x in v), v[0]["bytes"]["total"]) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_sessions", nargs="+")
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--b0", type=float, required=True, help="GB/s")
    ap.add_argument("--tau-us", type=float, default=1.7)
    ap.add_argument("--mode", default="graph")
    a = ap.parse_args()
    base = load(a.base_sessions, "e2", "cublas_f_linear", a.mode)
    ours = load(a.ours, "s1", "ours_bf16_gemv", a.mode)
    print(f"{'model':<12}{'layer':<14}{'MiB':>8}{'cuBLAS %B0':>11}{'ours %B0':>9}{'speedup':>8}{'tau ceiling %':>14}")
    for k in sorted(ours, key=lambda k: ours[k][1]):
        model, layer, N, K = k
        us, B = ours[k]
        ceil = 100 * (B / (a.tau_us * 1e-6 + B / (a.b0 * 1e9))) / (a.b0 * 1e9)
        if k in base:
            bus = base[k][0]
            print(f"{model:<12}{layer:<14}{B/2**20:>8.1f}{100*B/(bus*1e-6)/1e9/a.b0:>11.1f}{100*B/(us*1e-6)/1e9/a.b0:>9.1f}{bus/us:>8.2f}{ceil:>14.1f}")
        else:
            print(f"{model:<12}{layer:<14}{B/2**20:>8.1f}{'n/a':>11}{100*B/(us*1e-6)/1e9/a.b0:>9.1f}{'n/a':>8}{ceil:>14.1f}")


if __name__ == "__main__":
    main()
