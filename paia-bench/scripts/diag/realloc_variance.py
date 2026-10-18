"""Is there variance between allocations that within-cell statistics cannot see? For a few E2 shapes, measure
the graph-mode median (same harness code as the runner) (a) 10 times on ONE allocation and (b) once each on 10
fresh allocations. Run only on an idle machine. Diagnostic tooling."""
import json, statistics as st, sys, torch
from bench import config, timing
from bench.ops.cublas_bf16 import CublasBF16Linear
tcfg = config.timing_cfg()
L2 = torch.cuda.get_device_properties(0).L2_cache_size
out = {}
for (N, K) in [(1024, 4096), (4096, 4096), (14336, 4096), (512, 3584)]:
    op = CublasBF16Linear(M=1, N=N, K=K)
    rot = timing.plan_rotation(op.footprint_bytes(), False, int(12 * 2**30), L2, tcfg["l2_rotation_factor"])
    op.build(rot["n_copies"])
    same = [timing.measure(op, "graph", rot["n_copies"], tcfg)["median_us"] for _ in range(10)]
    op.free()
    fresh = []
    for _ in range(10):
        op = CublasBF16Linear(M=1, N=N, K=K); op.build(rot["n_copies"])
        fresh.append(timing.measure(op, "graph", rot["n_copies"], tcfg)["median_us"]); op.free()
    r = lambda v: 100 * (max(v) - min(v)) / st.median(v)
    out[f"{N}x{K}"] = {"same_alloc": same, "fresh_alloc": fresh}
    print(f"{N:6d}x{K:5d} n_copies {rot['n_copies']:3d}  same alloc: median {st.median(same):8.2f} us range {r(same):5.2f}%   "
          f"fresh allocs: median {st.median(fresh):8.2f} us range {r(fresh):5.2f}%  min {min(fresh):.2f} max {max(fresh):.2f}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
