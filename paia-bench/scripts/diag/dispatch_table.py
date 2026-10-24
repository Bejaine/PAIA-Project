"""Dispatch table: the CUDA kernels cuBLAS runs for every E2 shape (M=1 and the M sweep), recorded back-to-back in one
fresh process the way selfcheck does. No timing.
Why: on torch 2.11.0+cu128 (rtx6000ada stack) torch.profiler returns no CUDA events once the process has been idle for
a few seconds between profiler sessions, so most E2 cells recorded kernel_names = [] (findings.md). Dispatch depends only
on shape, GPU model and library stack, so bench.analyze --dispatch-table fills the empty lists from this table and checks
it against every cell that did record names.
Usage: python scripts/diag/dispatch_table.py <out.json>"""
import json, sys
import torch
from bench import config, timing
from bench.common import write_json_atomic
from bench.ops.cublas_bf16 import CublasBF16Linear
from bench.run import e2_cells

if __name__ == "__main__":
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    tcfg = config.timing_cfg()
    shapes = sorted({(c["shape"]["M"], c["shape"]["N"], c["shape"]["K"]) for c in e2_cells()})
    table = {}
    for M, N, K in shapes:
        op = CublasBF16Linear(M=M, N=N, K=K, model="dispatch", layer_name=f"{M}x{N}x{K}")
        rot = timing.plan_rotation(op.footprint_bytes(), False, 12 << 30, l2, tcfg["l2_rotation_factor"])
        op.build(rot["n_copies"])
        names = timing.kernel_names(op, rot["n_copies"])
        if not names:   # never accept an empty capture: retry once, then fail loudly
            names = timing.kernel_names(op, rot["n_copies"])
        assert names, f"no kernels captured for M={M} N={N} K={K}"
        table[f"{M}x{N}x{K}"] = names
        op.free(); del op; torch.cuda.empty_cache()
    from bench.common import gpu_uuid
    write_json_atomic(sys.argv[1], {"gpu_uuid": gpu_uuid(), "torch": torch.__version__, "n_shapes": len(table), "table": table})
    print(f"wrote {len(table)} shapes to {sys.argv[1]}")
