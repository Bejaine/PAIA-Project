"""Dispatch table: the CUDA kernels each implementation runs for every E2 / E4 cell shape (M=1 and the M sweep), recorded
back-to-back in one fresh process the way selfcheck does. No timing.
Why: on torch 2.11 (rtx6000ada stacks) torch.profiler returns no CUDA events once the process has been idle for a few
seconds between profiler sessions, so most cells recorded kernel_names = [] (findings.md). Dispatch depends only on
implementation, shape, GPU model and library stack, so bench.analyze --dispatch-table fills the empty lists from this
table and checks it against every cell that did record names.
Keys: "<impl>|<M>x<N>x<K>".
Usage: python scripts/diag/dispatch_table.py <out.json> [e2] [e4]   (default: e2)"""
import sys
import torch
from bench import config, timing
from bench.common import gpu_uuid, write_json_atomic
from bench.ops.cublas_bf16 import CublasBF16Linear
from bench.run import e2_cells, e4_cells


def make(impl, M, N, K):
    if impl == "cublas_f_linear":
        return CublasBF16Linear(M=M, N=N, K=K, model="dispatch", layer_name=f"{M}x{N}x{K}")
    from bench.ops.lib_quant import E4_OPS
    return E4_OPS[impl](M=M, N=N, K=K, model="dispatch", layer_name=f"{M}x{N}x{K}")


if __name__ == "__main__":
    phases = sys.argv[2:] or ["e2"]
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    tcfg = config.timing_cfg()
    cells = (e2_cells() if "e2" in phases else []) + (e4_cells() if "e4" in phases else [])
    keys = sorted({(c["impl"], c["shape"]["M"], c["shape"]["N"], c["shape"]["K"]) for c in cells})
    table = {}
    for impl, M, N, K in keys:
        op = make(impl, M, N, K)
        rot = timing.plan_rotation(op.footprint_bytes(), False, 12 << 30, l2, tcfg["l2_rotation_factor"])
        op.build(rot["n_copies"])
        names = timing.kernel_names(op, rot["n_copies"])
        if not names:   # never accept an empty capture: retry once, then fail loudly
            names = timing.kernel_names(op, rot["n_copies"])
        assert names, f"no kernels captured for {impl} M={M} N={N} K={K}"
        table[f"{impl}|{M}x{N}x{K}"] = names
        op.free(); del op; torch.cuda.empty_cache()
    write_json_atomic(sys.argv[1], {"gpu_uuid": gpu_uuid(), "torch": torch.__version__, "phases": phases,
                                    "n_entries": len(table), "table": table})
    print(f"wrote {len(table)} entries to {sys.argv[1]}")
