"""Diagnostic (not a result): does the DCGM exporter's background profiling (DCP metrics) change our timings?
Run once while DCGM profiling is paused and once after it is resumed, on the same card; compare the two JSON files.
Probes use the harness's ops and timing.measure: small and large Triton reads and fills, cuBLAS k_proj and q_proj, and
the bandwidth canary. 5 rounds each; the round-to-round spread is the noise floor for the comparison.
Usage: python scripts/diag/dcgm_ab.py <label> <out.json> [<compare_with.json>]"""
import json, statistics as st, sys
from pathlib import Path
import torch
from bench import config, timing, monitor as M
from bench.ops.bw_probe import TorchWrite, TritonRead, tune_triton_read
from bench.ops.cublas_bf16 import CublasBF16Linear

if __name__ == "__main__":
    label, out = sys.argv[1], sys.argv[2]
    tcfg = config.timing_cfg()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    tri = tune_triton_read(Path(out).parent / f"dcgm_ab_triton_cfg_{label}.json", sm)
    specs = [("triton_read_1MiB", lambda: TritonRead(1 << 20, tri), ("graph", "eager")),
             ("triton_read_64MiB", lambda: TritonRead(64 << 20, tri), ("graph",)),
             ("triton_read_1GiB", lambda: TritonRead(1 << 30, tri), ("graph",)),
             ("torch_fill_1MiB", lambda: TorchWrite(1 << 20), ("graph",)),
             ("cublas_llama_k", lambda: CublasBF16Linear(M=1, N=1024, K=4096, model="diag", layer_name="k"), ("graph", "eager", "isolated")),
             ("cublas_llama_q", lambda: CublasBF16Linear(M=1, N=4096, K=4096, model="diag", layer_name="q"), ("graph",))]
    built = []
    for name, mk, modes in specs:
        op = mk()
        rot = timing.plan_rotation(op.footprint_bytes(), False, 12 << 30, l2, tcfg["l2_rotation_factor"])
        op.build(rot["n_copies"])
        built += [(f"{name}_{m}", op, rot["n_copies"], m) for m in modes]
    canary = M.Canary(config.canary_cfg())
    res = {k: [] for k, *_ in built}; res["canary_gbps"] = []
    for rnd in range(5):
        for key, op, nc, mode in built:
            res[key].append(timing.measure(op, mode, nc, tcfg)["median_us"])
        res["canary_gbps"].append(st.median(canary.measure() for _ in range(5)))
    med = {k: st.median(v) for k, v in res.items()}
    spread = {k: (max(v) - min(v)) / st.median(v) for k, v in res.items()}
    from bench.common import gpu_uuid
    json.dump({"label": label, "gpu_uuid": gpu_uuid(), "raw": res, "median": med, "spread": spread}, open(out, "w"), indent=1)
    print(f"{label}: " + ", ".join(f"{k} {v:.3f}" for k, v in med.items()))
    if len(sys.argv) > 3:
        ref = json.load(open(sys.argv[3]))
        print(f"{'probe':28s}{ref['label']:>12s}{label:>12s}{'change':>10s}{'noise':>9s}")
        for k in med:
            a, b = ref["median"][k], med[k]
            print(f"{k:28s}{a:12.3f}{b:12.3f}{b / a - 1:+10.2%}{max(spread[k], ref['spread'][k]):9.2%}")
