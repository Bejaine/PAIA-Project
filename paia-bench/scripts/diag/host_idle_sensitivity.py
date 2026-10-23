"""Diagnostic (not a result): do E1/E2 cells change when other CPU cores are busy or idle?
Uses the harness's own ops and timing.measure (rotation, isolated/eager/graph) for small cold cells, and compares
conditions interleaved in shuffled rounds: 'none' (rest of our job idle), 'spin_far1' (one spinning core on the other
socket), 'stream_near4' (4 logical CPUs streaming memory on other physical cores sharing the probe's L3) and
'stream_far4' (4 on the other socket). Probe pinned to one core whose SMT sibling stays idle: SLURM gives each job whole
cores, so foreign processes never share our physical cores.
Usage: PYTHONPATH=. python host_idle_sensitivity.py <out.json> <probe_cpu> <far_cpu> <near_cpus,...> <far_cpus,...>"""
import json, multiprocessing as mp, os, random, statistics as st, sys, time
import torch
from bench import config, timing
from bench.ops.bw_probe import TorchRead, TorchWrite, TorchCopy, TritonRead, tune_triton_read
from bench.ops.cublas_bf16 import CublasBF16Linear


def spin(cpu, stop):
    os.sched_setaffinity(0, [cpu])
    while not stop.is_set():
        pass


def stream(cpu, stop):
    import numpy as np
    os.sched_setaffinity(0, [cpu])
    a = np.ones(32 << 20); b = np.empty_like(a)
    while not stop.is_set():
        np.copyto(b, a); np.copyto(a, b)


if __name__ == "__main__":
    out, probe_cpu, far = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    near = [int(c) for c in sys.argv[4].split(",")]; farl = [int(c) for c in sys.argv[5].split(",")]
    os.sched_setaffinity(0, [probe_cpu])
    tcfg = config.timing_cfg()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    from pathlib import Path
    tri = tune_triton_read(Path(out).parent / "host_idle_triton_cfg.json", sm)
    cells = []
    for size in (1 << 20, 8 << 20):
        for name, mk in (("triton_read", lambda s: TritonRead(s, tri)), ("torch_sum", TorchRead),
                         ("torch_fill", TorchWrite), ("torch_copy", TorchCopy)):
            for mode in ("isolated", "eager", "graph"):
                cells.append((f"{name}_{size >> 20}MiB_{mode}", mk, size, mode))
    for (n, k), lbl in (((1024, 4096), "llama_k"), ((4096, 4096), "llama_q")):
        for mode in ("isolated", "eager", "graph"):
            cells.append((f"cublas_{lbl}_{mode}", lambda s, n=n, k=k, lbl=lbl: CublasBF16Linear(M=1, N=n, K=k, model="diag", layer_name=lbl), None, mode))
    built = {}
    for key, mk, size, mode in cells:
        op = mk(size)
        rot = timing.plan_rotation(op.footprint_bytes(), False, 12 << 30, l2, tcfg["l2_rotation_factor"])
        op.build(rot["n_copies"]); built[key] = (op, rot["n_copies"], mode)
    conds = {"none": [], "spin_far1": [("spin", far)], "stream_near4": [("stream", c) for c in near],
             "stream_far4": [("stream", c) for c in farl]}
    ctx = mp.get_context("spawn"); res = {c: {k: [] for k in built} for c in conds}
    for rnd in range(5):
        order = list(conds); random.Random(100 + rnd).shuffle(order)
        for c in order:
            stop = ctx.Event()
            ps = [ctx.Process(target=spin if kind == "spin" else stream, args=(cpu, stop)) for kind, cpu in conds[c]]
            for p in ps: p.start()
            time.sleep(1.0)
            for key, (op, nc, mode) in built.items():
                res[c][key].append(timing.measure(op, mode, nc, tcfg)["median_us"])
            stop.set()
            for p in ps: p.join()
        print(f"round {rnd} done", flush=True)
    summ = {c: {k: st.median(v) for k, v in d.items()} for c, d in res.items()}
    spread = {k: (max(res["none"][k]) - min(res["none"][k])) / summ["none"][k] for k in built}
    print(f"{'cell':30s}{'none us':>10s}" + "".join(f"{c:>14s}" for c in list(conds)[1:]) + f"{'none spread':>13s}")
    for k in built:
        b = summ["none"][k]
        print(f"{k:30s}{b:10.2f}" + "".join(f"{summ[c][k] / b - 1:+14.2%}" for c in list(conds)[1:]) + f"{spread[k]:13.2%}")
    json.dump({"raw": res, "median": summ, "none_spread": spread, "probe_cpu": probe_cpu, "far": far, "near": near, "far_cpus": farl,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton_cfg": tri}, open(out, "w"), indent=1)
