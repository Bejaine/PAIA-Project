"""Diagnostic (not a result): does CPU load on other cores change our timings on a discrete GPU?
Probes are pinned to one core; synthetic load runs on other cores of the same job: pure spin, or memory streaming
(numpy copies of 256 MiB arrays), on the same L3 as the probe core or on the other socket. Conditions are interleaved
in a shuffled order over several rounds, so slow drift and other users' load fall on every condition alike.
Usage: python foreign_cpu_sensitivity.py <out.json> <probe_cpu> <near_cpus,...> <far_cpus,...>"""
import json, multiprocessing as mp, os, random, statistics as st, sys, time
import torch, torch.nn.functional as F


def spin(cpu, stop):
    os.sched_setaffinity(0, [cpu])
    while not stop.is_set():
        pass


def stream(cpu, stop):
    import numpy as np
    os.sched_setaffinity(0, [cpu])
    a = np.ones(32 << 20); b = np.empty_like(a)   # 256 MiB each
    while not stop.is_set():
        np.copyto(b, a); np.copyto(a, b)


def probes():
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    small = [torch.rand((1 << 20) // 4, device="cuda") for _ in range(4 * l2 // (1 << 20) + 1)]   # 1 MiB, rotated
    big = torch.rand((1 << 30) // 4, device="cuda")
    ws = [torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16) for _ in range(4 * l2 // (8 << 20) + 1)]  # k_proj
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for w in ws[:3]: F.linear(x, w)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for i in range(20): F.linear(x, ws[i % len(ws)])

    def isolated(fn, n=300):
        out = []
        for i in range(n):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); a.record(); fn(i); b.record(); b.synchronize()
            out.append(a.elapsed_time(b) * 1e3)
        return st.median(out)

    def eager(fn, n=30, r=100):
        out = []
        for _ in range(n):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); a.record()
            for i in range(r): fn(i)
            b.record(); b.synchronize(); out.append(a.elapsed_time(b) * 1e3 / r)
        return st.median(out)

    def graph(n=30):
        out = []
        for _ in range(n):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); a.record(); g.replay(); b.record(); b.synchronize()
            out.append(a.elapsed_time(b) * 1e3 / 20)
        return st.median(out)

    return {
        "read1MiB_isolated_us": lambda: isolated(lambda i: small[i % len(small)].sum()),
        "read1MiB_eager_us": lambda: eager(lambda i: small[i % len(small)].sum()),
        "kproj_isolated_us": lambda: isolated(lambda i: F.linear(x, ws[i % len(ws)])),
        "kproj_eager_us": lambda: eager(lambda i: F.linear(x, ws[i % len(ws)])),
        "kproj_graph_us": lambda: graph(),
        "read1GiB_isolated_us": lambda: isolated(lambda i: big.sum(), n=40),
    }


if __name__ == "__main__":
    out, probe_cpu = sys.argv[1], int(sys.argv[2])
    near = [int(c) for c in sys.argv[3].split(",")]; far = [int(c) for c in sys.argv[4].split(",")]
    os.sched_setaffinity(0, [probe_cpu])
    conds = {"none": [], "spin_far1": [("spin", far[0])], "spin_near2": [("spin", c) for c in near[:2]], "stream_near2": [("stream", c) for c in near[:2]],
             "stream_far4": [("stream", c) for c in far[:4]],
             "stream_all8": [("stream", c) for c in (near + far)[:8]]}
    P = probes()
    for f in P.values(): f()   # warm up every probe once
    ctx = mp.get_context("spawn"); res = {c: {k: [] for k in P} for c in conds}
    rounds = 6
    for rnd in range(rounds):
        order = list(conds); random.Random(rnd).shuffle(order)
        for c in order:
            stop = ctx.Event(); ps = [ctx.Process(target=spin if k == "spin" else stream, args=(cpu, stop)) for k, cpu in conds[c]]
            for p in ps: p.start()
            time.sleep(1.0)   # same settle for every condition, including none
            for k, f in P.items(): res[c][k].append(f())
            res[c].setdefault("probe_cpu_khz", []).append(int(open(f"/sys/devices/system/cpu/cpu{probe_cpu}/cpufreq/scaling_cur_freq").read()))
            stop.set()
            for p in ps: p.join()
        print(f"round {rnd} done", flush=True)
    summ = {c: {k: st.median(v) for k, v in d.items()} for c, d in res.items()}
    print("probe-core kHz after each condition:", {c: summ[c]["probe_cpu_khz"] for c in conds})
    print(f"{'probe':22s}" + "".join(f"{c:>14s}" for c in conds))
    for k in P:
        base = summ["none"][k]
        print(f"{k:22s}{base:14.2f}" + "".join(f"{summ[c][k] / base - 1:+13.2%} " for c in list(conds)[1:]))
    spread = {k: (max(res['none'][k]) - min(res['none'][k])) / summ['none'][k] for k in P}
    print("round-to-round spread of 'none' (max-min)/median:", {k: f"{v:.2%}" for k, v in spread.items()})
    json.dump({"raw": res, "median": summ, "none_spread": spread, "probe_cpu": probe_cpu, "near": near, "far": far,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}, open(out, "w"), indent=1)
