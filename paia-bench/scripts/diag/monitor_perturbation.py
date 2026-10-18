"""Does our own monitor perturb measurements? Times a cold 64 MiB fill and read (isolated mode, 600 samples)
with (a) nothing running, (b) an NVML-polling process (100 ms), (c) a psutil-scanning process (500 ms),
(d) both, pinned like the real monitor. Reports median and the fraction of samples > 5% above median.
Run only when the machine is idle. Diagnostic tooling, not a benchmark."""
import multiprocessing as mp, os, statistics as st, sys, time, json
import torch
from bench.monitor import cpu_sets

def nvml_loop(stop):
    import pynvml as N
    N.nvmlInit(); h = N.nvmlDeviceGetHandleByIndex(0)
    while not stop.is_set():
        N.nvmlDeviceGetUtilizationRates(h); N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM)
        N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU); N.nvmlDeviceGetPowerUsage(h)
        N.nvmlDeviceGetCurrentClocksEventReasons(h); N.nvmlDeviceGetComputeRunningProcesses(h)
        try: N.nvmlDeviceGetProcessUtilization(h, 0)
        except N.NVMLError: pass
        time.sleep(0.1)

def psutil_loop(stop):
    import psutil
    while not stop.is_set():
        for p in psutil.process_iter(["uids", "username"]):
            try: p.cpu_percent(None)
            except psutil.Error: pass
        time.sleep(0.5)

def worker(kind, stop, cpus):
    os.sched_setaffinity(0, cpus)
    if kind in ("nvml", "both"):
        import threading; threading.Thread(target=nvml_loop, args=(stop,), daemon=True).start()
    if kind in ("psutil", "both"):
        import threading; threading.Thread(target=psutil_loop, args=(stop,), daemon=True).start()
    stop.wait()

def time_op(fn, bufs, n=600):
    for i in range(30): fn(bufs[i % len(bufs)])
    torch.cuda.synchronize()
    out = []
    for i in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record(); fn(bufs[i % len(bufs)]); e.record(); e.synchronize()
        out.append(s.elapsed_time(e) * 1e3)
    m = st.median(out)
    return {"median_us": m, "slow_frac": sum(x > 1.05 * m for x in out) / n, "p95_over_med": sorted(out)[int(0.95 * n)] / m}

if __name__ == "__main__":
    fast, slow = cpu_sets()
    os.sched_setaffinity(0, fast)
    n = (64 << 20) // 4
    bufs = [torch.rand(n, device="cuda") for _ in range(2)]
    ops = {"fill": lambda b: b.fill_(0.5), "read": lambda b: b.sum()}
    res = {}
    ctx = mp.get_context("spawn")
    for rep in range(2):
        for kind in ("none", "nvml", "psutil", "both"):
            stop = ctx.Event(); p = None
            if kind != "none":
                p = ctx.Process(target=worker, args=(kind, stop, slow)); p.start(); time.sleep(1.0)
            for name, fn in ops.items():
                r = time_op(fn, bufs)
                res.setdefault(f"{name}/{kind}", []).append(r)
                print(f"rep{rep} {name:4s} {kind:6s} median {r['median_us']:8.1f} us  slow>5%: {100*r['slow_frac']:5.1f}%  p95/med {r['p95_over_med']:.3f}", flush=True)
            if p: stop.set(); p.join()
    json.dump(res, open(sys.argv[1], "w"), indent=1)
