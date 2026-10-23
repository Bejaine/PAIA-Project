"""Diagnostic (not a result): does freeing a large buffer slow the harness canary right afterwards, and for how long?
Canary = bench.monitor.Canary's procedure: 8 x 64 MiB fp32, 2 warmup passes, median of 10 timed passes."""
import statistics, time, torch, json, sys

g = torch.Generator(device="cuda").manual_seed(1234)
bufs = [torch.rand((512 << 20) // 8 // 4, device="cuda", generator=g) for _ in range(8)]
nbytes = sum(b.numel() * 4 for b in bufs)

def canary(warm=2, reps=10):
    for _ in range(warm):
        for b in bufs: b.sum()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for s, e in ev:
        s.record()
        for b in bufs: b.sum()
        e.record()
    torch.cuda.synchronize()
    return statistics.median(nbytes / (s.elapsed_time(e) * 1e-3) / 1e9 for s, e in ev)

base = [canary() for _ in range(20)]
print(f"baseline canary {statistics.median(base):.1f} GB/s (min {min(base):.1f} max {max(base):.1f})", flush=True)
res = {"baseline": base}
for gib in (0.5, 1, 2, 4, 8):
    for mode in ("empty_cache", "keep_cached"):
        x = torch.empty(int(gib * (1 << 30)) // 4, device="cuda"); x.fill_(1.0); x.sum(); torch.cuda.synchronize()
        del x
        if mode == "empty_cache":
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        trace = []
        while time.perf_counter() - t0 < 1.5:
            v = canary(); trace.append((round((time.perf_counter() - t0) * 1e3, 1), round(v, 1)))
        torch.cuda.empty_cache(); time.sleep(1.0)
        first = trace[0][1]; low = [t for t, v in trace if v < 0.97 * statistics.median(base)]
        print(f"{gib:>4} GiB {mode:11s} first canary {first:7.1f} GB/s ({first / statistics.median(base):.3f}x), "
              f"slow (<0.97x) until {max(low) if low else 0:6.1f} ms; trace head {trace[:4]}", flush=True)
        res[f"{gib}GiB_{mode}"] = trace
json.dump(res, open(sys.argv[1], "w"))
