"""Reproducer (rtx6000ada stack): torch.profiler stops reporting CUDA kernels after idle gaps between profiler sessions.
Usage: python scripts/diag/profiler_idle_repro.py <pause_s> [warm]   (12 s: names lost after 2 cells; 2 s: never)"""
from bench.ops.cublas_bf16 import CublasBF16Linear
from bench import timing, config
if __name__ == "__main__":
    pause = float(sys.argv[1]); tcfg = config.timing_cfg(); got = []; t0 = time.time(); fail_t = None
    for i in range(30):
        time.sleep(pause)
        op = CublasBF16Linear(M=1, N=256 * (i + 1), K=4096, model="d", layer_name=f"n{i}"); op.build(4)
        if len(sys.argv) > 2:
            t1 = time.time()
            while time.time() - t1 < 0.2: op.call(0)
            torch.cuda.synchronize()
        n = len(timing.kernel_names(op, 4)); got.append(n)
        if n == 0 and fail_t is None: fail_t = time.time() - t0
        timing.measure(op, "graph", 4, tcfg); op.free(); del op; torch.cuda.empty_cache()
    print(f"pause {pause:5.1f}s warm={len(sys.argv) > 2}", got, "first failure at", got.index(0) if 0 in got else None, "after", fail_t and round(fail_t), "s", flush=True)
