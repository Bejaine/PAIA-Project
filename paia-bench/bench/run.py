"""Resumable cell runner.

python -m bench.run --phase e1 --session S [--platform-tag T] [--contact C] [--max-retries 5] [--mem-cap-gb 12]
                    [--only REGEX] [--limit N] [--list] [--retry-failed]

A cell is one configuration (op, impl, shape, M, timing mode, hot/cold). Each cell's result is written
atomically to results/<tag>/<session>/cells/<cell_id>.json. Clean cells are never redone. Contaminated cells
are retried up to --max-retries, then marked unclean_exhausted. A cell whose op fails its correctness check
is marked failed and never timed.

Sessions named debug_* are for developing the harness on a busy machine: gate failures are recorded as
contamination instead of waiting, and their numbers must never be used.
"""
import argparse
import atexit
import datetime as dt
import hashlib
import json
import logging
import math
import os
import random
import re
import signal
import statistics
import sys
import time
from pathlib import Path

from bench import config
from bench.common import RESULTS_ROOT, SCHEMA_VERSION, default_platform_tag, provenance, write_json_atomic

log = logging.getLogger("paia.run")
STATUS_DONE = {"clean", "unclean_exhausted"}


# ----------------------------------------------------------------------------- cells

def cell_id(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def e1_cells():
    s = config.shapes_cfg()["e1_sizes_bytes"]
    sizes, b = [], s["start"]
    while b <= s["stop"]:
        sizes.append(b)
        b *= s["factor"]
    probes = [("read", "torch_sum"), ("copy", "torch_copy_"), ("write", "torch_fill_"), ("read", "triton_read")]
    cells = []
    for size in sizes:
        for op, impl in probes:
            for hot in ("hot", "cold"):
                for mode in ("isolated", "eager", "graph"):
                    cells.append({"phase": "e1", "op": op, "impl": impl, "precision": "fp32",
                                  "shape": {"M": None, "N": size, "K": None}, "model": None, "layer_name": None,
                                  "timing_mode": mode, "hot_or_cold": hot})
    return cells


def selftest_cells(n=12):
    """GPU-free cells for testing resume and retry logic. 'contaminate_attempts' makes the fake monitor
    report contamination on the first k attempts."""
    cells = []
    for i in range(n):
        k = 0 if i % 4 else (2 if i == 4 else (99 if i == 8 else 0))
        cells.append({"phase": "selftest", "op": "sleep", "impl": "fake", "precision": None,
                      "shape": {"M": None, "N": i, "K": None}, "model": None, "layer_name": None,
                      "timing_mode": "isolated", "hot_or_cold": "cold", "contaminate_attempts": k})
    return cells


def e2_cells():
    """cuBLAS BF16: synthetic sweep and real shapes at M=1 in all modes; M sweep on real shapes in graph mode."""
    cfg = config.shapes_cfg()
    base = {"phase": "e2", "op": "linear", "impl": "cublas_f_linear", "precision": "bf16", "hot_or_cold": "cold"}
    cells = []
    for name, sw in cfg["synthetic"].items():
        for n in sw["N"]:
            for mode in ("isolated", "eager", "graph"):
                cells.append({**base, "shape": {"M": 1, "N": n, "K": sw["K"]}, "model": "synthetic",
                              "layer_name": f"{name}_N{n}", "timing_mode": mode})
    for model, m in cfg["models"].items():
        for layer, (n, k) in m["shapes"].items():
            for mode in ("isolated", "eager", "graph"):
                cells.append({**base, "shape": {"M": 1, "N": n, "K": k}, "model": model, "layer_name": layer,
                              "timing_mode": mode})
            for M in cfg["batch_sweep"]["M"]:
                if M > 1:
                    cells.append({**base, "shape": {"M": M, "N": n, "K": k}, "model": model, "layer_name": layer,
                                  "timing_mode": "graph"})
    return cells


PHASES = {"e1": e1_cells, "e2": e2_cells, "selftest": selftest_cells}


def ordered(cells, seed):
    """Fixed shuffled order (stable across resumes) so slow drift is not confounded with any one variable."""
    cells = sorted(cells, key=cell_id)
    random.Random(seed).shuffle(cells)
    return cells


# ----------------------------------------------------------------------------- executors

class GpuContext:
    def __init__(self, sess_dir, mem_cap_bytes):
        import torch
        self.torch = torch
        p = torch.cuda.get_device_properties(0)
        self.sm_count, self.l2 = p.multi_processor_count, p.L2_cache_size
        self.sess_dir, self.mem_cap = sess_dir, mem_cap_bytes
        self.kernel_cache = {}
        self.triton_cfg = None

    def make_op(self, spec):
        from bench.ops import bw_probe as bp
        n = spec["shape"]["N"]
        if spec["phase"] == "e1":
            if spec["impl"] == "torch_sum":
                return bp.TorchRead(n)
            if spec["impl"] == "torch_copy_":
                return bp.TorchCopy(n)
            if spec["impl"] == "torch_fill_":
                return bp.TorchWrite(n)
            if spec["impl"] == "triton_read":
                return bp.TritonRead(n, self.triton_cfg)
        if spec["phase"] == "e2" and spec["impl"] == "cublas_f_linear":
            from bench.ops.cublas_bf16 import CublasBF16Linear
            sh = spec["shape"]
            return CublasBF16Linear(M=sh["M"], N=sh["N"], K=sh["K"], model=spec["model"], layer_name=spec["layer_name"])
        raise ValueError(f"no op for {spec}")


def execute_gpu(spec, ctx):
    """Build, check, then time one cell. Returns the measurement part of the cell record."""
    from bench import timing
    torch = ctx.torch
    tcfg = config.timing_cfg()
    rec = {}
    op = ctx.make_op(spec)
    try:
        rec["bytes"] = op.bytes()
        rec["bytes"]["total"] = sum(rec["bytes"].values())
        footprint = op.footprint_bytes()
        if footprint > ctx.mem_cap:
            return {**rec, "status": "failed", "error": f"footprint {footprint} B exceeds mem cap {ctx.mem_cap} B"}
        rot = timing.plan_rotation(footprint, spec["hot_or_cold"] == "hot", ctx.mem_cap, ctx.l2,
                                   tcfg["l2_rotation_factor"])
        rec.update(rot)
        try:
            op.build(rot["n_copies"])
        except torch.OutOfMemoryError as e:
            return {**rec, "status": "failed", "error": f"OOM building {rot['n_copies']} copies: {e}"[:500]}
        ok, detail = op.check()
        torch.cuda.synchronize()
        rec["correctness"] = {"ok": bool(ok), **detail}
        if not ok:
            return {**rec, "status": "failed", "error": "incorrect output; not timed"}
        key = (spec["op"], spec["impl"], json.dumps(spec["shape"]))
        if key not in ctx.kernel_cache:
            ctx.kernel_cache[key] = timing.kernel_names(op, rot["n_copies"])
        rec["kernel_names"] = ctx.kernel_cache[key]
        if spec["impl"] == "triton_read":
            rec["triton_cfg"] = {k: ctx.triton_cfg[k] for k in ("BLOCK", "num_warps", "waves")}
        if spec["timing_mode"] == "graph" and not op.graph_ok:
            return {**rec, "status": "failed", "error": "op not graph-capturable; graph mode skipped"}
        m = timing.measure(op, spec["timing_mode"], rot["n_copies"], tcfg)
        rec.update(m)
        rec["achieved_gbps"] = rec["bytes"]["total"] / (m["median_us"] * 1e-6) / 1e9
        return rec
    finally:
        op.free()
        del op
        torch.cuda.empty_cache()


def execute_fake(spec, ctx):
    with open(ctx.sess_dir / "executions.log", "a") as f:
        f.write(cell_id(spec) + "\n")
    time.sleep(0.3)
    samples = [10.0 + 0.01 * i for i in range(30)]
    from bench.timing import stats
    return {"bytes": {"weights": 0, "metadata": 0, "act_in": 0, "out": 0, "total": 0}, "samples_us": samples,
            **stats(samples, 0.03), "R_per_iter": 1, "n_copies": 1}


# ----------------------------------------------------------------------------- fake shared-machine pieces

class FakeMonitor:
    def __init__(self):
        self.next_reasons = []

    def gate(self):
        return {"ok": True, "reasons": []}

    def start(self):
        pass

    def stop(self):
        return {"telemetry": {}, "contamination_reasons": self.next_reasons}

    def close(self):
        pass


class FakeCanary:
    def measure(self):
        return 100.0

    def reference(self, runs):
        return {"median_gbps": 100.0, "cv": 0.0, "values": [100.0] * runs}

    def release(self):
        pass


# ----------------------------------------------------------------------------- session

class Session:
    def __init__(self, args):
        self.args = args
        self.debug = args.session.startswith("debug_")
        self.fake = args.phase == "selftest"
        root = Path(args.results_root) if args.results_root else RESULTS_ROOT
        self.tag = args.platform_tag or ("selftest" if self.fake else default_platform_tag())
        self.dir = root / self.tag / args.session
        self.cells_dir = self.dir / "cells"
        self.cells_dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self.gcfg, self.ccfg = config.gate_cfg(), config.canary_cfg()
        self.ref = None
        self.cell_durations = []

    # --- canary reference
    def ref_path(self):
        return self.dir / "canary_ref.json"

    def load_ref(self):
        p = self.ref_path()
        if p.exists():
            self.ref = json.loads(p.read_text())
        return self.ref

    def ref_age_min(self):
        if not self.ref:
            return math.inf
        return (time.time() - self.ref["current"]["t"]) / 60

    def take_ref(self, mon, canary):
        """Median of ref_runs canaries while idle; CV must be below ref_max_cv, else wait and retry."""
        backoff = self.gcfg["backoff_start_s"]
        while True:
            g = mon.gate()
            if g["ok"] or self.debug:
                r = canary.reference(self.ccfg["ref_runs"])
                r["t"] = time.time()
                r["t_iso"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
                r["gate"] = g
                if r["cv"] < self.ccfg["ref_max_cv"] or self.debug:
                    hist = (self.ref or {}).get("history", [])
                    if hist:
                        prev = hist[-1]["median_gbps"]
                        r["drift_vs_prev_pct"] = 100 * (r["median_gbps"] - prev) / prev
                        log.info("canary ref drift vs previous: %+.2f%%", r["drift_vs_prev_pct"])
                    self.ref = {"current": r, "history": hist + [r]}
                    write_json_atomic(self.ref_path(), self.ref)
                    log.info("canary reference %.2f GB/s (cv %.4f)%s", r["median_gbps"], r["cv"],
                             " [debug session, not validated]" if self.debug else "")
                    return
                log.info("canary reference unstable (cv %.4f >= %.4f); waiting", r["cv"], self.ccfg["ref_max_cv"])
            else:
                log.info("not idle for canary reference: %s", g["reasons"])
            canary.release()
            time.sleep(backoff)
            backoff = min(2 * backoff, self.gcfg["backoff_max_s"])

    # --- idle gate
    def wait_for_idle(self, mon, canary):
        """Block until the machine is idle and the canary is healthy. Returns (gate, canary_before, reasons).
        In debug sessions, returns immediately with the failure reasons attached."""
        backoff = self.gcfg["backoff_start_s"]
        while True:
            if not self.fake and self.ref_age_min() > self.ccfg["ref_refresh_min"]:
                g = mon.gate()
                if g["ok"]:
                    self.take_ref(mon, canary)
            g = mon.gate()
            reasons = list(g["reasons"])
            c = None
            if g["ok"] or self.debug:
                c = canary.measure()
                thr = self.ccfg["contaminated_below"] * self.ref["current"]["median_gbps"]
                if c < thr:
                    reasons.append(f"canary_before {c:.2f} < {thr:.2f} GB/s")
            if not reasons:
                return g, c, []
            if self.debug:
                return g, c, ["debug_session_gate_bypassed"] + reasons
            log.info("waiting %ds: %s", backoff, "; ".join(reasons))
            canary.release()
            if not self.fake:
                import torch
                torch.cuda.empty_cache()
            time.sleep(backoff)
            backoff = min(2 * backoff, self.gcfg["backoff_max_s"])

    def update_lock(self, remaining):
        from bench import monitor as M
        per = statistics.median(self.cell_durations) if self.cell_durations else 8.0
        eta = dt.datetime.now().astimezone() + dt.timedelta(seconds=remaining * per * 1.3)
        M.write_lock(self.args.contact, eta.isoformat(timespec="minutes"), self.args.session)


def load_cell(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run(args):
    sess = Session(args)
    fh = logging.FileHandler(sess.dir / "logs" / f"run_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), fh])
    rcfg = config.shapes_cfg()["runner"]
    cells = ordered(PHASES[args.phase](), rcfg["shuffle_seed"])
    if args.only:
        rx = re.compile(args.only)
        cells = [c for c in cells if rx.search(json.dumps(c, sort_keys=True))]
    if args.limit:
        cells = cells[:args.limit]
    if args.list:
        for c in cells:
            p = sess.cells_dir / f"{cell_id(c)}.json"
            prev = load_cell(p)
            print(cell_id(c), (prev or {}).get("status", "-"), c["op"], c["impl"], c["shape"], c["timing_mode"], c["hot_or_cold"])
        print(f"{len(cells)} cells")
        return 0

    todo = []
    for c in cells:
        prev = load_cell(sess.cells_dir / f"{cell_id(c)}.json")
        if prev and (prev.get("status") in STATUS_DONE or (prev.get("status") == "failed" and not args.retry_failed)):
            continue
        todo.append(c)
    log.info("session %s phase %s: %d cells, %d to run (%s)", args.session, args.phase, len(cells), len(todo),
             "DEBUG SESSION: numbers are not valid results" if sess.debug else "resume skips clean cells")
    if not todo:
        return 0

    if not (sess.fake or sess.debug or os.environ.get("PAIA_IN_TMUX") == "1" or args.no_tmux_check):
        log.error("real sessions must run inside tmux (so an SSH drop cannot kill a measurement); "
                  "start tmux and rerun, or pass --no-tmux-check")
        return 2
    from bench import monitor as M
    mcfg = config.monitor_cfg()
    fast, slow = M.cpu_sets()
    if mcfg["pin_cpus"] and fast != slow:
        os.sched_setaffinity(0, fast)
    if sess.fake:
        mon, canary, ctx, execute = FakeMonitor(), FakeCanary(), None, execute_fake
        sess.ref = {"current": {"median_gbps": 100.0, "t": time.time()}, "history": []}

        class _Ctx:
            sess_dir = sess.dir
        ctx = _Ctx()
    else:
        M.acquire_lock(args.contact, "estimating", args.session)
        atexit.register(M.release_lock)
        from bench.common import gpu_uuid
        mon = M.Monitor(mcfg, sess.gcfg, slow if (mcfg["pin_cpus"] and fast != slow) else None, gpu_uuid())
        canary = M.Canary(sess.ccfg)
        ctx = GpuContext(sess.dir, int(args.mem_cap_gb * 2**30))
        execute = execute_gpu
        if not (sess.dir / "env.json").exists():
            from bench import env
            write_json_atomic(sess.dir / "env.json", env.collect(sess.tag, args.session))
        sess.load_ref()
        if sess.ref_age_min() > sess.ccfg["ref_refresh_min"]:
            sess.take_ref(mon, canary)
        if any(c["impl"] == "triton_read" for c in todo):
            ctx.triton_cfg = tune_triton_under_gate(sess, mon, canary, ctx)

    def _term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, _term)

    max_retries = args.max_retries if args.max_retries is not None else rcfg["max_retries"]
    try:
        for idx, spec in enumerate(todo):
            cid = cell_id(spec)
            path = sess.cells_dir / f"{cid}.json"
            prev = load_cell(path)
            retries = prev.get("retries", 0) if prev and prev.get("status") == "contaminated" else 0
            while True:
                if not sess.fake:
                    sess.update_lock(len(todo) - idx)
                gate, canary_before, gate_reasons = sess.wait_for_idle(mon, canary)
                if sess.fake and spec.get("contaminate_attempts", 0) > retries:
                    mon.next_reasons = ["selftest_injected"]
                elif sess.fake:
                    mon.next_reasons = []
                t0 = time.time()
                mon.start()
                try:
                    res = execute(spec, ctx)
                except Exception as e:
                    import traceback
                    res = {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
                mon_res = mon.stop()
                t1 = time.time()
                canary_after = canary.measure()
                reasons = list(gate_reasons) + list(mon_res.get("contamination_reasons", []))
                ref = sess.ref["current"]["median_gbps"]
                thr = sess.ccfg["contaminated_below"] * ref
                if canary_after is not None and canary_after < thr:
                    reasons.append(f"canary_after {canary_after:.2f} < {thr:.2f} GB/s")
                reasons = list(dict.fromkeys(reasons))
                if res.get("status") == "failed":
                    status = "failed"
                elif reasons:
                    status = "contaminated"
                    retries += 1
                    if retries > max_retries or sess.debug:
                        status = "unclean_exhausted" if not sess.debug else "contaminated"
                else:
                    status = "clean"
                rec = build_record(spec, cid, sess, res, status, reasons, retries, canary_before, canary_after,
                                   ref, mon_res.get("telemetry", {}), t0, t1, os.sched_getaffinity(0))
                write_json_atomic(path, rec)
                sess.cell_durations.append(t1 - t0)
                extra = ""
                if rec.get("median_us") is not None and rec.get("achieved_gbps") is not None:
                    extra += f" median {rec['median_us']:.2f} us {rec['achieved_gbps']:.1f} GB/s"
                if status == "failed":
                    extra += f" ({rec.get('error')})"
                if reasons:
                    extra += f" reasons={reasons[:3]}"
                log.info("[%d/%d] %s %s %s %s N=%s %s -> %s%s", idx + 1, len(todo), cid, spec["op"], spec["impl"],
                         spec["timing_mode"], spec["shape"]["N"], spec["hot_or_cold"], status, extra)
                if status != "contaminated" or sess.debug:
                    break
    except KeyboardInterrupt as e:
        log.info("interrupted (%s); finished cells are saved, rerun with the same session to resume", e)
        return 130
    finally:
        mon.close()
    return 0


def tune_triton_under_gate(sess, mon, canary, ctx):
    from bench.ops import bw_probe as bp
    path = sess.dir / "triton_read_cfg.json"
    if path.exists():
        return json.loads(path.read_text())
    while True:
        gate, c0, gate_reasons = sess.wait_for_idle(mon, canary)
        mon.start()
        tmp = sess.dir / ".triton_read_cfg.candidate.json"
        tmp.unlink(missing_ok=True)
        cfg = bp.tune_triton_read(tmp, ctx.sm_count)
        mon_res = mon.stop()
        reasons = gate_reasons + mon_res["contamination_reasons"]
        if not reasons or sess.debug:
            cfg["tuned_clean"] = not reasons
            cfg["tune_reasons"] = reasons
            write_json_atomic(path, cfg)
            tmp.unlink(missing_ok=True)
            log.info("triton read probe config: BLOCK=%d warps=%d waves=%d%s", cfg["BLOCK"], cfg["num_warps"],
                     cfg["waves"], "" if not reasons else f" (debug, contaminated: {reasons[:2]})")
            return cfg
        log.info("triton tuning contaminated (%s); retrying", reasons[:2])


def build_record(spec, cid, sess, res, status, reasons, retries, c_before, c_after, ref, tel, t0, t1, affinity):
    iso = lambda t: dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")
    prov = provenance()
    keys_from_res = ["R_per_iter", "n_copies", "rotation_bytes", "l2_bytes", "l2_safe", "l2_coverage", "bytes",
                     "samples_us", "median_us", "p5_us", "p95_us", "mean_us", "std_us", "cv", "noisy", "n_samples",
                     "achieved_gbps", "kernel_names", "correctness", "est_op_us", "warmup_calls", "warmup_ms",
                     "triton_cfg", "n_copies_wanted", "error", "traceback"]
    rec = {
        "schema_version": SCHEMA_VERSION, "cell_id": cid, "session_id": sess.args.session, "platform_tag": sess.tag,
        "phase": spec["phase"], "op": spec["op"], "impl": spec["impl"], "precision": spec["precision"],
        "shape": spec["shape"], "model": spec["model"], "layer_name": spec["layer_name"],
        "timing_mode": spec["timing_mode"], "hot_or_cold": spec["hot_or_cold"],
        **{k: res.get(k) for k in keys_from_res},
        "pct_of_b0": None,  # filled by analyze.py from this session's measured B0
        "status": status, "contamination_reasons": reasons, "retries": retries,
        "canary_before_gbps": c_before, "canary_after_gbps": c_after, "canary_ref_gbps": ref,
        "telemetry": tel, "cpu_affinity": sorted(affinity),
        "git_commit": prov["git_commit"], "git_dirty": prov["git_dirty"], "image_id": prov["image_id"],
        "env_ref": "../env.json", "debug_session": sess.debug,
        "t_start": iso(t0), "t_end": iso(t1), "spec": spec,
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--session", required=True)
    ap.add_argument("--platform-tag", default=None)
    ap.add_argument("--contact", default=os.environ.get("PAIA_CONTACT", "see PAIA-Project repo (P2)"))
    ap.add_argument("--max-retries", type=int, default=None)
    ap.add_argument("--mem-cap-gb", type=float, default=config.timing_cfg()["mem_cap_gb"])
    ap.add_argument("--only", default=None, help="regex over the cell spec JSON")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--resume", action="store_true", help="accepted for clarity; resuming is always on")
    ap.add_argument("--results-root", default=None, help="override results/ (tests)")
    ap.add_argument("--no-tmux-check", action="store_true")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
