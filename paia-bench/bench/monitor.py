"""Shared-machine protocol: idle gate, live monitor, bandwidth canary, courtesy lockfile.

The monitor runs in a separate process (not a thread) pinned to the slowest CPU cores, so its NVML and
psutil polling never contends for the benchmark's GIL or fast cores. The main process talks to it over a pipe.
"""
import datetime as dt
import json
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path

LOCKFILE = Path("/tmp/paia_bench.lock")

CLOCK_REASON_BITS = {
    0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting",
}


def decode_clock_reasons(mask):
    return [n for b, n in CLOCK_REASON_BITS.items() if mask & b]


# ----------------------------------------------------------------------------- CPU topology

def cpu_sets():
    """(fast_cpus, slow_cpus) by cpuinfo_max_freq. Equal sets when cores are homogeneous or unreadable."""
    allowed = sorted(os.sched_getaffinity(0))
    freq = {}
    for c in allowed:
        try:
            freq[c] = int(open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq").read())
        except OSError:
            return allowed, allowed
    hi, lo = max(freq.values()), min(freq.values())
    if hi == lo:
        return allowed, allowed
    return [c for c in allowed if freq[c] == hi], [c for c in allowed if freq[c] == lo]


# ----------------------------------------------------------------------------- monitor child

class _Sampler:
    """Runs inside the monitor process."""

    def __init__(self, main_pid, cfg, gpu_uuid=None):
        import psutil
        import pynvml as N
        from bench.common import nvml_handle
        self.psutil, self.N = psutil, N
        N.nvmlInit()
        self.h = nvml_handle(N, gpu_uuid)
        self.main_pid, self.cfg = main_pid, cfg
        self.uid = os.getuid()
        self.unsupported = set()
        self.procs = {}
        self.last_cpu_scan = None
        self.last_util_ts = 0

    # --- helpers
    def _nvml(self, name, fn):
        if name in self.unsupported:
            return None
        try:
            return fn()
        except self.N.NVMLError_NotSupported:
            self.unsupported.add(name)
            return None
        except self.N.NVMLError:
            return None

    def tree(self):
        try:
            p = self.psutil.Process(self.main_pid)
            return {p.pid} | {c.pid for c in p.children(recursive=True)}
        except self.psutil.Error:
            return {self.main_pid, os.getpid()}

    def _proc_desc(self, pid):
        try:
            p = self.psutil.Process(pid)
            return f"{pid}:{p.username()}:{p.name()}"
        except self.psutil.Error:
            return f"{pid}:?:?"

    def foreign_gpu(self, tree):
        N = self.N
        procs = self._nvml("compute_procs", lambda: N.nvmlDeviceGetComputeRunningProcesses(self.h)) or []
        foreign = [p.pid for p in procs if p.pid not in tree]
        active = []
        try:
            samples = N.nvmlDeviceGetProcessUtilization(self.h, self.last_util_ts)
            for s in samples:
                self.last_util_ts = max(self.last_util_ts, s.timeStamp)
                if s.pid not in tree and (s.smUtil > 0 or s.memUtil > 0):
                    active.append(s.pid)
        except N.NVMLError:
            pass  # NotFound when there are no new samples
        return foreign, active

    def foreign_cpu_cores(self):
        """CPU cores used by processes of other uids since the previous call."""
        ps = self.psutil
        seen, total, by_user = set(), 0.0, {}
        for p in ps.process_iter(["uids", "username"]):
            pid = p.pid
            seen.add(pid)
            try:
                if p.info["uids"] is None or p.info["uids"].real == self.uid:
                    continue
                proc = self.procs.get(pid)
                if proc is None:
                    self.procs[pid] = p
                    p.cpu_percent(None)  # prime; first reading is meaningless
                    continue
                c = proc.cpu_percent(None) / 100.0
                total += c
                if c > 0.05:
                    by_user[p.info["username"]] = by_user.get(p.info["username"], 0) + c
            except ps.Error:
                pass
        for pid in list(self.procs):
            if pid not in seen:
                del self.procs[pid]
        top = sorted(by_user.items(), key=lambda kv: -kv[1])[:3]
        return total, top

    def gpu_sample(self):
        N, h = self.N, self.h
        util = self._nvml("utilization", lambda: N.nvmlDeviceGetUtilizationRates(h))
        mask = self._nvml("clock_reasons", lambda: N.nvmlDeviceGetCurrentClocksEventReasons(h))
        power = self._nvml("power", lambda: N.nvmlDeviceGetPowerUsage(h))
        return {
            "t": time.time(),
            "util": util.gpu if util is not None else None,
            "sm_clock": self._nvml("sm_clock", lambda: N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM)),
            "mem_clock": self._nvml("mem_clock", lambda: N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM)),
            "temp": self._nvml("temp", lambda: N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)),
            "power_w": power / 1000.0 if power is not None else None,
            "reasons": decode_clock_reasons(mask) if mask is not None else None,
        }

    # --- modes
    def gate(self, window_s, max_util, max_cpu):
        """Idle check while the benchmark is idle. Returns ok + reasons."""
        tree = self.tree()
        self.foreign_cpu_cores()  # prime per-process counters
        t_end = time.time() + window_s
        utils, reasons = [], []
        fg_all, act_all = set(), set()
        while time.time() < t_end:
            s = self.gpu_sample()
            if s["util"] is not None:
                utils.append(s["util"])
            fg, act = self.foreign_gpu(tree)
            fg_all.update(fg); act_all.update(act)
            time.sleep(0.1)
        cpu, top = self.foreign_cpu_cores()
        if fg_all:
            reasons.append("foreign_gpu_process " + ",".join(self._proc_desc(p) for p in sorted(fg_all)))
        if act_all - fg_all:
            reasons.append("foreign_gpu_activity " + ",".join(self._proc_desc(p) for p in sorted(act_all - fg_all)))
        if utils and max(utils) >= max_util:
            reasons.append(f"gpu_util max {max(utils)}% >= {max_util}%")
        if cpu >= max_cpu:
            reasons.append(f"foreign_cpu {cpu:.2f} cores >= {max_cpu} top={top}")
        return {"ok": not reasons, "reasons": reasons, "util_max": max(utils) if utils else None,
                "foreign_cpu_cores": round(cpu, 3), "foreign_cpu_top": top}

    def sample_until_stop(self, conn):
        cfg = self.cfg
        tree = self.tree()
        gpu_dt, cpu_dt = cfg["gpu_sample_ms"] / 1000.0, cfg["cpu_scan_ms"] / 1000.0
        bad_reasons = set(cfg["contaminating_clock_reasons"])
        samples, reasons, cpu_windows = [], [], []
        self.foreign_cpu_cores()
        next_cpu = time.time() + cpu_dt
        while not conn.poll(gpu_dt):
            s = self.gpu_sample()
            samples.append(s)
            fg, act = self.foreign_gpu(tree)
            if fg:
                reasons.append("foreign_gpu_process " + ",".join(self._proc_desc(p) for p in fg))
            if act:
                reasons.append("foreign_gpu_activity " + ",".join(self._proc_desc(p) for p in act))
            for r in s["reasons"] or []:
                if r in bad_reasons:
                    reasons.append(f"clock_reason {r}")
            if time.time() >= next_cpu:
                cpu, top = self.foreign_cpu_cores()
                cpu_windows.append(cpu)
                if cpu > cfg["max_foreign_cpu_cores"]:
                    reasons.append(f"foreign_cpu {cpu:.2f} cores > {cfg['max_foreign_cpu_cores']} top={top}")
                next_cpu = time.time() + cpu_dt
                tree = self.tree()
        conn.recv()  # consume the stop message
        # one final foreign-CPU window so short cells still get a CPU check
        cpu, top = self.foreign_cpu_cores()
        cpu_windows.append(cpu)
        if cpu > cfg["max_foreign_cpu_cores"]:
            reasons.append(f"foreign_cpu {cpu:.2f} cores > {cfg['max_foreign_cpu_cores']} top={top}")
        return summarize(samples, cpu_windows, reasons, self.unsupported)


def summarize(samples, cpu_windows, reasons, unsupported):
    def stats(key):
        v = [s[key] for s in samples if s[key] is not None]
        if not v:
            return None
        return {"min": min(v), "median": statistics.median(v), "max": max(v)}
    seen = sorted({r for s in samples for r in (s["reasons"] or [])})
    temps = [s["temp"] for s in samples if s["temp"] is not None]
    powers = [s["power_w"] for s in samples if s["power_w"] is not None]
    # de-duplicate reasons, keep order
    uniq = list(dict.fromkeys(reasons))
    return {
        "telemetry": {
            "n_samples": len(samples),
            "sm_clock_mhz": stats("sm_clock"),
            "mem_clock_mhz": stats("mem_clock"),
            "gpu_util_pct": stats("util"),
            "temp_c_max": max(temps) if temps else None,
            "power_w_median": statistics.median(powers) if powers else None,
            "throttle_reasons_seen": seen,
            "foreign_cpu_cores_max": round(max(cpu_windows), 3) if cpu_windows else None,
            "nvml_unsupported": sorted(unsupported),
        },
        "contamination_reasons": uniq,
    }


def _monitor_main(conn, main_pid, cfg, cpus, gpu_uuid):
    if cpus:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError:
            pass
    s = _Sampler(main_pid, cfg, gpu_uuid)
    conn.send({"ready": True, "pid": os.getpid()})
    while True:
        msg = conn.recv()
        cmd = msg["cmd"]
        try:
            if cmd == "gate":
                conn.send(s.gate(msg["window_s"], msg["max_util"], msg["max_cpu"]))
            elif cmd == "start":
                conn.send({"started": True})
                conn.send(s.sample_until_stop(conn))
            elif cmd == "quit":
                conn.send({"bye": True})
                return
        except Exception as e:  # never let the monitor die silently
            conn.send({"error": f"{type(e).__name__}: {e}"})


class Monitor:
    """Main-process handle for the monitor process."""

    def __init__(self, mon_cfg, gate_cfg, monitor_cpus=None, gpu_uuid=None):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_monitor_main, args=(child, os.getpid(), mon_cfg, monitor_cpus, gpu_uuid),
                                daemon=True)
        self.proc.start()
        hello = self.conn.recv()
        self.pid = hello["pid"]
        self.gate_cfg = gate_cfg

    def gate(self):
        g = self.gate_cfg
        self.conn.send({"cmd": "gate", "window_s": g["util_window_s"], "max_util": g["max_gpu_util_pct"],
                        "max_cpu": g["max_foreign_cpu_cores"]})
        r = self.conn.recv()
        if "error" in r:
            return {"ok": False, "reasons": [f"monitor_error {r['error']}"]}
        return r

    def start(self):
        self.conn.send({"cmd": "start"})
        self.conn.recv()

    def stop(self):
        self.conn.send({"cmd": "stop"})
        r = self.conn.recv()
        if "error" in r:
            return {"telemetry": {}, "contamination_reasons": [f"monitor_error {r['error']}"]}
        return r

    def close(self):
        try:
            self.conn.send({"cmd": "quit"})
            self.conn.recv()
        except Exception:
            pass
        self.proc.join(timeout=5)


# ----------------------------------------------------------------------------- canary

class Canary:
    """Cold read of rotating buffers on the GPU. On unified-memory machines this is the only signal for
    CPU-side jobs stealing memory bandwidth, which GPU tools cannot see."""

    def __init__(self, cfg, n_bufs=8):
        self.total = int(cfg["total_bytes"])
        self.reps = int(cfg["reps"])
        self.n_bufs = n_bufs
        self.bufs = None

    def _alloc(self):
        import torch
        n = self.total // self.n_bufs // 4
        g = torch.Generator(device="cuda").manual_seed(1234)
        self.bufs = [torch.rand(n, device="cuda", generator=g) for _ in range(self.n_bufs)]
        self.bytes = sum(b.numel() * 4 for b in self.bufs)

    def release(self):
        import torch
        self.bufs = None
        torch.cuda.empty_cache()

    def measure(self):
        """Median GB/s over `reps` cold passes over all buffers."""
        import torch
        if self.bufs is None:
            self._alloc()
        for _ in range(2):
            for b in self.bufs:
                b.sum()
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(self.reps)]
        for s, e in ev:
            s.record()
            for b in self.bufs:
                b.sum()
            e.record()
        torch.cuda.synchronize()
        gbps = [self.bytes / (s.elapsed_time(e) * 1e-3) / 1e9 for s, e in ev]
        return statistics.median(gbps)

    def reference(self, runs):
        vals = [self.measure() for _ in range(runs)]
        med = statistics.median(vals)
        cv = statistics.pstdev(vals) / statistics.mean(vals)
        return {"median_gbps": med, "cv": cv, "values": vals}


# ----------------------------------------------------------------------------- lockfile

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(contact, expected_end, session):
    """Courtesy lockfile for other users; also prevents two of our runners from overlapping."""
    if LOCKFILE.exists():
        try:
            cur = json.loads(LOCKFILE.read_text())
            if cur.get("pid") != os.getpid() and _pid_alive(int(cur.get("pid", -1))):
                raise RuntimeError(f"another paia benchmark is running: {cur}")
        except (ValueError, json.JSONDecodeError):
            pass
    write_lock(contact, expected_end, session)


def write_lock(contact, expected_end, session):
    info = {"user": os.environ.get("PAIA_HOST_USER") or os.environ.get("USER"), "contact": contact,
            "pid": os.getpid(), "session": session,
            "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "expected_end": expected_end,
            "note": "PAIA GPU benchmark in progress. It pauses itself automatically when the GPU or CPU is busy."}
    tmp = LOCKFILE.with_name(f".paia_bench.lock.{os.getpid()}")
    tmp.write_text(json.dumps(info, indent=1) + "\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, LOCKFILE)


def release_lock():
    try:
        cur = json.loads(LOCKFILE.read_text())
        if cur.get("pid") == os.getpid():
            LOCKFILE.unlink()
    except (OSError, ValueError):
        pass
