"""E0 fingerprint: what exactly is this machine and software stack.

python -m bench.env --session S [--platform-tag T] [--probe-ncu]
Writes results/<platform_tag>/<session>/env.json. Every hardware property is read at runtime.
"""
import argparse
import ctypes
import datetime as dt
import os
import platform
import shutil
import subprocess
import sys

from bench.common import SCHEMA_VERSION, default_platform_tag, provenance, session_dir, write_json_atomic


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"rc": r.returncode, "out": r.stdout.strip(), "err": r.stderr.strip()[-2000:]}
    except Exception as e:
        return {"rc": None, "out": "", "err": f"{type(e).__name__}: {e}"}


def try_(fn):
    try:
        return fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def gpu_props():
    import torch
    p = torch.cuda.get_device_properties(0)
    keys = ["name", "major", "minor", "multi_processor_count", "total_memory", "L2_cache_size", "warp_size",
            "max_threads_per_multi_processor", "regs_per_multiprocessor", "shared_memory_per_block_optin", "uuid"]
    d = {k: getattr(p, k, None) for k in keys}
    d["compute_capability"] = f"{p.major}.{p.minor}"
    d["torch_arch_list"] = torch.cuda.get_arch_list()
    return d


def cuda_versions():
    import torch
    out = {"torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    rt = None
    for cand in ["libcudart.so.13", "libcudart.so", "libcudart.so.12"]:
        try:
            rt = ctypes.CDLL(cand); break
        except OSError:
            pass
    if rt is not None:
        v = ctypes.c_int()
        rt.cudaRuntimeGetVersion(ctypes.byref(v)); out["cudart_runtime"] = v.value
        rt.cudaDriverGetVersion(ctypes.byref(v)); out["cuda_driver_api"] = v.value
    out["nvcc"] = sh("nvcc --version | tail -2")["out"] or None
    return out


def cublas_version():
    import torch
    a = torch.zeros(8, 8, device="cuda")
    (a @ a).sum().item()  # force libcublas to load
    libs = sorted({l for l in open("/proc/self/maps").read().split() if "libcublas.so" in l and "Lt" not in l})
    lib = ctypes.CDLL(libs[0])
    parts = []
    for i in range(3):
        v = ctypes.c_int()
        lib.cublasGetProperty(i, ctypes.byref(v)); parts.append(v.value)
    return {"version": ".".join(map(str, parts)), "path": libs[0]}


def py_packages():
    import importlib
    out = {"python": sys.version}
    for m in ["torch", "triton", "vllm", "transformers", "numpy", "scipy", "pandas", "matplotlib", "yaml",
              "psutil", "pynvml", "nvtx"]:
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "present")
        except Exception as e:
            out[m] = None
    import torch
    out["preferred_blas_library"] = try_(lambda: str(torch.backends.cuda.preferred_blas_library()))
    return out


def nvml_fields():
    """Which NVML queries work here. The monitor records null for the ones that do not."""
    import pynvml as N
    from bench.common import gpu_uuid, nvml_handle
    N.nvmlInit()
    h = nvml_handle(N, gpu_uuid())
    q = {
        "device_count_visible_to_nvml": lambda: N.nvmlDeviceGetCount(),
        "device_index": lambda: N.nvmlDeviceGetIndex(h),
        "device_name": lambda: N.nvmlDeviceGetName(h),
        "device_uuid": lambda: N.nvmlDeviceGetUUID(h),
        "pci_bus_id": lambda: N.nvmlDeviceGetPciInfo(h).busId,
        "ecc_mode_current_pending": lambda: list(N.nvmlDeviceGetEccMode(h)),
        "memory_bus_width_bits": lambda: N.nvmlDeviceGetMemoryBusWidth(h),
        "mig_mode_current_pending": lambda: list(N.nvmlDeviceGetMigMode(h)),
        "driver_version": lambda: N.nvmlSystemGetDriverVersion(),
        "cuda_driver_version": lambda: N.nvmlSystemGetCudaDriverVersion_v2(),
        "utilization": lambda: vars_of(N.nvmlDeviceGetUtilizationRates(h)),
        "compute_procs": lambda: [p.pid for p in N.nvmlDeviceGetComputeRunningProcesses(h)],
        "process_utilization": lambda: len(N.nvmlDeviceGetProcessUtilization(h, 0)),
        "sm_clock_mhz": lambda: N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
        "mem_clock_mhz": lambda: N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM),
        "max_sm_clock_mhz": lambda: N.nvmlDeviceGetMaxClockInfo(h, N.NVML_CLOCK_SM),
        "max_mem_clock_mhz": lambda: N.nvmlDeviceGetMaxClockInfo(h, N.NVML_CLOCK_MEM),
        "applications_clock_sm_mhz": lambda: N.nvmlDeviceGetApplicationsClock(h, N.NVML_CLOCK_SM),
        "temperature_c": lambda: N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU),
        "power_mw": lambda: N.nvmlDeviceGetPowerUsage(h),
        "power_limit_mw": lambda: N.nvmlDeviceGetEnforcedPowerLimit(h),
        "clock_event_reasons": lambda: N.nvmlDeviceGetCurrentClocksEventReasons(h),
        "pstate": lambda: N.nvmlDeviceGetPerformanceState(h),
        "memory_info": lambda: vars_of(N.nvmlDeviceGetMemoryInfo(h)),
        "persistence_mode": lambda: N.nvmlDeviceGetPersistenceMode(h),
        "compute_mode": lambda: N.nvmlDeviceGetComputeMode(h),
    }
    out = {}
    for k, fn in q.items():
        try:
            out[k] = {"supported": True, "sample": fn()}
        except Exception as e:
            out[k] = {"supported": False, "error": f"{type(e).__name__}: {e}"}
    return out


def vars_of(s):
    return {f[0]: getattr(s, f[0]) for f in getattr(s, "_fields_", [])}


def perf_counter_access(probe_ncu):
    out = {"RmProfilingAdminOnly": None, "uid": os.getuid()}
    try:
        for line in open("/proc/driver/nvidia/params"):
            if line.startswith("RmProfilingAdminOnly"):
                out["RmProfilingAdminOnly"] = int(line.split(":")[1])
    except OSError as e:
        out["params_error"] = str(e)
    if probe_ncu:
        code = ("import torch; a=torch.empty(1<<20,device='cuda'); a.fill_(1); torch.cuda.synchronize()")
        # Only a metric that exists on every arch: this probe tests counter permission, not metric names.
        r = sh(f"ncu --clock-control none -c 1 --metrics gpu__time_duration.sum "
               f"python -c \"{code}\"", timeout=300)
        out["ncu_probe_rc"] = r["rc"]
        out["ncu_probe_ok"] = r["rc"] == 0 and "gpu__time_duration.sum" in r["out"]
        out["ncu_probe_tail"] = "\n".join((r["out"] + "\n" + r["err"]).splitlines()[-8:])
    return out


def cpu_info():
    d = {"machine": platform.machine(), "logical_cpus": os.cpu_count()}
    lscpu = sh("lscpu")["out"]
    d["model_names"] = sorted({l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")})
    d["lscpu"] = lscpu
    try:
        d["mem_total_kb"] = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
    except Exception:
        pass
    return d


def collect(tag, session, probe_ncu=False):
    env = {
        "schema_version": SCHEMA_VERSION,
        "platform_tag": tag,
        "session_id": session,
        "t_captured": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.environ.get("PAIA_HOST", platform.node()),
        "host_user": os.environ.get("PAIA_HOST_USER"),
        "provenance": provenance(),
        "gpu": try_(gpu_props),
        "cuda": try_(cuda_versions),
        "cublas": try_(cublas_version),
        "packages": try_(py_packages),
        "nvml": try_(nvml_fields),
        "perf_counters": try_(lambda: perf_counter_access(probe_ncu)),
        "tools": {
            "ncu": sh("ncu --version | tail -1")["out"] or None,
            "nsys": sh("nsys --version")["out"] or None,
            "ncu_path": shutil.which("ncu"),
            "nsys_path": shutil.which("nsys"),
        },
        "clock_lock": {
            "attempted": False,
            "reason": "Locking clocks (nvidia-smi -lgc/-lmc) needs root and changes clocks for every user of this shared "
                      "machine; policy forbids system-wide changes. Clocks are recorded per cell instead.",
            "supported_clocks_query": sh("nvidia-smi -q -d SUPPORTED_CLOCKS")["out"][-3000:],
        },
        "power_and_perf": sh("nvidia-smi -q -d POWER,PERFORMANCE,CLOCK")["out"],
        "nvidia_smi_q": sh("nvidia-smi -q")["out"],
        "uname": sh("uname -a")["out"],
        "os_release": try_(lambda: open("/etc/os-release").read()),
        "cpu": try_(cpu_info),
        "pip_freeze": sh("pip freeze 2>/dev/null")["out"].splitlines(),
        "container": {"image": os.environ.get("PAIA_IMAGE"), "image_id": os.environ.get("PAIA_IMAGE_ID"),
                      "nvidia_build": os.environ.get("NVIDIA_BUILD_ID"),
                      "ngc_release": os.environ.get("NVIDIA_PYTORCH_VERSION") or os.environ.get("NVIDIA_VLLM_VERSION")},
    }
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--platform-tag", default=None)
    ap.add_argument("--probe-ncu", action="store_true", help="run a 1-kernel ncu probe to test counter access")
    a = ap.parse_args()
    tag = a.platform_tag or default_platform_tag()
    env = collect(tag, a.session, a.probe_ncu)
    out = session_dir(tag, a.session) / "env.json"
    write_json_atomic(out, env)
    g, p = env["gpu"], env["packages"]
    print(f"wrote {out}")
    print(f"GPU {g.get('name')} cc {g.get('compute_capability')} SMs {g.get('multi_processor_count')} "
          f"L2 {g.get('L2_cache_size')} mem {g.get('total_memory')}")
    print(f"torch {p.get('torch')} cuda {env['cuda'].get('torch_cuda')} cublas {env['cublas'].get('version')} "
          f"triton {p.get('triton')} vllm {p.get('vllm')}")
    print("NVML unsupported:", [k for k, v in env["nvml"].items() if isinstance(v, dict) and not v.get("supported")])
    print("perf counters:", {k: v for k, v in env["perf_counters"].items() if k != "ncu_probe_tail"})


if __name__ == "__main__":
    main()
