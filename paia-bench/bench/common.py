"""Shared helpers: paths, platform tag, provenance, atomic JSON writes."""
import json
import os
import re
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = BENCH_ROOT / "results"
SCHEMA_VERSION = 2  # v2 (2026-10-19): each cell measured over several independent allocations


def default_platform_tag():
    import torch
    name = torch.cuda.get_device_name(0)
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("nvidia", "")).strip("_")


def gpu_uuid():
    """NVML-style UUID ('GPU-...') of CUDA device 0, i.e. the GPU this process benchmarks after
    CUDA_VISIBLE_DEVICES is applied. NVML ignores CUDA_VISIBLE_DEVICES, so this is how the monitor finds it."""
    import torch
    u = str(torch.cuda.get_device_properties(0).uuid)
    return u if u.startswith(("GPU-", "MIG-")) else f"GPU-{u}"


def nvml_handle(N, uuid=None):
    """NVML handle for the benchmarked GPU. Index 0 is only safe when NVML sees a single device
    (GB10, or a container started with --gpus device=X); on multi-GPU hosts it may be someone else's GPU."""
    if uuid:
        return N.nvmlDeviceGetHandleByUUID(uuid)
    if N.nvmlDeviceGetCount() == 1:
        return N.nvmlDeviceGetHandleByIndex(0)
    raise RuntimeError("NVML sees several GPUs and no CUDA device UUID was given; cannot tell which GPU to monitor")


def session_dir(platform_tag, session_id):
    d = RESULTS_ROOT / platform_tag / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def provenance():
    return {
        "git_commit": os.environ.get("PAIA_GIT_COMMIT", "unknown"),
        "git_dirty": os.environ.get("PAIA_GIT_DIRTY", "unknown") == "1",
        "image": os.environ.get("PAIA_IMAGE"),
        "image_id": os.environ.get("PAIA_IMAGE_ID"),
    }


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
