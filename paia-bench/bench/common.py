"""Shared helpers: paths, platform tag, provenance, atomic JSON writes."""
import json
import os
import re
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = BENCH_ROOT / "results"
SCHEMA_VERSION = 1


def default_platform_tag():
    import torch
    name = torch.cuda.get_device_name(0)
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("nvidia", "")).strip("_")


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
