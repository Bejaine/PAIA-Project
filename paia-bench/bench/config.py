"""Load frozen configs and per-platform notes."""
from functools import lru_cache

import yaml

from bench.common import BENCH_ROOT


@lru_cache(maxsize=None)
def shapes_cfg():
    return yaml.safe_load(open(BENCH_ROOT / "configs" / "shapes.yaml"))


@lru_cache(maxsize=None)
def platform_notes(platform_tag):
    notes = yaml.safe_load(open(BENCH_ROOT / "configs" / "platform_notes.yaml"))
    merged = dict(notes.get("defaults", {}))
    for k, v in (notes.get(platform_tag) or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def timing_cfg():
    return shapes_cfg()["timing"]


def canary_cfg():
    return shapes_cfg()["canary"]


def gate_cfg():
    return shapes_cfg()["idle_gate"]


def monitor_cfg():
    return shapes_cfg()["monitor"]
