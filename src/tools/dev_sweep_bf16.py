"""Developer sweep of the S1 kernel variants on one GPU, using the harness's own timing code (bench.timing).
DEV ONLY: no idle gate and no canary, so these numbers are for choosing variants, never results. Results of record
come from `bench.run --phase s1` under P2's gate. Refuses to run while /tmp/paia_bench.lock exists (P2's courtesy lock:
a measurement session is in progress and this would contaminate it) unless --ignore-lock.

    scripts/drun.sh python /src/tools/dev_sweep_bf16.py --platform-tag dgxspark_gb10 --shapes llama31_8b:q_proj qwen25_7b:k_proj
    ... --all-real                     # every real shape (about 20; 96 variants each: budget ~30 min)
    ... --variants 'R=2,U>=4,H=1'      # filter variants
Writes src/tuned/<platform>.json (best variant per N x K, plus top 5 with their % of the best) when --write-tuned.
"""
import argparse, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
LOCK = Path("/tmp/paia_bench.lock")


def parse_filter(expr):
    conds = []
    for part in filter(None, (expr or "").split(",")):
        m = re.fullmatch(r"([RUWSH])(>=|<=|=)(\d+)", part.strip())
        if not m:
            raise SystemExit(f"bad filter term {part!r}")
        conds.append((m.group(1), m.group(2), int(m.group(3))))
    ops = {"=": lambda a, b: a == b, ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b}
    return lambda v: all(ops[o](v[k], n) for k, o, n in conds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform-tag", required=True)
    ap.add_argument("--kind", choices=["bf16", "fp8", "int4"], default="bf16", help="which kernel family to sweep")
    ap.add_argument("--shapes", nargs="*", default=[], help="model:layer, e.g. llama31_8b:q_proj")
    ap.add_argument("--all-real", action="store_true")
    ap.add_argument("--variants", default="")
    ap.add_argument("--write-tuned", action="store_true")
    ap.add_argument("--ignore-lock", action="store_true")
    a = ap.parse_args()
    if LOCK.exists() and not a.ignore_lock:
        sys.exit(f"{LOCK} exists: a gated measurement session is running ({LOCK.read_text()[:200]!r}). Not running.")

    import torch
    from bench import config, timing
    from bench.ops.ours_bf16 import OursBF16Gemv
    from bench.ops.ours_quant import OursFP8Gemv, OursInt4Gemv
    from paia_gemv import load_lib

    lib, cfg, tcfg = load_lib(), config.shapes_cfg(), config.timing_cfg()
    OpCls = {"bf16": OursBF16Gemv, "fp8": OursFP8Gemv, "int4": OursInt4Gemv}[a.kind]
    variants = {"bf16": lib.variants, "fp8": lib.variants_fp8, "int4": lib.variants_int4}[a.kind]
    kmult = {"bf16": 8, "fp8": 16, "int4": 128}[a.kind]
    tkey = {"bf16": "shapes", "fp8": "fp8_shapes", "int4": "int4_shapes"}[a.kind]
    props = torch.cuda.get_device_properties(0)
    mem_cap = int(tcfg["mem_cap_gb"] * 2**30)
    keep = parse_filter(a.variants)
    targets = []
    for model, m in cfg["models"].items():
        for layer, (n, k) in m["shapes"].items():
            if (a.all_real or f"{model}:{layer}" in a.shapes) and k % kmult == 0:
                targets.append((model, layer, n, k))
    seen, uniq = set(), []
    for t in targets:                      # identical N x K share one tuned entry
        if (t[2], t[3]) not in seen:
            seen.add((t[2], t[3])); uniq.append(t)
    out_f = HERE.parent / "tuned" / f"{a.platform_tag}.json"
    tuned = json.load(open(out_f)) if out_f.exists() else {"platform": a.platform_tag, "shapes": {}}
    tuned.setdefault(tkey, {})
    atcfg = {**tcfg, "sample_target_ms": 400, "warmup_min_ms": 100}
    for model, layer, N, K in uniq:
        rows = []
        for v in variants:
            if not keep(v):
                continue
            op = OpCls(M=1, N=N, K=K, model=model, layer_name=layer, variant=v["id"])
            rot = timing.plan_rotation(op.footprint_bytes(), False, mem_cap, props.L2_cache_size, tcfg["l2_rotation_factor"])
            try:
                op.build(rot["n_copies"])
                ok, detail = op.check()
                if not ok:
                    print(f"  variant {v} INCORRECT {detail}"); continue
                m = timing.measure(op, "graph", rot["n_copies"], atcfg)
                rows.append((m["median_us"], v, rot["l2_safe"]))
            finally:
                op.free()
        rows.sort(key=lambda r: r[0])
        tb = sum(op.bytes().values())
        print(f"\n{model}:{layer} N={N} K={K}  ({tb/2**20:.1f} MiB)  l2_safe={rows[0][2] if rows else None}")
        for us, v, _ in rows[:5]:
            print(f"  {tb/(us*1e-6)/1e9:7.1f} GB/s  {us:9.2f} us   R={v['R']} U={v['U']} W={v['W']} S={v['S']} H={v['H']}")
        if rows:
            tuned[tkey][f"{N}x{K}"] = {"variant": {k: rows[0][1][k] for k in "RUWSH"}, "us": rows[0][0],
                                            "gbps": tb / (rows[0][0] * 1e-6) / 1e9,
                                            "top5": [{"variant": {k: v[k] for k in "RUWSH"}, "pct_of_best": 100 * rows[0][0] / us}
                                                     for us, v, _ in rows[:5]],
                                            "dev_only": True, "when": time.strftime("%Y-%m-%d %H:%M")}
    if a.write_tuned:
        out_f.parent.mkdir(parents=True, exist_ok=True)
        json.dump(tuned, open(out_f, "w"), indent=1)
        print(f"\nwrote {out_f} (dev sweep; re-validate under the gated harness before relying on it)")


if __name__ == "__main__":
    main()
