"""Fits, tables, plots and session comparison.

python -m bench.analyze --session S [--platform-tag T] [--compare S2 [S3 ...]] [--use-contaminated]

Writes results/<tag>/<session>/summary/: summary.md, fits.json, cells.csv, unclean.csv, plots.
Only clean cells enter fits and tables. --use-contaminated is for testing the analysis code on debug
sessions; its output is stamped as not valid.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from bench.common import RESULTS_ROOT, write_json_atomic

MiB = 2 ** 20

# Reference categorical palette (dataviz skill, light mode), fixed order: color follows the entity.
SERIES = {"read/torch_sum": "#2a78d6", "read/triton_read": "#eb6834", "copy/torch_copy_": "#1baf7a",
          "write/torch_fill_": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODES = ("isolated", "eager", "graph")


# ----------------------------------------------------------------------------- loading

def load_cells(sess_dir):
    out = []
    for p in sorted((sess_dir / "cells").glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            pass
    return out


def fill_kernel_names(cells, table_path):
    """Fill cells whose kernel_names is empty from a dispatch table (same GPU model and stack), in memory only.
    Cells that did record names must match the table exactly; a mismatch aborts, since then the table is not valid."""
    t = json.loads(Path(table_path).read_text())["table"]
    filled = checked = 0
    for c in cells:
        if c.get("phase") not in ("e2", "e4") or not c.get("shape"):
            continue
        shape = f"{c['shape']['M']}x{c['shape']['N']}x{c['shape']['K']}"
        names = t.get(f"{c['impl']}|{shape}", t.get(shape) if c["impl"] == "cublas_f_linear" else None)
        if names is None:
            continue
        if c.get("kernel_names"):
            if sorted(c["kernel_names"]) != sorted(names):
                raise SystemExit(f"dispatch table disagrees with recorded kernel names for cell {c['cell_id']}")
            checked += 1
        else:
            c["kernel_names"] = names
            c["kernel_names_source"] = "dispatch_table"
            filled += 1
    print(f"dispatch table {table_path}: filled {filled} cells, matched {checked} recorded cells exactly")
    return {"table": str(table_path), "filled": filled, "matched": checked}


def key(c):
    return f"{c['op']}/{c['impl']}"


def robust_cv(samples):
    """(IQR / 1.349) / median: a spread measure that, unlike CV, is not driven by a few slow tail samples."""
    if not samples or len(samples) < 4:
        return None
    q1, med, q3 = np.percentile(samples, [25, 50, 75])
    return float((q3 - q1) / 1.349 / med) if med > 0 else None


def flat(c):
    b = c.get("bytes") or {}
    t = c.get("telemetry") or {}
    sm = t.get("sm_clock_mhz") or {}
    return {
        "cell_id": c["cell_id"], "session_id": c["session_id"], "phase": c["phase"], "op": c["op"], "impl": c["impl"],
        "precision": c["precision"], "model": c["model"], "layer_name": c["layer_name"],
        "M": c["shape"]["M"], "N": c["shape"]["N"], "K": c["shape"]["K"], "timing_mode": c["timing_mode"],
        "hot_or_cold": c["hot_or_cold"], "R_per_iter": c.get("R_per_iter"), "n_copies": c.get("n_copies"),
        "l2_safe": c.get("l2_safe"), "bytes_weights": b.get("weights"), "bytes_metadata": b.get("metadata"),
        "bytes_act_in": b.get("act_in"), "bytes_out": b.get("out"), "bytes_total": b.get("total"),
        "median_us": c.get("median_us"), "p5_us": c.get("p5_us"), "p95_us": c.get("p95_us"),
        "mean_us": c.get("mean_us"), "std_us": c.get("std_us"), "cv": c.get("cv"), "noisy": c.get("noisy"),
        "n_samples": c.get("n_samples"), "robust_cv": robust_cv(c.get("samples_us")), "achieved_gbps": c.get("achieved_gbps"), "pct_of_b0": c.get("pct_of_b0"),
        "status": c["status"], "retries": c.get("retries"), "canary_before_gbps": c.get("canary_before_gbps"),
        "canary_after_gbps": c.get("canary_after_gbps"), "canary_ref_gbps": c.get("canary_ref_gbps"),
        "sm_clock_min": sm.get("min"), "sm_clock_median": sm.get("median"), "sm_clock_max": sm.get("max"),
        "temp_c_max": t.get("temp_c_max"), "power_w_median": t.get("power_w_median"),
        "kernel_names": ";".join(c.get("kernel_names") or []), "git_commit": c.get("git_commit"),
        "schema_version": c.get("schema_version"), "n_allocations": c.get("n_allocations"),
        "alloc_spread_pct": c.get("alloc_spread_pct"),
        "t_start": c.get("t_start"),
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- fits

def hockney_fit(B, t, n_boot=2000, seed=0):
    """t = tau + B / R. Primary: least squares weighted by 1/t (minimizes relative error, so small ops,
    where tau lives, are not swamped by large ones). Cross-check: Theil-Sen on (B, t).
    95% CIs from a bootstrap over points."""
    from scipy.stats import theilslopes
    B, t = np.asarray(B, float), np.asarray(t, float)

    def wls(Bv, tv):
        A = np.stack([1 / tv, Bv / tv], axis=1)
        (tau, inv_r), *_ = np.linalg.lstsq(A, np.ones_like(tv), rcond=None)
        return tau, inv_r

    tau, inv_r = wls(B, t)
    pred = tau + B * inv_r
    ss_res, ss_tot = float(((t - pred) ** 2).sum()), float(((t - t.mean()) ** 2).sum())
    lres = np.log(t) - np.log(np.clip(pred, 1e-12, None))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(B), len(B))
        if len(set(B[idx])) < 3:
            continue
        bt, bi = wls(B[idx], t[idx])
        if bi > 0:
            boots.append((bt, 1 / bi))
    boots = np.array(boots)
    ts = theilslopes(t, B)
    R = 1 / inv_r  # bytes per us
    out = {
        "n_points": int(len(B)),
        "tau_us": float(tau), "R_inf_gbps": float(R * 1e6 / 1e9), "B_half_bytes": float(tau * R),
        "B_half_mib": float(tau * R / MiB),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else None,
        "rms_rel_residual": float(np.sqrt(np.mean(((t - pred) / t) ** 2))),
        "rms_log_residual": float(np.sqrt(np.mean(lres ** 2))),
        "theil_sen": {"tau_us": float(ts.intercept), "R_inf_gbps": float(1e6 / ts.slope / 1e9) if ts.slope > 0 else None},
        "points": [{"B": float(b), "t_us": float(x), "pred_us": float(p), "rel_residual": float((x - p) / x)}
                   for b, x, p in zip(B, t, pred)],
    }
    if len(boots):
        out["tau_ci95_us"] = [float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))]
        gb = boots[:, 1] * 1e6 / 1e9
        out["R_inf_ci95_gbps"] = [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))]
    return out


# ----------------------------------------------------------------------------- E1

def e1_analysis(cells, l2):
    clean = [c for c in cells if c["phase"] == "e1" and c["status"] == "clean" and c.get("median_us")]
    res = {"plateaus": {}, "fits": {}, "hot": {}, "l2_bytes": l2}
    plateau_min_B = 10 * l2
    for k in sorted({key(c) for c in clean}):
        for mode in MODES:
            cold = [c for c in clean if key(c) == k and c["timing_mode"] == mode and c["hot_or_cold"] == "cold"
                    and c.get("l2_safe")]
            big = [c["achieved_gbps"] for c in cold if c["bytes"]["total"] >= plateau_min_B]
            if big:
                res["plateaus"][f"{k}/{mode}"] = {"median_gbps": float(np.median(big)), "min_gbps": float(min(big)),
                                                   "max_gbps": float(max(big)), "n": len(big),
                                                   "min_bytes": plateau_min_B}
            if len({c["bytes"]["total"] for c in cold}) >= 4:
                cold.sort(key=lambda c: c["bytes"]["total"])
                res["fits"][f"{k}/{mode}"] = hockney_fit([c["bytes"]["total"] for c in cold],
                                                         [c["median_us"] for c in cold])
        hot = sorted([c for c in clean if key(c) == k and c["timing_mode"] == "graph" and c["hot_or_cold"] == "hot"],
                     key=lambda c: c["shape"]["N"])
        coldg = {c["shape"]["N"]: c["achieved_gbps"] for c in clean
                 if key(c) == k and c["timing_mode"] == "graph" and c["hot_or_cold"] == "cold"}
        if hot:
            peak = max(hot, key=lambda c: c["achieved_gbps"])
            fall = next((c["shape"]["N"] for c in hot if c["shape"]["N"] > peak["shape"]["N"]
                         and c["shape"]["N"] in coldg and c["achieved_gbps"] <= 1.1 * coldg[c["shape"]["N"]]), None)
            res["hot"][k] = {"peak_gbps": peak["achieved_gbps"], "peak_buffer_bytes": peak["shape"]["N"],
                             "falloff_buffer_bytes": fall,
                             "note": "buffer size per op (copy touches 2x). falloff = first size above the peak "
                                     "where hot <= 1.1 x cold (graph mode)"}
    for f in res["fits"].values():
        f["fit_ok"] = f["rms_rel_residual"] <= 0.05
    # tau of the probe: the single-kernel Triton read when available. torch.sum changes its launch configuration
    # with input size (and launches a memset), so its fixed cost is not constant and t = tau + B/R misfits.
    for cand in ("read/triton_read/graph", "read/torch_sum/graph"):
        if cand in res["fits"]:
            res["tau_B0"] = {"source": cand, **{k: res["fits"][cand][k] for k in
                             ("tau_us", "tau_ci95_us", "R_inf_gbps", "B_half_mib", "rms_rel_residual", "fit_ok")
                             if k in res["fits"][cand]}}
            break
    reads = {k: v for k, v in res["plateaus"].items() if k.startswith("read/") and k.endswith("/graph")}
    if reads:
        best = max(reads, key=lambda k: reads[k]["median_gbps"])
        res["R_inf_B0_gbps"] = reads[best]["median_gbps"]
        res["R_inf_B0_source"] = best
        res["R_inf_B0_definition"] = (f"median achieved GB/s of clean, l2-safe, cold, graph-mode reads with "
                                      f">= {plateau_min_B / MiB:.0f} MiB (10x L2); best read implementation")
    return res


# ----------------------------------------------------------------------------- canary / telemetry / compare

def canary_summary(sess_dir, cells):
    out = {}
    p = sess_dir / "canary_ref.json"
    if p.exists():
        ref = json.loads(p.read_text())
        out["references"] = [{"t": h.get("t_iso"), "median_gbps": h["median_gbps"], "cv": h["cv"],
                              "drift_vs_prev_pct": h.get("drift_vs_prev_pct")} for h in ref["history"]]
    clean = [c for c in cells if c["status"] == "clean"]
    vals = [v for c in clean for v in (c.get("canary_before_gbps"), c.get("canary_after_gbps")) if v]
    if vals:
        out["per_cell"] = {"n": len(vals), "median_gbps": float(np.median(vals)),
                           "cv": float(np.std(vals) / np.mean(vals)), "min_gbps": float(min(vals)),
                           "max_gbps": float(max(vals))}
    return out


def telemetry_summary(cells):
    clean = [c for c in cells if c["status"] == "clean" and c.get("telemetry")]
    sm = [c["telemetry"]["sm_clock_mhz"]["median"] for c in clean if c["telemetry"].get("sm_clock_mhz")]
    temps = [c["telemetry"]["temp_c_max"] for c in clean if c["telemetry"].get("temp_c_max") is not None]
    reasons = sorted({r for c in clean for r in c["telemetry"].get("throttle_reasons_seen") or []})
    return {"sm_clock_median_mhz": {"min": min(sm), "median": float(np.median(sm)), "max": max(sm)} if sm else None,
            "temp_c_max": max(temps) if temps else None, "clock_reasons_seen": reasons}


def compare_sessions(base, others, threshold=0.03):
    rows = []
    bmap = {c["cell_id"]: c for c in base if c["status"] == "clean"}
    for sid, cells in others.items():
        for c in cells:
            b = bmap.get(c["cell_id"])
            if c["status"] != "clean" or not b:
                continue
            d = (c["median_us"] - b["median_us"]) / b["median_us"]
            rows.append({"session": sid, "cell_id": c["cell_id"], "op": key(c), "mode": c["timing_mode"],
                         "hot_or_cold": c["hot_or_cold"], "bytes": c["bytes"]["total"], "base_median_us": b["median_us"],
                         "median_us": c["median_us"], "diff_pct": 100 * d, "flag": abs(d) > threshold})
    return rows


# ----------------------------------------------------------------------------- plots

def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ax.spines.values():
        s.set_visible(False)


def _size_axis(ax):
    ax.set_xscale("log", base=2)
    ticks = [2 ** p for p in range(18, 33, 2)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / MiB:g} MiB" if t < 2 ** 30 else f"{t / 2**30:g} GiB" for t in ticks], rotation=0)


def plot_e1(cells, e1, out_dir, l2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clean = [c for c in cells if c["phase"] == "e1" and c["status"] == "clean" and c.get("median_us")]
    paths = {}
    # 1. cold bandwidth vs size, small multiples per timing mode
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, facecolor=SURFACE)
    for ax, mode in zip(axes, MODES):
        for k, col in SERIES.items():
            pts = sorted((c["shape"]["N"], c["achieved_gbps"]) for c in clean
                         if key(c) == k and c["timing_mode"] == mode and c["hot_or_cold"] == "cold")
            if pts:
                ax.plot(*zip(*pts), color=col, lw=2, marker="o", ms=4, label=k)
        if "R_inf_B0_gbps" in e1:
            ax.axhline(e1["R_inf_B0_gbps"], color=INK2, lw=1, ls=":")
        _style(ax, f"Cold, {mode} mode", "buffer size", "achieved GB/s" if mode == "isolated" else "")
        _size_axis(ax)
    if "R_inf_B0_gbps" in e1:
        axes[-1].annotate(f"R_inf_B0 = {e1['R_inf_B0_gbps']:.1f} GB/s", (2 ** 18, e1["R_inf_B0_gbps"]),
                          xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    paths["e1_cold_bandwidth"] = out_dir / "e1_cold_bandwidth.png"
    fig.savefig(paths["e1_cold_bandwidth"], dpi=140, facecolor=SURFACE)
    plt.close(fig)

    # 2. hot vs cold (graph mode); hot dashed with open markers so the pair is not color-alone
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    for k, col in SERIES.items():
        for hc, ls, mfc in (("cold", "-", col), ("hot", "--", SURFACE)):
            pts = sorted((c["shape"]["N"], c["achieved_gbps"]) for c in clean
                         if key(c) == k and c["timing_mode"] == "graph" and c["hot_or_cold"] == hc)
            if pts:
                ax.plot(*zip(*pts), color=col, lw=2, ls=ls, marker="o", ms=5, mfc=mfc, label=f"{k} {hc}")
    ax.axvline(l2, color=INK2, lw=1, ls=":")
    ax.annotate(f"L2 = {l2 / MiB:g} MiB", (l2, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                color=INK2, fontsize=8)
    _style(ax, "Hot (one buffer reused) vs cold (L2-rotated), graph mode", "buffer size", "achieved GB/s")
    _size_axis(ax)
    ax.legend(frameon=False, fontsize=7, ncol=2, labelcolor=INK)
    fig.tight_layout()
    paths["e1_hot_vs_cold"] = out_dir / "e1_hot_vs_cold.png"
    fig.savefig(paths["e1_hot_vs_cold"], dpi=140, facecolor=SURFACE)
    plt.close(fig)

    # 3. fit and relative residuals for cold graph-mode reads
    reads = [k for k in ("read/torch_sum", "read/triton_read") if f"{k}/graph" in e1["fits"]]
    if reads:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
        for k in reads:
            f = e1["fits"][f"{k}/graph"]
            B = np.array([p["B"] for p in f["points"]])
            t = np.array([p["t_us"] for p in f["points"]])
            a1.plot(B, t, "o", color=SERIES[k], ms=5, label=f"{k} measured")
            xs = np.geomspace(B.min(), B.max(), 200)
            # R in GB/s is R * 1e3 bytes per us
            a1.plot(xs, f["tau_us"] + xs / (f["R_inf_gbps"] * 1e3), color=SERIES[k], lw=1.5,
                    label=f"{k} fit: tau {f['tau_us']:.2f} us, R {f['R_inf_gbps']:.1f} GB/s")
            a2.plot(B, 100 * np.array([p["rel_residual"] for p in f["points"]]), "o-", color=SERIES[k], lw=1.5, ms=5,
                    label=k)
        a1.set_yscale("log")
        _style(a1, "Cold graph-mode read: t = tau + B / R", "bytes per op", "median time per op (us)")
        _size_axis(a1)
        a1.legend(frameon=False, fontsize=7, labelcolor=INK)
        a2.axhline(0, color=INK2, lw=1)
        _style(a2, "Fit residuals", "bytes per op", "(measured - fit) / measured  (%)")
        _size_axis(a2)
        a2.legend(frameon=False, fontsize=7, labelcolor=INK)
        fig.tight_layout()
        paths["e1_fit"] = out_dir / "e1_fit_residuals.png"
        fig.savefig(paths["e1_fit"], dpi=140, facecolor=SURFACE)
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- report

def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def summary_md(sess, cells, e1, canary, tel, compare_rows, plot_paths, use_contaminated):
    st = {}
    for c in cells:
        st[c["status"]] = st.get(c["status"], 0) + 1
    L = [f"# Summary: {sess}", ""]
    if use_contaminated:
        L += ["> **NOT VALID: built with --use-contaminated for testing the analysis code. Do not use these numbers.**", ""]
    L += [f"Cells: {len(cells)} total; " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())), ""]
    sp = [c["alloc_spread_pct"] for c in cells if c["status"] == "clean" and c.get("alloc_spread_pct") is not None]
    if sp:
        L += [f"Schema v2: {cells[0].get('n_allocations')} allocations per cell; spread of per-allocation medians "
              f"(max - min) / median: median {np.median(sp):.2f}%, p90 {np.percentile(sp, 90):.2f}%, max {max(sp):.2f}%.", ""]
    noisy = sum(1 for c in cells if c["status"] == "clean" and c.get("noisy"))
    rob = [robust_cv(c.get("samples_us")) for c in cells if c["status"] == "clean"]
    rob_noisy = sum(1 for r in rob if r is not None and r > 0.03)
    L += [f"Clean cells flagged noisy (CV > 3%): {noisy}. With the tail-insensitive robust CV (IQR/1.349/median): "
          f"{rob_noisy} above 3%, median {100 * float(np.median([r for r in rob if r is not None])):.2f}%.", ""]
    if e1:
        L += ["## E1 (B0) achievable bandwidth", ""]
        if "R_inf_B0_gbps" in e1:
            L += [f"**R_inf_B0 = {e1['R_inf_B0_gbps']:.2f} GB/s** ({e1['R_inf_B0_source']}; {e1['R_inf_B0_definition']}).", ""]
        if "tau_B0" in e1:
            t = e1["tau_B0"]; ci = t.get("tau_ci95_us")
            L += [f"**Probe tau = {t['tau_us']:.2f} us** (95% CI {fmt(ci[0]) if ci else '-'} to {fmt(ci[1]) if ci else '-'}), "
                  f"fitted R {t['R_inf_gbps']:.1f} GB/s, B_half {t['B_half_mib']:.3f} MiB, rms residual "
                  f"{100 * t['rms_rel_residual']:.1f}% ({t['source']}).", ""]
        L += ["### Cold plateaus (>= 10x L2)", "", "| op/impl/mode | median GB/s | min | max | n |", "|---|---|---|---|---|"]
        for k, v in sorted(e1["plateaus"].items()):
            L.append(f"| {k} | {v['median_gbps']:.2f} | {v['min_gbps']:.2f} | {v['max_gbps']:.2f} | {v['n']} |")
        L += ["", "### Fits t = tau + B / R (cold, l2-safe; weighted LS, Theil-Sen cross-check, bootstrap 95% CI)", "",
              "Fits with rms relative residual above 5% are marked `misfit`: the single-tau model does not describe that op.", "",
              "| op/impl/mode | tau us | tau CI95 | R_inf GB/s | R CI95 | B_half MiB | R^2 | rms rel resid | Theil-Sen tau | Theil-Sen R |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, f in sorted(e1["fits"].items()):
            ci = f.get("tau_ci95_us"); rci = f.get("R_inf_ci95_gbps")
            L.append(f"| {k} | {f['tau_us']:.2f} | {fmt(ci[0]) + ' to ' + fmt(ci[1]) if ci else '-'} | {f['R_inf_gbps']:.1f} | "
                     f"{fmt(rci[0], 1) + ' to ' + fmt(rci[1], 1) if rci else '-'} | {f['B_half_mib']:.3f} | {fmt(f['r2'], 5)} | "
                     f"{100 * f['rms_rel_residual']:.1f}%{'' if f['fit_ok'] else ' misfit'} | {fmt(f['theil_sen']['tau_us'])} | {fmt(f['theil_sen']['R_inf_gbps'], 1)} |")
        L += ["", "### Hot (L2) behaviour, graph mode", "", "| op/impl | peak GB/s | at buffer size | falls off at |", "|---|---|---|---|"]
        for k, v in sorted(e1["hot"].items()):
            fb = v["falloff_buffer_bytes"]
            L.append(f"| {k} | {v['peak_gbps']:.1f} | {v['peak_buffer_bytes'] / MiB:g} MiB | "
                     f"{f'{fb / MiB:g} MiB' if fb else 'not reached'} |")
        L.append("")
    L += ["## Canary", ""]
    for r in canary.get("references", []):
        L.append(f"- reference {r['t']}: {r['median_gbps']:.2f} GB/s, CV {100 * r['cv']:.2f}%"
                 + (f", drift {r['drift_vs_prev_pct']:+.2f}%" if r.get("drift_vs_prev_pct") is not None else ""))
    if "per_cell" in canary:
        p = canary["per_cell"]
        L.append(f"- per-cell canary over clean cells: median {p['median_gbps']:.2f} GB/s, CV {100 * p['cv']:.2f}%, "
                 f"range {p['min_gbps']:.2f} to {p['max_gbps']:.2f} (n={p['n']})")
    L += ["", "## Telemetry (clean cells)", "", f"- SM clock median per cell: {tel['sm_clock_median_mhz']}",
          f"- max temperature: {tel['temp_c_max']} C", f"- clock-event reasons seen: {tel['clock_reasons_seen'] or 'none'}", ""]
    if compare_rows:
        fl = [r for r in compare_rows if r["flag"]]
        d = np.abs([r["diff_pct"] for r in compare_rows])
        L += ["## Cross-session comparison", "",
              f"{len(compare_rows)} matching clean cells; |diff| median {np.median(d):.2f}%, max {d.max():.2f}%; "
              f"{len(fl)} differ by more than 3% (listed in compare.csv).", ""]
    L += ["## Plots", ""] + [f"- [{k}]({p.name})" for k, p in plot_paths.items()]
    L += ["", "Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted)."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--platform-tag", default="dgxspark_gb10")
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--use-contaminated", action="store_true")
    ap.add_argument("--e1-session", default=None, help="E1 session providing B0 and the same-size reference (E2+)")
    ap.add_argument("--bf16-sessions", nargs="*", default=[], help="E2 sessions providing the BF16 reference (E4)")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--dispatch-table", default=None,
                    help="scripts/diag/dispatch_table.py output: fills empty kernel_names in memory (cell files unchanged)")
    a = ap.parse_args()
    root = Path(a.results_root) if a.results_root else RESULTS_ROOT
    sess_dir = root / a.platform_tag / a.session
    cells = load_cells(sess_dir)
    dispatch = fill_kernel_names(cells, a.dispatch_table) if a.dispatch_table else None
    if a.use_contaminated:
        for c in cells:
            if c["status"] == "contaminated" and c.get("median_us"):
                c["status"] = "clean"
    env = json.loads((sess_dir / "env.json").read_text()) if (sess_dir / "env.json").exists() else {}
    l2 = (env.get("gpu") or {}).get("L2_cache_size") or next((c["l2_bytes"] for c in cells if c.get("l2_bytes")), None)
    out = sess_dir / "summary"
    out.mkdir(exist_ok=True)
    e1 = e1_analysis(cells, l2) if any(c["phase"] == "e1" for c in cells) else None
    b0 = e1.get("R_inf_B0_gbps") if e1 else None
    e1_cells = []
    if a.e1_session:
        e1_dir = root / a.platform_tag / a.e1_session
        e1_cells = load_cells(e1_dir)
        if b0 is None and (e1_dir / "summary" / "b0.json").exists():
            b0 = json.loads((e1_dir / "summary" / "b0.json").read_text())["R_inf_B0_gbps"]
    if b0 and e1:
        write_json_atomic(out / "b0.json", {"R_inf_B0_gbps": b0, "source": e1["R_inf_B0_source"],
                                            "definition": e1["R_inf_B0_definition"], "session": a.session})
    rows = []
    for c in cells:
        r = flat(c)
        if b0 and r["achieved_gbps"]:
            r["pct_of_b0"] = 100 * r["achieved_gbps"] / b0
        rows.append(r)
    write_csv(out / "cells.csv", [r for r in rows if r["status"] == "clean"])
    write_csv(out / "unclean.csv", [{**{k: r[k] for k in ("cell_id", "op", "impl", "N", "timing_mode", "hot_or_cold",
                                                           "status", "retries")},
                                     "reasons": " | ".join(c.get("contamination_reasons") or []), "error": c.get("error")}
                                    for r, c in zip(rows, cells) if r["status"] != "clean"])
    others = {s: load_cells(root / a.platform_tag / s) for s in a.compare}
    cmp_rows = compare_sessions(cells, others)
    write_csv(out / "compare.csv", cmp_rows)
    canary = canary_summary(sess_dir, cells)
    tel = telemetry_summary(cells)
    plots = plot_e1(cells, e1, out, l2) if e1 else {}
    e2, e2_md = None, []
    if any(c["phase"] == "e2" for c in cells):
        from bench import analyze_e2
        e2, e2_md, p2 = analyze_e2.analyze(cells, e1_cells, b0, out)
        plots.update(p2)
    e4, e4_md = None, []
    if any(c["phase"] == "e4" for c in cells):
        from bench import analyze_e4
        bf16_cells = [c for s in a.bf16_sessions for c in load_cells(root / a.platform_tag / s)]
        if a.dispatch_table:
            fill_kernel_names(bf16_cells, a.dispatch_table)
        e4, e4_md, p4 = analyze_e4.analyze(cells, bf16_cells, b0, out)
        plots.update(p4)
    fits = {"session": a.session, "valid": not a.use_contaminated, "e1": e1, "e2": e2, "e4": e4, "b0_gbps": b0,
            "e1_reference_session": a.e1_session, "bf16_reference_sessions": a.bf16_sessions, "canary": canary,
            "telemetry": tel, "dispatch_table": dispatch}
    write_json_atomic(out / "fits.json", fits)
    text = summary_md(a.session, cells, e1, canary, tel, cmp_rows, plots, a.use_contaminated)
    if e2_md or e4_md:
        text = text.replace("## Canary", "\n".join(e2_md + e4_md) + "\n## Canary", 1)
    if dispatch:
        text = text.replace("\n## Canary", f"\nKernel names: {dispatch['filled']} cells filled from `{Path(dispatch['table']).name}` "
                            f"(profiler capture lost on this stack, findings.md); {dispatch['matched']} recorded cells matched it "
                            "exactly.\n\n## Canary", 1)
    (out / "summary.md").write_text(text)
    print(f"wrote {out}/summary.md")


if __name__ == "__main__":
    main()
