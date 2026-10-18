"""E2 (B1) analysis: cuBLAS BF16 GEMV per shape, Hockney fits per K and mode, launch-cost split, efficiency
curves, predicted efficiency at BF16 and 4-bit sizes, M sweep. Called from bench.analyze."""
import numpy as np

from bench.analyze import INK, INK2, MiB, MODES, SURFACE, _size_axis, _style, hockney_fit

# Fixed categorical order (reference palette): color follows the entity across charts.
COL = {"K4096": "#2a78d6", "K14336": "#eb6834", "llama31_8b": "#1baf7a", "qwen25_7b": "#eda100"}
MODE_COL = {"isolated": "#2a78d6", "eager": "#eb6834", "graph": "#1baf7a"}


def e1_reference(e1_cells):
    """Cold, l2-safe Triton read curve per mode: (bytes, GB/s) sorted. Used for same-size comparisons."""
    ref = {}
    for mode in MODES:
        pts = sorted((c["bytes"]["total"], c["achieved_gbps"]) for c in e1_cells
                     if c["status"] == "clean" and c["impl"] == "triton_read" and c["hot_or_cold"] == "cold"
                     and c["timing_mode"] == mode and c.get("l2_safe"))
        if pts:
            ref[mode] = pts
    return ref


def interp_gbps(ref_pts, B):
    xs = np.log([p[0] for p in ref_pts])
    ys = [p[1] for p in ref_pts]
    return float(np.interp(np.log(B), xs, ys))


def kernel_label(names):
    """Short, stable label of the dispatched kernel set, e.g. 'gemvx/v6', 'gemvx-f32/v7', 'cutlass_sm80+splitK'."""
    import re
    parts = []
    for n in sorted(names or []):
        m = re.search(r"gemvx::kernel<int, int, __nv_bfloat16, (\w+), \w+, float, (?:true|false), (?:true|false), "
                      r"(?:true|false), (?:true|false), (\d+),", n)
        if m:
            parts.append(("gemvx" if m.group(1) == "__nv_bfloat16" else "gemvx-f32") + f"/v{m.group(2)}")
        elif "splitKreduce" in n:
            parts.append("splitK")
        elif "cutlass_80" in n:
            parts.append("cutlass_sm80")
        elif "nvjet" in n:
            parts.append("nvjet")
        else:
            parts.append(n.split("<")[0][-30:])
    return "+".join(sorted(set(parts)))


def is_gemv(names):
    return any("gemv" in n.lower() for n in names or [])


def analyze(cells, e1_cells, b0, out_dir):
    c2 = [c for c in cells if c["phase"] == "e2" and c["status"] == "clean" and c.get("median_us")]
    ref = e1_reference(e1_cells) if e1_cells else {}
    res = {"b0_gbps": b0, "fits": {}, "shapes": [], "msweep": [], "non_gemv_m1": []}

    # --- fits on the synthetic sweep (M=1), per K and per mode; pooled over all M=1 cells per mode
    for mode in MODES:
        for kname in ("K4096", "K14336"):
            pts = sorted((c["bytes"]["total"], c["median_us"]) for c in c2 if c["model"] == "synthetic"
                         and c["layer_name"].startswith(kname + "_") and c["timing_mode"] == mode)
            if len(pts) >= 4:
                res["fits"][f"{kname}/{mode}"] = hockney_fit(*zip(*pts))
        pts = sorted((c["bytes"]["total"], c["median_us"]) for c in c2 if c["shape"]["M"] == 1 and c["timing_mode"] == mode)
        if len(pts) >= 4:
            res["fits"][f"pooled_M1/{mode}"] = hockney_fit(*zip(*pts))
    # one tau + B/R per dispatched kernel: mixing kernels with different per-byte efficiency is not a valid fit
    for mode in MODES:
        groups = {}
        for c in c2:
            if c["shape"]["M"] == 1 and c["timing_mode"] == mode:
                groups.setdefault(kernel_label(c.get("kernel_names")), []).append((c["bytes"]["total"], c["median_us"]))
        for lab, pts in groups.items():
            if len({b for b, _ in pts}) >= 4:
                res["fits"][f"kernel:{lab}/{mode}"] = hockney_fit(*zip(*sorted(pts)))
    for f in res["fits"].values():
        f["fit_ok"] = f["rms_rel_residual"] <= 0.05

    def tau(k):
        return res["fits"].get(k, {}).get("tau_us")
    res["launch_split"] = {}
    for kname in ("K4096", "K14336", "pooled_M1"):
        g, e, i = tau(f"{kname}/graph"), tau(f"{kname}/eager"), tau(f"{kname}/isolated")
        if g is not None:
            res["launch_split"][kname] = {"tau_graph_us": g, "tau_eager_us": e, "tau_isolated_us": i,
                                          "exposed_launch_us": (e - g) if e is not None else None,
                                          "isolated_minus_graph_us": (i - g) if i is not None else None}

    # --- per real shape at M=1
    pooled = res["fits"].get("pooled_M1/graph")
    for model in ("llama31_8b", "qwen25_7b"):
        layers = sorted({c["layer_name"] for c in c2 if c["model"] == model},
                        key=lambda l: next(c["bytes"]["weights"] for c in c2 if c["model"] == model and c["layer_name"] == l))
        for layer in layers:
            row = {"model": model, "layer": layer}
            for mode in MODES:
                c = next((c for c in c2 if c["model"] == model and c["layer_name"] == layer and c["shape"]["M"] == 1
                          and c["timing_mode"] == mode), None)
                if not c:
                    continue
                row.update({"N": c["shape"]["N"], "K": c["shape"]["K"], "weight_mib": c["bytes"]["weights"] / MiB,
                            "bytes": c["bytes"]["total"]})
                row[f"{mode}_us"] = c["median_us"]
                row[f"{mode}_gbps"] = c["achieved_gbps"]
                if b0:
                    row[f"{mode}_pct_b0"] = 100 * c["achieved_gbps"] / b0
                if mode in ref:
                    row[f"{mode}_pct_e1_same_size"] = 100 * c["achieved_gbps"] / interp_gbps(ref[mode], c["bytes"]["total"])
                if mode == "graph":
                    row["kernels"] = c.get("kernel_names")
                    row["kernel_label"] = kernel_label(c.get("kernel_names"))
                    row["gemv"] = is_gemv(c.get("kernel_names"))
                    if not row["gemv"]:
                        res["non_gemv_m1"].append(f"{model}/{layer}")
            kf = res["fits"].get(f"kernel:{row.get('kernel_label')}/graph")
            fit = kf if kf and kf["fit_ok"] else None
            row["eff_fit_used"] = f"kernel:{row.get('kernel_label')}/graph" if fit else None
            if fit and "bytes" in row:
                bh = fit["B_half_bytes"]
                B = row["bytes"]
                B4 = B - 0.75 * 2 * row["N"] * row["K"]  # weights at 4 bits, activations/output unchanged
                row["eff_model_bf16"] = B / (B + bh)
                row["eff_model_4bit"] = B4 / (B4 + bh)
                row["measured_eff_graph"] = (B / (fit["R_inf_gbps"] * 1e3)) / row["graph_us"] if "graph_us" in row else None
            res["shapes"].append(row)

    # --- M sweep (graph mode)
    for model in ("llama31_8b", "qwen25_7b"):
        for layer in sorted({c["layer_name"] for c in c2 if c["model"] == model}):
            cs = sorted([c for c in c2 if c["model"] == model and c["layer_name"] == layer and c["timing_mode"] == "graph"],
                        key=lambda c: c["shape"]["M"])
            if not cs or cs[0]["shape"]["M"] != 1:
                continue
            t1 = cs[0]["median_us"]
            res["msweep"].append({"model": model, "layer": layer, "N": cs[0]["shape"]["N"], "K": cs[0]["shape"]["K"],
                                  "t_us": {c["shape"]["M"]: c["median_us"] for c in cs},
                                  "t_over_t1": {c["shape"]["M"]: c["median_us"] / t1 for c in cs},
                                  "kernels": {c["shape"]["M"]: c.get("kernel_names") for c in cs}})
    plots = plot_e2(c2, res, ref, out_dir)
    return res, md(res), plots


def plot_e2(c2, res, ref, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, facecolor=SURFACE)
    for ax, mode in zip(axes, MODES):
        if mode in ref:
            xs, ys = zip(*ref[mode])
            ax.plot(xs, ys, color=INK2, lw=1, ls=":", label="E1 Triton read (cold)")
        for kname in ("K4096", "K14336"):
            pts = sorted((c["bytes"]["total"], c["achieved_gbps"]) for c in c2 if c["model"] == "synthetic"
                         and c["layer_name"].startswith(kname + "_") and c["timing_mode"] == mode)
            if pts:
                ax.plot(*zip(*pts), color=COL[kname], lw=2, marker="o", ms=4, label=f"synthetic {kname}")
        for model, mk in (("llama31_8b", "s"), ("qwen25_7b", "D")):
            pts = [(c["bytes"]["total"], c["achieved_gbps"]) for c in c2 if c["model"] == model
                   and c["shape"]["M"] == 1 and c["timing_mode"] == mode]
            if pts:
                ax.plot(*zip(*pts), ls="none", marker=mk, ms=7, color=COL[model], mec=SURFACE, mew=1.5, label=f"{model} real")
        _style(ax, f"cuBLAS BF16 GEMV, M=1, {mode}", "bytes per op", "achieved GB/s" if mode == "isolated" else "")
        _size_axis(ax)
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    paths["e2_bandwidth"] = out_dir / "e2_bandwidth.png"
    fig.savefig(paths["e2_bandwidth"], dpi=140, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=SURFACE)
    xs = np.geomspace(2 ** 17, 2 ** 31, 300)
    for mode in MODES:
        f = res["fits"].get(f"pooled_M1/{mode}")
        if not f:
            continue
        ax.plot(xs, xs / (xs + f["B_half_bytes"]), color=MODE_COL[mode], lw=2,
                label=f"{mode}: B/(B+B_half), B_half {f['B_half_mib']:.2f} MiB")
        pts = [(c["bytes"]["total"], (c["bytes"]["total"] / (f["R_inf_gbps"] * 1e3)) / c["median_us"]) for c in c2
               if c["shape"]["M"] == 1 and c["timing_mode"] == mode]
        ax.plot(*zip(*pts), ls="none", marker="o", ms=5, color=MODE_COL[mode], mfc=SURFACE, mew=1.5)
    _style(ax, "Efficiency vs bytes (M=1): model curve and measured points", "bytes per op",
           "efficiency = (B / R_inf) / t")
    _size_axis(ax)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    paths["e2_efficiency"] = out_dir / "e2_efficiency.png"
    fig.savefig(paths["e2_efficiency"], dpi=140, facecolor=SURFACE)
    plt.close(fig)

    if res["msweep"]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, facecolor=SURFACE)
        for ax, model in zip(axes, ("llama31_8b", "qwen25_7b")):
            rows = [r for r in res["msweep"] if r["model"] == model]
            rows.sort(key=lambda r: r["N"] * r["K"])
            Ms = sorted({m for r in rows for m in r["t_over_t1"]})
            x = np.arange(len(rows))
            w = 0.8 / max(1, len(Ms) - 1)
            seq = ["#c6dbf7", "#8fb8ee", "#5795e2", "#2a78d6", "#1a56a3"]  # one hue, light to dark (magnitude of M)
            for j, M in enumerate([m for m in Ms if m != 1]):
                ax.bar(x + (j - (len(Ms) - 2) / 2) * w, [r["t_over_t1"].get(M, np.nan) for r in rows], w * 0.9,
                       color=seq[min(j + 1, len(seq) - 1)], label=f"M={M}")
            ax.axhline(1, color=INK2, lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([r["layer"] for r in rows], rotation=35, ha="right", fontsize=8)
            _style(ax, f"{model}: time at M relative to M=1 (graph)", "", "t(M) / t(1)" if model == "llama31_8b" else "")
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        paths["e2_msweep"] = out_dir / "e2_msweep.png"
        fig.savefig(paths["e2_msweep"], dpi=140, facecolor=SURFACE)
        plt.close(fig)
    return paths


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def md(res):
    L = ["## E2 (B1) cuBLAS BF16 GEMV", ""]
    if res["b0_gbps"]:
        L += [f"B0 = {res['b0_gbps']:.2f} GB/s (from the E1 session). `%E1` compares with the cold E1 Triton read of the "
              f"same byte size and mode (bandwidth on GB10 depends slightly on footprint).", ""]
    L += ["### Per real shape, M=1", "",
          "| model | layer | N x K | MiB | isolated us | eager us | graph us | graph GB/s | %B0 | %E1 same size | kernel (M=1) | model eff BF16 | model eff 4-bit |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["shapes"]:
        L.append(f"| {r['model']} | {r['layer']} | {r.get('N')} x {r.get('K')} | {_f(r.get('weight_mib'))} | "
                 f"{_f(r.get('isolated_us'), 2)} | {_f(r.get('eager_us'), 2)} | {_f(r.get('graph_us'), 2)} | "
                 f"{_f(r.get('graph_gbps'))} | {_f(r.get('graph_pct_b0'))} | {_f(r.get('graph_pct_e1_same_size'))} | "
                 f"{r.get('kernel_label', '-')} | {_f(100 * r['eff_model_bf16']) + '%' if 'eff_model_bf16' in r else '-'} | "
                 f"{_f(100 * r['eff_model_4bit']) + '%' if 'eff_model_4bit' in r else '-'} |")
    L += ["", "Model efficiency uses the graph-mode fit of the kernel cuBLAS dispatched for that shape (only fits with "
          "<= 5% residual): eff = B / (B + B_half), i.e. the tau-only loss relative to that kernel's own R_inf. "
          "4-bit replaces weight bytes by a quarter, activations and output unchanged (no scale metadata). "
          "The per-byte loss of a kernel shows up in %B0, not here.", ""]
    if res["non_gemv_m1"]:
        L += [f"**Non-GEMV kernel at M=1:** {', '.join(res['non_gemv_m1'])}", ""]
    else:
        L += ["cuBLAS dispatched a GEMV kernel (`gemvx`) for every real shape at M=1.", ""]
    L += ["### Fits t = tau + B / R (M=1)", "",
          "| fit | tau us | tau CI95 | R_inf GB/s | B_half MiB | rms rel resid | Theil-Sen tau | Theil-Sen R |", "|---|---|---|---|---|---|---|---|"]
    for k, f in sorted(res["fits"].items()):
        ci = f.get("tau_ci95_us")
        L.append(f"| {k} | {f['tau_us']:.2f} | {_f(ci[0], 2) + ' to ' + _f(ci[1], 2) if ci else '-'} | {f['R_inf_gbps']:.1f} | "
                 f"{f['B_half_mib']:.3f} | {100 * f['rms_rel_residual']:.1f}%{'' if f['fit_ok'] else ' misfit'} | "
                 f"{_f(f['theil_sen']['tau_us'], 2)} | {_f(f['theil_sen']['R_inf_gbps'])} |")
    L += ["", "### Launch-cost split", "", "| K | tau_graph | tau_eager | tau_isolated | eager - graph (exposed launch) | isolated - graph |",
          "|---|---|---|---|---|---|"]
    for k, v in res["launch_split"].items():
        L.append(f"| {k} | {_f(v['tau_graph_us'], 2)} | {_f(v['tau_eager_us'], 2)} | {_f(v['tau_isolated_us'], 2)} | "
                 f"{_f(v['exposed_launch_us'], 2)} | {_f(v['isolated_minus_graph_us'], 2)} |")
    if res["msweep"]:
        L += ["", "### M sweep (graph mode): t(M) / t(1) and dispatched kernel family", "",
              "| model | layer | M=2 | M=4 | M=8 | M=16 | kernels at M=2 / 16 |", "|---|---|---|---|---|---|---|"]
        for r in res["msweep"]:
            t = r["t_over_t1"]
            fam = lambda ks: "+".join(sorted({("gemv" if "gemv" in k.lower() else "splitK" if "splitK" in k else
                                                 "cutlass_sm80" if "cutlass_80" in k else "nvjet" if "nvjet" in k else k[:20])
                                                for k in (ks or [])}))
            L.append(f"| {r['model']} | {r['layer']} | {_f(t.get(2), 2)} | {_f(t.get(4), 2)} | {_f(t.get(8), 2)} | "
                     f"{_f(t.get(16), 2)} | {fam(r['kernels'].get(2))} / {fam(r['kernels'].get(16))} |")
    L.append("")
    return L
