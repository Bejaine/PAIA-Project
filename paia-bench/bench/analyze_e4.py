"""E4 analysis: library FP8 / INT4 against cuBLAS BF16 (E2) per shape, metadata-corrected theoretical speedups, the
efficiency identity, per-kernel Hockney fits, a per-op / per-byte split of each kernel's excess time over B/B0,
activation-quantization cost, per-token GEMV cost, M sweep. Called from bench.analyze.

Definitions (per shape, same timing mode, M):
  S_measured   = t_bf16 / t_q
  S_theory     = bytes_bf16 / bytes_q          (stored bytes: weights + metadata + x + y)
  efficiency   = S_measured / S_theory
  BW ratio     = (bytes_q / t_q) / (bytes_bf16 / t_bf16)   (algebraically equal to efficiency; checked numerically)
  excess time  = t - B / B0 = per-byte part B * (1/R_asym - 1/B0) + per-op part (the rest)
  R_asym       = asymptotic streaming rate of an implementation: Theil-Sen slope of t against B over its M=1 cells with
                 B >= ASYM_MIN_BYTES (whatever kernels the library dispatches there). A single tau + B/R fit over all sizes
                 misfits several library kernels (tile and wave quantization), so the split does not rely on it.
"""
import numpy as np

from bench.analyze import INK, INK2, MiB, MODES, SURFACE, _size_axis, _style, hockney_fit

MODELS = ("llama31_8b", "qwen25_7b")
GEMM_IMPLS = ("torch_scaled_mm_fp8_pt", "vllm_cutlass_fp8_rw", "tinygemm_int4_g128", "marlin_int4_g128")
FULL_OF = {"torch_scaled_mm_fp8_pt_full": "torch_scaled_mm_fp8_pt", "vllm_cutlass_fp8_rw_full": "vllm_cutlass_fp8_rw"}
SHORT = {"cublas_f_linear": "cuBLAS BF16", "torch_scaled_mm_fp8_pt": "FP8 pt (_scaled_mm)",
         "vllm_cutlass_fp8_rw": "FP8 rw (CUTLASS)", "tinygemm_int4_g128": "INT4 tinygemm", "marlin_int4_g128": "INT4 Marlin",
         "torch_scaled_mm_fp8_pt_full": "FP8 pt + act quant", "vllm_cutlass_fp8_rw_full": "FP8 rw + act quant"}
# Fixed categorical order (reference palette slots 1 to 5); color follows the implementation across charts.
COL = {"cublas_f_linear": "#2a78d6", "torch_scaled_mm_fp8_pt": "#eb6834", "vllm_cutlass_fp8_rw": "#1baf7a",
       "tinygemm_int4_g128": "#eda100", "marlin_int4_g128": "#e87ba4"}
MARK = {"cublas_f_linear": "o", "torch_scaled_mm_fp8_pt": "s", "vllm_cutlass_fp8_rw": "D", "tinygemm_int4_g128": "^",
        "marlin_int4_g128": "v"}
ASYM_MIN_BYTES = 64 * 2**20
PER_LAYER = {"unfused": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
             "fused": ("qkv_fused", "o_proj", "gate_up_fused", "down_proj")}


def kernel_label(names):
    """Short label of the GEMM kernel(s) a call dispatched; activation-quant kernels and memsets are left out."""
    import re
    parts = []
    for n in names or []:
        if "nvjet" in n:
            m = re.search(r"mma_(\d+x\d+)x\d+", n)
            parts.append("nvjet_" + (m.group(1) if m else "?"))
        elif "xmma_gemm" in n:
            m = re.search(r"(sm\d+)_xmma", n)
            parts.append(f"xmma_{m.group(1) if m else '?'}")
        elif "cutlass" in n and ("Sm120" in n or "sm120" in n):
            parts.append("cutlass_sm120")
        elif "cutlass" in n:
            parts.append("cutlass")
        elif "tinygemm" in n:
            parts.append("tinygemm")
        elif "marlin" in n.lower():
            parts.append("marlin")
        elif "splitK" in n:
            parts.append("splitK")
    return "+".join(sorted(set(parts))) or "?"


def _index(cells, phase):
    """{(impl, model, layer, M, mode): cell} over clean cells; several sessions: mean of medians, kept as a cell copy."""
    acc = {}
    for c in cells:
        if c["phase"] != phase or c["status"] != "clean" or not c.get("median_us"):
            continue
        k = (c["impl"], c["model"], c["layer_name"], c["shape"]["M"], c["timing_mode"])
        acc.setdefault(k, []).append(c)
    out = {}
    for k, cs in acc.items():
        c = dict(cs[0])
        c["median_us"] = float(np.mean([x["median_us"] for x in cs]))
        c["achieved_gbps"] = c["bytes"]["total"] / (c["median_us"] * 1e-6) / 1e9
        c["n_sessions"] = len(cs)
        out[k] = c
    return out


def analyze(cells, bf16_cells, b0, out_dir):
    q = _index(cells, "e4")
    r16 = _index(bf16_cells, "e2")
    res = {"b0_gbps": b0, "bf16_sessions": sorted({c["session_id"] for c in bf16_cells if c["phase"] == "e2"}),
           "fits": {}, "shapes": [], "identity_max_abs_diff": 0.0, "act_quant": [], "per_token": [], "msweep": []}

    # --- per-kernel fits per impl and mode (M=1, synthetic + real), grouped by dispatched GEMM kernel
    for impl in GEMM_IMPLS:
        for mode in MODES:
            groups = {}
            for (im, model, layer, M, md_), c in q.items():
                if im == impl and M == 1 and md_ == mode:
                    groups.setdefault(kernel_label(c.get("kernel_names")), []).append((c["bytes"]["total"], c["median_us"]))
            for lab, pts in groups.items():
                if len({b for b, _ in pts}) >= 4:
                    f = hockney_fit(*zip(*sorted(pts)))
                    f["fit_ok"] = f["rms_rel_residual"] <= 0.05
                    f["n_points"] = len(pts)
                    res["fits"][f"{impl}|{lab}|{mode}"] = f

    # --- asymptotic streaming rate per impl and per dispatched kernel (graph, M=1, large ops only)
    res["asym"] = {}
    from scipy.stats import theilslopes
    for impl in GEMM_IMPLS:
        big = [(kernel_label(c.get("kernel_names")), c["bytes"]["total"], c["median_us"]) for k, c in q.items()
               if k[0] == impl and k[3] == 1 and k[4] == "graph" and c["bytes"]["total"] >= ASYM_MIN_BYTES]
        groups = {}   # one asymptote per dispatched kernel; a pooled one across kernels would mix different R
        for lab in {g for g, _, _ in big}:
            groups[lab] = [p for p in big if p[0] == lab]
        for lab, pts in groups.items():
            if len(pts) < 3:
                continue
            _, B, t = (np.array(v) for v in zip(*sorted(pts, key=lambda p: p[1])))
            B, t = B.astype(float), t.astype(float)
            slope, icpt, lo, hi = theilslopes(t, B)
            ci = [1e-3 / hi if hi > 0 else None, 1e-3 / lo if lo > 0 else None]   # None: slope CI reaches 0 (unbounded)
            res["asym"][f"{impl}|{lab}"] = {"R_asym_gbps": 1e-3 / slope, "R_asym_ci95_gbps": ci,
                                           "intercept_us": icpt, "n_points": len(pts), "min_bytes": int(B.min()),
                                           "kernels": sorted({g for g, _, _ in pts})}

    def asym_for(impl, c):
        return res["asym"].get(f"{impl}|{kernel_label(c.get('kernel_names'))}")

    # --- per real shape, M=1, every impl, every mode
    for model in MODELS:
        layers = sorted({k[2] for k in q if k[1] == model}, key=lambda l: next(
            c["bytes"]["weights"] for k, c in q.items() if k[1] == model and k[2] == l))
        for layer in layers:
            for impl in GEMM_IMPLS + tuple(FULL_OF):
                row = {"model": model, "layer": layer, "impl": impl}
                for mode in MODES:
                    c = q.get((impl, model, layer, 1, mode))
                    c16 = r16.get(("cublas_f_linear", model, layer, 1, mode))
                    if not c:
                        continue
                    row.update({"N": c["shape"]["N"], "K": c["shape"]["K"], "bytes_q": c["bytes"]["total"],
                                "weight_mib_q": c["bytes"]["weights"] / MiB, "meta_bytes": c["bytes"]["metadata"]})
                    row[f"{mode}_us"] = c["median_us"]
                    row[f"{mode}_gbps"] = c["achieved_gbps"]
                    if b0:
                        row[f"{mode}_pct_b0"] = 100 * c["achieved_gbps"] / b0
                    if mode == "graph":
                        row["kernel_label"] = kernel_label(c.get("kernel_names"))
                        row["kernels"] = c.get("kernel_names")
                    if c16:
                        row["bytes_bf16"] = c16["bytes"]["total"]
                        S = c16["median_us"] / c["median_us"]
                        St = c16["bytes"]["total"] / c["bytes"]["total"]
                        bw = c["achieved_gbps"] / c16["achieved_gbps"]
                        row[f"{mode}_bf16_us"] = c16["median_us"]
                        row[f"{mode}_speedup"] = S
                        row["theory_speedup"] = St
                        row[f"{mode}_efficiency"] = S / St
                        row[f"{mode}_bw_ratio"] = bw
                        res["identity_max_abs_diff"] = max(res["identity_max_abs_diff"], abs(S / St - bw))
                        if b0:
                            row[f"{mode}_speedup_vs_ideal_bf16"] = (c16["bytes"]["total"] / (b0 * 1e3)) / c["median_us"]
                # per-byte / per-op split of the excess over B/B0 (graph), from the impl's asymptotic rate
                c = q.get((impl, model, layer, 1, "graph"))
                a = asym_for(impl, c) if c and impl in GEMM_IMPLS else None
                if c and b0 and a:
                    B, t = c["bytes"]["total"], c["median_us"]
                    t_ideal = B / (b0 * 1e3)
                    row["excess_us"] = t - t_ideal
                    row["per_byte_us"] = B / (a["R_asym_gbps"] * 1e3) - t_ideal
                    row["per_op_us"] = row["excess_us"] - row["per_byte_us"]
                if "graph_us" in row or "eager_us" in row or "isolated_us" in row:
                    res["shapes"].append(row)

    # --- activation quantization cost: *_full minus GEMM-only, same shape and mode
    for full, gemm in FULL_OF.items():
        for (im, model, layer, M, mode), c in q.items():
            if im != full:
                continue
            g = q.get((gemm, model, layer, M, mode))
            if g:
                res["act_quant"].append({"impl": full, "model": model, "layer": layer, "M": M, "mode": mode,
                                         "full_us": c["median_us"], "gemm_us": g["median_us"],
                                         "act_quant_us": c["median_us"] - g["median_us"]})

    # --- per-token GEMV cost (graph, M=1): sum of per-op times over all layers + lm_head
    from bench import config
    mcfg = config.shapes_cfg()["models"]
    for model in MODELS:
        L = mcfg[model]["n_layers"]
        for variant, layers in PER_LAYER.items():
            for impl in ("cublas_f_linear",) + GEMM_IMPLS + tuple(FULL_OF):
                src = r16 if impl == "cublas_f_linear" else q
                cs = [src.get((impl, model, l, 1, "graph")) for l in layers]
                head = src.get((impl, model, "lm_head", 1, "graph"))
                if not all(cs) or not head:
                    continue
                t = L * sum(c["median_us"] for c in cs) + head["median_us"]
                B = L * sum(c["bytes"]["total"] for c in cs) + head["bytes"]["total"]
                res["per_token"].append({"model": model, "variant": variant, "impl": impl, "t_ms": t / 1e3,
                                         "bytes_gib": B / 2**30, "ideal_ms": (B / (b0 * 1e3) / 1e3) if b0 else None})
        for variant in PER_LAYER:
            base = next((r for r in res["per_token"] if r["model"] == model and r["variant"] == variant
                         and r["impl"] == "cublas_f_linear"), None)
            for r in res["per_token"]:
                if base and r["model"] == model and r["variant"] == variant:
                    r["speedup_vs_bf16"] = base["t_ms"] / r["t_ms"]
                    r["theory_speedup"] = base["bytes_gib"] / r["bytes_gib"]

    # --- M sweep (graph)
    for impl in GEMM_IMPLS + tuple(FULL_OF):
        for model in MODELS:
            for layer in sorted({k[2] for k in q if k[0] == impl and k[1] == model and k[2] in
                                 {l for v in PER_LAYER.values() for l in v} | {"lm_head"}}):
                cs = {k[3]: c for k, c in q.items() if k[0] == impl and k[1] == model and k[2] == layer and k[4] == "graph"}
                if 1 not in cs:
                    continue
                t1 = cs[1]["median_us"]
                res["msweep"].append({"impl": impl, "model": model, "layer": layer,
                                      "t_over_t1": {M: cs[M]["median_us"] / t1 for M in sorted(cs)},
                                      "kernels": {M: kernel_label(cs[M].get("kernel_names")) for M in sorted(cs)}})
    plots = plot_e4(q, r16, res, out_dir)
    return res, md(res), plots


def plot_e4(q, r16, res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = {}
    b0 = res["b0_gbps"]
    # 1. achieved bandwidth vs bytes per op (graph, M=1), cuBLAS BF16 for reference
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for impl in ("cublas_f_linear",) + GEMM_IMPLS:
        src = r16 if impl == "cublas_f_linear" else q
        pts = sorted((c["bytes"]["total"], c["achieved_gbps"]) for k, c in src.items()
                     if k[0] == impl and k[3] == 1 and k[4] == "graph")
        if pts:
            ax.plot(*zip(*pts), ls="none", marker=MARK[impl], ms=8, color=COL[impl], mec=SURFACE, mew=2, label=SHORT[impl])
    _style(ax, "Achieved bandwidth per op, M=1, graph mode (stored bytes incl. metadata)", "bytes per op", "GB/s")
    _size_axis(ax)
    if b0:
        ax.axhline(b0, color=INK2, lw=1, ls=":")
        ax.annotate(f"B0 {b0:.0f} GB/s", (ax.get_xlim()[0], b0), xytext=(4, 4), textcoords="offset points", color=INK2,
                    fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    paths["e4_bandwidth"] = out_dir / "e4_bandwidth.png"
    fig.savefig(paths["e4_bandwidth"], dpi=140, facecolor=SURFACE)
    plt.close(fig)

    # 2. %B0 per real shape (efficiency against an ideal kernel streaming at B0), small multiples per model
    impls = ("cublas_f_linear",) + GEMM_IMPLS
    if b0 and any(k[0] in GEMM_IMPLS for k in q):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharey=True, facecolor=SURFACE)
        for ax, model in zip(axes, MODELS):
            layers = sorted({k[2] for k in q if k[1] == model}, key=lambda l: next(
                c["shape"]["N"] * c["shape"]["K"] for k, c in q.items() if k[1] == model and k[2] == l))
            x = np.arange(len(layers))
            w = 0.84 / len(impls)
            for j, impl in enumerate(impls):
                src = r16 if impl == "cublas_f_linear" else q
                ys = [100 * src[(impl, model, l, 1, "graph")]["achieved_gbps"] / b0
                      if (impl, model, l, 1, "graph") in src else np.nan for l in layers]
                ax.bar(x + (j - (len(impls) - 1) / 2) * w, ys, w, color=COL[impl], label=SHORT[impl],
                       edgecolor=SURFACE, linewidth=2)
            ax.axhline(100, color=INK2, lw=1, ls=":")
            ax.set_xticks(x)
            ax.set_xticklabels(layers, fontsize=8)
            _style(ax, f"{model}: achieved bandwidth as % of B0 ({b0:.0f} GB/s), graph, M=1", "", "% of B0")
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.08))
        fig.tight_layout()
        paths["e4_pct_b0"] = out_dir / "e4_pct_b0.png"
        fig.savefig(paths["e4_pct_b0"], dpi=140, facecolor=SURFACE)
        plt.close(fig)
    return paths


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def md(res):
    L = ["## E4 library FP8 / INT4 vs cuBLAS BF16", ""]
    L += [f"B0 = {_f(res['b0_gbps'], 2)} GB/s. BF16 reference: E2 session(s) {', '.join(res['bf16_sessions']) or 'none'} "
          f"(mean of medians). Theoretical speedup = stored bytes ratio (weights + metadata + x + y). "
          f"Identity check, max |efficiency - BW ratio| over all rows: {res['identity_max_abs_diff']:.2e}.", ""]
    L += ["### Per real shape, M=1, graph mode", "",
          "| model | layer | impl | kernel | graph us | BF16 us | GB/s | %B0 | S measured | S theory | efficiency | "
          "eager S | isolated S | excess us | per-op us | per-byte us |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["shapes"]:
        L.append(f"| {r['model']} | {r['layer']} | {SHORT.get(r['impl'], r['impl'])} | {r.get('kernel_label', '-')} | "
                 f"{_f(r.get('graph_us'), 2)} | {_f(r.get('graph_bf16_us'), 2)} | {_f(r.get('graph_gbps'))} | "
                 f"{_f(r.get('graph_pct_b0'))} | {_f(r.get('graph_speedup'), 2)} | {_f(r.get('theory_speedup'), 2)} | "
                 f"{_f(100 * r['graph_efficiency']) + '%' if 'graph_efficiency' in r else '-'} | "
                 f"{_f(r.get('eager_speedup'), 2)} | {_f(r.get('isolated_speedup'), 2)} | {_f(r.get('excess_us'), 2)} | "
                 f"{_f(r.get('per_op_us'), 2)} | {_f(r.get('per_byte_us'), 2)} |")
    L += ["", "%B0 is also the efficiency against an ideal BF16 kernel at B0 (cuBLAS BF16 itself runs at about 70% of B0 on "
          "13 of 20 shapes, so efficiency against cuBLAS can exceed 100%). excess = t - B / B0; per-byte = B (1/R_asym - 1/B0) "
          "with the asymptotic rate of the kernel dispatched for that shape (below; '-' when that kernel has fewer than 3 ops "
          ">= 64 MiB); per-op = excess - per-byte.", "",
          "### Asymptotic streaming rate per implementation (graph, M=1, ops >= 64 MiB)", "",
          "| impl | R_asym GB/s | CI95 | %B0 | intercept us | points | kernels |", "|---|---|---|---|---|---|---|"]
    for key, a in sorted(res.get("asym", {}).items()):
        impl, lab = key.split("|")
        lo, hi = a["R_asym_ci95_gbps"]
        L.append(f"| {SHORT[impl]} ({lab}) | {a['R_asym_gbps']:.1f} | {_f(lo)} to {_f(hi) if hi else 'unbounded'} | "
                 f"{_f(100 * a['R_asym_gbps'] / res['b0_gbps']) if res['b0_gbps'] else '-'} | {a['intercept_us']:.2f} | "
                 f"{a['n_points']} | {', '.join(a['kernels'])} |")
    L.append("")
    L += ["### Per-kernel fits t = tau + B / R (M=1; informational, misfits are expected for tiled kernels)", "",
          "| impl | kernel | mode | points | tau us | tau CI95 | R_inf GB/s | R_inf %B0 | B_half MiB | rms rel resid |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, f in sorted(res["fits"].items()):
        impl, lab, mode = k.split("|")
        ci = f.get("tau_ci95_us")
        L.append(f"| {SHORT[impl]} | {lab} | {mode} | {f['n_points']} | {f['tau_us']:.2f} | "
                 f"{_f(ci[0], 2) + ' to ' + _f(ci[1], 2) if ci else '-'} | {f['R_inf_gbps']:.1f} | "
                 f"{_f(100 * f['R_inf_gbps'] / res['b0_gbps']) if res['b0_gbps'] else '-'} | {f['B_half_mib']:.3f} | "
                 f"{100 * f['rms_rel_residual']:.1f}%{'' if f['fit_ok'] else ' misfit'} |")
    if res["act_quant"]:
        L += ["", "### Activation quantization cost (full op minus GEMM only), M=1", "",
              "| impl | mode | median us | min us | max us | shapes |", "|---|---|---|---|---|---|"]
        for impl in FULL_OF:
            for mode in MODES:
                v = [a["act_quant_us"] for a in res["act_quant"] if a["impl"] == impl and a["mode"] == mode and a["M"] == 1]
                if v:
                    L.append(f"| {SHORT[impl]} | {mode} | {np.median(v):.2f} | {min(v):.2f} | {max(v):.2f} | {len(v)} |")
    if res["per_token"]:
        L += ["", "### Per-token GEMV cost (graph, M=1; all layers + lm_head)", "",
              "| model | layers | impl | ms/token | GiB/token | ideal ms at B0 | % of ideal | speedup vs BF16 | theory |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in res["per_token"]:
            L.append(f"| {r['model']} | {r['variant']} | {SHORT[r['impl']]} | {r['t_ms']:.2f} | {r['bytes_gib']:.2f} | "
                     f"{_f(r['ideal_ms'], 2)} | {_f(100 * r['ideal_ms'] / r['t_ms']) if r['ideal_ms'] else '-'} | "
                     f"{_f(r.get('speedup_vs_bf16'), 2)} | {_f(r.get('theory_speedup'), 2)} |")
    if res["msweep"]:
        L += ["", "### M sweep (graph): t(M) / t(1)", "", "| impl | model | layer | M=2 | M=4 | M=8 | M=16 | kernels M=1 / 16 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in res["msweep"]:
            t = r["t_over_t1"]
            L.append(f"| {SHORT[r['impl']]} | {r['model']} | {r['layer']} | {_f(t.get(2), 2)} | {_f(t.get(4), 2)} | "
                     f"{_f(t.get(8), 2)} | {_f(t.get(16), 2)} | {r['kernels'].get(1)} / {r['kernels'].get(16)} |")
    L.append("")
    return L
