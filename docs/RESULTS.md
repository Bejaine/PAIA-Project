# PAIA Results (living document)

What we have measured so far, the numbers of record, and where the evidence lives. Detailed reasoning, caveats and
surprises are in `paia-bench/findings.md`; the chronological story is in `docs/PROGRESS.md`; what comes next is in
`docs/PLAN.md`. Update this file whenever a phase produces results of record.

Last updated: 2026-10-22 (after E1/E2 schema v2 reruns and the focused E3 on GB10).

## Status by experiment

| Experiment | GB10 (DGX Spark, P2) | RTX PRO 5000 Blackwell (P3) |
|---|---|---|
| E0 fingerprint + selfcheck | done (`phase0_20261016`) | done (`paia-results/rtxpro5000bw/env_20261016`) |
| E1 bandwidth ceiling (B0) | **done, schema v2, 2 sessions** (`e1v2_s1`, `e1v2_s2`) | preliminary: 80 of 360 cells, schema v1, different stack |
| E2 cuBLAS BF16 GEMV | **done, schema v2, 2 sessions** (`e2v2_s1`, `e2v2_s2`) | not started |
| E3 Nsight Compute | focused: v6 vs v7 GEMV (`e3_ncu_gemv_variants`); byte-match check done in Phase 0 | counters accessible; not run |
| E4 library FP8 / INT4 | **not started** (all library paths verified functional) | not started (no vLLM, so no Marlin) |
| E5 whole-token decode | feasibility only (vLLM Llama-8B eager and graph decode; nsys works) | not started |

Results of record live under `paia-bench/results/dgxspark_gb10/<session>/summary/` (`summary.md`, `fits.json`,
`cells.csv`, plots). v1 sessions (`s1_20261016`, `s2_20261018`, `e2s1_20261019`, `e2s2_20261019`) are kept for the record
but superseded by v2. `debug_*` sessions are never results (git-ignored).

## E1: bandwidth ceiling and per-op cost (GB10, schema v2)

| Quantity | e1v2_s1 | e1v2_s2 |
|---|---|---|
| **R_inf_B0** (cold graph-mode read, >= 10x L2, median across allocations) | **239.0 GB/s** | **239.5 GB/s** |
| as % of 273 GB/s spec | 87.5% | 87.7% |
| **probe tau** (single-kernel Triton read, graph mode) | **1.70 us** [1.50, 1.80] | **1.66 us** [1.40, 1.76] |
| probe tau, eager / isolated | 4.85 / 11.31 us | 4.66 / 11.43 us |
| **B_half** (graph) | **0.39 MiB** | 0.38 MiB |
| copy plateau (2 B counted) / write (fill) plateau | 240.2 / 196.4 GB/s | 241.8 / 196.8 GB/s |
| hot L2 peak (Triton read) | 1407 GB/s at 16 MiB | 1520 GB/s at 16 MiB |
| L2 falloff | between 16 and 32 MiB (L2 = 24 MiB) | same |

- Writes are the slow direction on LPDDR5x (fill 0.81x read).
- Bandwidth is about 4% higher at <= 100 MiB footprints than at GiB scale (open question; TLB reach suspected).
- torch `sum` is not a valid tau probe (launch config changes with size); the Triton probe defines tau.

## E2: cuBLAS BF16 GEMV (GB10, schema v2)

Graph mode, M = 1, % of B0 (239.25 GB/s), mean of the two v2 sessions:

| Shape | Llama-3.1-8B | Qwen2.5-7B | cuBLAS kernel |
|---|---|---|---|
| q_proj | 70.6% | 67.9% | gemvx v6 |
| k_proj / v_proj | 70.7% / 68.6% | 87.0% / 87.1% | Llama v6, Qwen v7 |
| o_proj | 69.2% | 68.8% | v6 |
| qkv_fused | 70.3% | 69.5% | v6 |
| gate_proj / up_proj | 97.8% / 97.9% | 71.1% / 71.0% | Llama v7, Qwen v6 |
| down_proj | 96.5% | 96.7% | v7 |
| gate_up_fused | 98.4% | 71.5% | Llama v7, Qwen v6 |
| lm_head | 73.1% | 72.3% | v7 (slow regime at very large N) |

- **13 of 20 real shapes run at about 70% of B0.** 11 of them use `gemvx` v6; the 2 lm_heads use v7 in its slow large-N regime.
- **Per-kernel fits (graph):** v6 tau 0.38 us (CI includes 0), R_inf 167.8 GB/s (1.9% residual): the loss is per byte,
  not per op. v7 about 226 to 235 GB/s.
- **Exposed launch cost** (tau_eager - tau_graph): about 2 us (K = 4096) to 3.5 us (K = 14336); isolated - graph about 8 to 11 us.
- **M sweep:** from M = 2 cuBLAS switches to an sm80 CUTLASS GEMM; on the v6 shapes and lm_heads t(M=2..16) is 0.71 to
  0.82x t(M=1). Qwen k/v_proj at M >= 2 (split-K GEMM) is 1.2x slower.
- **Per-token cost** (GEMVs only, graph mode, from per-op times): Llama 70.3 ms vs 62.7 ms at B0 (+12%); Qwen 77.2 ms vs
  59.1 ms (+30%).
- cuBLAS dispatched a GEMV kernel (`gemvx`) for every real shape at M = 1 (none chose a GEMM).

## E3: why cuBLAS v6 is slow (GB10, Nsight Compute)

| | v6 (Llama q, Qwen gate) | v7 (Llama gate, down) |
|---|---|---|
| block | 64 threads (16 x 4) | 128 threads (32 x 4) |
| registers / occupancy | 164 / 25% (12 of 48 warps) | 164 / 25% |
| DRAM fills / weight bytes | 1.00 | 1.00 |
| long-scoreboard stall cycles per issue | 29 to 32 | 17 to 20 |
| issue active | 4.9% | 6.0 to 6.7% |

Same bytes, same register-limited occupancy, but v6's 16-lane rows keep about half the bytes in flight per warp, so it
waits on memory and loses about 30%. Byte accounting works on GB10 via `lts__d_sectors_fill_sysmem` x 32 B (no `dram__*`
metrics exist on GB10); measured/expected bytes = 1.00 to 1.002.

## Measurement quality (GB10)

- Cross-session agreement (v2): E2 median |diff| 0.27% (8 of 218 cells > 3%); E1 0.38% (outliers are hot L2-edge cells).
- Variance between allocations is real on GB10 (up to 6% for an 8 MiB GEMV across fresh allocations), which is why schema
  v2 measures 5 allocations per cell.
- Our monitor does not bias medians (< 0.2% in a controlled test); cold cells are truly cold (ncu DRAM fills = bytes).
- Canary reference about 238 to 244 GB/s with CV 0.2 to 0.8%; in-session canary runs about 1.3% below its reference.
- SM clock 2177 to 2190 MHz in every cell (max 3003); no throttle reasons; max 61 C.

## RTX PRO 5000 Blackwell (P3, preliminary, not yet comparable)

R_inf_B0 1238.8 GB/s (92% of 1344 spec), probe tau about 1.4 us (poorly constrained), B_half about 1.61 MiB (about 4x
GB10's), fill 1.00x read, copy 0.89x read. Measured in a host venv (torch 2.14.1, cuBLAS 13.1.1, Triton 3.8.0, no vLLM),
SM clock unstable under load (180 to 2595 MHz). Row-wise FP8 `_scaled_mm` is correct on sm_120 (broken on GB10's sm_121).
Details: `paia-bench/findings.md`, section "Platform rtxpro5000bw".

## Hypothesis status

"GB10 is less tau-sensitive than a fast discrete GPU": **supported so far, not confirmed.** GB10 B_half 0.39 MiB vs RTX
about 1.6 MiB, with similar tau. Needs RTX on a matched stack, a stable SM clock, and a full E1 before it can be claimed.

## Predictions vs measured (GB10)

| Quantity | Predicted (`paia-bench/predictions.md`) | Measured |
|---|---|---|
| R_inf_B0 | 225 GB/s | 239 GB/s |
| probe tau (graph) | 3.5 us | 1.7 us |
| B_half | 0.75 MiB | 0.39 MiB |
| copy / write vs read | 0.85x / 0.90x | 1.00x / 0.81x |
| hot L2 peak | about 1 TB/s | 1.4 to 1.5 TB/s |
| cuBLAS GEMV R_inf vs B0 | 90% | 70% (v6) to 98% (v7), shape dependent |
| non-GEMV dispatch at M = 1 | none | none |
