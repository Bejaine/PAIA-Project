# Predictions (fill in BEFORE running E1/E2)

Write predictions here and commit them before the corresponding experiment runs, so the results can be
compared against what we expected. Do not edit a prediction after its experiment has run; add a note below it instead.

Platform: `________`  Predicted by: `____`  Date committed: `________`

## E1 (B0) achievable bandwidth

| Quantity | Prediction | Reasoning |
|---|---|---|
| Cold read plateau R_inf_B0 (GB/s) | | |
| R_inf_B0 as % of spec bandwidth | | |
| Copy GB/s (counting 2B) vs read | | |
| Write (fill) GB/s vs read | | |
| tau from cold graph-mode read fit (us) | | |
| B_half = tau * R_inf (MiB) | | |
| Hot (L2) plateau GB/s | | |
| Size where the hot curve falls off L2 (MiB) | | |

## E2 (B1) cuBLAS BF16 GEMV

| Quantity | Prediction | Reasoning |
|---|---|---|
| tau_graph (us) | | |
| tau_isolated (us) | | |
| tau_eager - tau_graph, exposed launch cost (us) | | |
| R_inf of GEMV as % of R_inf_B0 | | |
| Efficiency of smallest real shape (k_proj) at BF16 | | |
| Efficiency of k_proj at 4-bit size (B/4) | | |
| Any shape where cuBLAS picks a non-GEMV kernel at M=1 | | |

## E4 library FP8 / INT4 (M=1)

| Impl | Theoretical speedup | Predicted measured speedup | Reasoning |
|---|---|---|---|
| FP8 `_scaled_mm` per-tensor | ~2.0x | | |
| INT4 tinygemm g128 scale+zero | ~3.76x | | |
| INT4 Marlin g128 scale-only | ~3.88x | | |

## E5 whole-token decode (BF16, batch 1)

| Quantity | Prediction | Reasoning |
|---|---|---|
| Bandwidth floor tokens/s (bytes/token / R_inf_B0) | | |
| vLLM eager tokens/s | | |
| vLLM CUDA graph tokens/s | | |
| Kernels per token | | |
| Matmul share of kernel time | | |
| GPU idle fraction between kernels (eager / graph) | | |

## Hypothesis check

Because B_half = tau * R_inf, GB10 should be less tau-sensitive than a fast discrete GPU unless its tau is larger.
Prediction for GB10 tau vs RTX 5000-class tau: `________`
