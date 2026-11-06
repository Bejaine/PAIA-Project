# Summary: e2v2_s1_20261026

Cells: 218 total; clean 218

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.08%, p90 0.65%, max 3.97%.

Clean cells flagged noisy (CV > 3%): 18. With the tail-insensitive robust CV (IQR/1.349/median): 0 above 3%, median 0.07%.

## E2 (B1) cuBLAS BF16 GEMV

B0 = 891.69 GB/s (from the E1 session). `%E1` compares with the cold E1 Triton read of the same byte size and mode (bandwidth on GB10 depends slightly on footprint).

### Per real shape, M=1

| model | layer | N x K | MiB | isolated us | eager us | graph us | graph GB/s | %B0 | %E1 same size | kernel (M=1) | model eff BF16 | model eff 4-bit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | k_proj | 1024 x 4096 | 8.0 | 23.60 | 12.50 | 11.84 | 709.4 | 79.6 | 89.3 | gemvx/v6 | 79.1% | 48.7% |
| llama31_8b | v_proj | 1024 x 4096 | 8.0 | 23.55 | 12.46 | 11.85 | 708.5 | 79.5 | 89.2 | gemvx/v6 | 79.1% | 48.7% |
| llama31_8b | q_proj | 4096 x 4096 | 32.0 | 52.00 | 40.97 | 40.34 | 832.2 | 93.3 | 96.9 | gemvx/v6 | 93.8% | 79.1% |
| llama31_8b | o_proj | 4096 x 4096 | 32.0 | 52.00 | 40.95 | 40.40 | 831.0 | 93.2 | 96.8 | gemvx/v6 | 93.8% | 79.1% |
| llama31_8b | qkv_fused | 6144 x 4096 | 48.0 | 70.46 | 59.28 | 58.66 | 858.3 | 96.3 | 98.0 | gemvx/v7 | 95.9% | 85.5% |
| llama31_8b | up_proj | 14336 x 4096 | 112.0 | 145.89 | 134.68 | 134.06 | 876.3 | 98.3 | 99.1 | gemvx/v7 | 98.2% | 93.2% |
| llama31_8b | down_proj | 4096 x 14336 | 112.0 | 146.75 | 135.52 | 134.94 | 870.6 | 97.6 | 98.5 | gemvx/v7 | 98.2% | 93.2% |
| llama31_8b | gate_proj | 14336 x 4096 | 112.0 | 145.82 | 134.63 | 134.08 | 876.2 | 98.3 | 99.1 | gemvx/v7 | 98.2% | 93.2% |
| llama31_8b | gate_up_fused | 28672 x 4096 | 224.0 | 277.89 | 266.41 | 265.89 | 883.6 | 99.1 | 99.6 | gemvx/v7 | 99.1% | 96.5% |
| llama31_8b | lm_head | 128256 x 4096 | 1002.0 | 1196.74 | 1182.57 | 1179.80 | 890.8 | 99.9 | 100.0 | gemvx/v6 | 99.8% | 99.2% |
| qwen25_7b | k_proj | 512 x 3584 | 3.5 | 21.66 | 10.00 | 5.94 | 618.7 | 69.4 | 92.0 | gemvx/v7 | 63.2% | 30.2% |
| qwen25_7b | v_proj | 512 x 3584 | 3.5 | 21.63 | 10.00 | 5.92 | 620.9 | 69.6 | 92.3 | gemvx/v7 | 63.2% | 30.2% |
| qwen25_7b | q_proj | 3584 x 3584 | 24.5 | 43.71 | 32.80 | 32.19 | 798.6 | 89.6 | 94.0 | gemvx/v6 | 92.0% | 74.3% |
| qwen25_7b | o_proj | 3584 x 3584 | 24.5 | 43.58 | 32.72 | 32.18 | 798.9 | 89.6 | 94.0 | gemvx/v6 | 92.0% | 74.3% |
| qwen25_7b | qkv_fused | 4608 x 3584 | 31.5 | 51.98 | 40.94 | 40.26 | 820.9 | 92.1 | 95.6 | gemvx/v6 | 93.7% | 78.8% |
| qwen25_7b | up_proj | 18944 x 3584 | 129.5 | 166.40 | 155.11 | 154.49 | 879.3 | 98.6 | 99.5 | gemvx/v7 | 98.4% | 94.1% |
| qwen25_7b | down_proj | 3584 x 18944 | 129.5 | 168.00 | 156.80 | 156.19 | 869.7 | 97.5 | 98.4 | gemvx/v7 | 98.4% | 94.1% |
| qwen25_7b | gate_proj | 18944 x 3584 | 129.5 | 166.37 | 155.11 | 154.50 | 879.2 | 98.6 | 99.5 | gemvx/v7 | 98.4% | 94.1% |
| qwen25_7b | gate_up_fused | 37888 x 3584 | 259.0 | 319.17 | 307.41 | 306.40 | 886.6 | 99.4 | 99.9 | gemvx/v7 | 99.2% | 96.9% |
| qwen25_7b | lm_head | 152064 x 3584 | 1039.5 | 1241.41 | 1227.04 | 1223.99 | 890.8 | 99.9 | 100.0 | gemvx/v6 | 99.8% | 99.2% |

Model efficiency uses the graph-mode fit of the kernel cuBLAS dispatched for that shape (only fits with <= 5% residual): eff = B / (B + B_half), i.e. the tau-only loss relative to that kernel's own R_inf. 4-bit replaces weight bytes by a quarter, activations and output unchanged (no scale metadata). The per-byte loss of a kernel shows up in %B0, not here.

cuBLAS dispatched a GEMV kernel (`gemvx`) for every real shape at M=1.

### Fits t = tau + B / R (M=1)

| fit | tau us | tau CI95 | R_inf GB/s | B_half MiB | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|
| K14336/eager | 9.05 | 5.26 to 10.88 | 976.1 | 8.421 | 13.1% misfit | 5.42 | 898.9 |
| K14336/graph | 3.09 | 2.64 to 4.00 | 878.2 | 2.591 | 6.4% misfit | 3.89 | 892.7 |
| K14336/isolated | 20.69 | 17.77 to 22.63 | 942.2 | 18.590 | 7.8% misfit | 16.66 | 901.2 |
| K4096/eager | 7.67 | 5.59 to 8.94 | 1006.7 | 7.365 | 15.2% misfit | 2.98 | 904.9 |
| K4096/graph | 2.47 | 2.13 to 2.66 | 903.4 | 2.130 | 4.2% | 2.38 | 892.6 |
| K4096/isolated | 18.72 | 16.52 to 20.22 | 949.8 | 16.956 | 8.6% misfit | 14.07 | 907.6 |
| kernel:gemvx/v6/eager | 3.05 | 2.78 to 3.21 | 886.7 | 2.576 | 1.4% | 3.90 | 890.9 |
| kernel:gemvx/v6/graph | 2.51 | 2.29 to 2.70 | 887.4 | 2.121 | 1.3% | 3.31 | 892.6 |
| kernel:gemvx/v6/isolated | 14.24 | 13.86 to 14.59 | 888.1 | 12.065 | 1.1% | 14.79 | 888.5 |
| kernel:gemvx/v7/eager | 7.79 | 6.54 to 8.95 | 931.0 | 6.919 | 8.2% misfit | 4.08 | 899.5 |
| kernel:gemvx/v7/graph | 2.38 | 1.97 to 2.58 | 900.5 | 2.045 | 4.2% | 2.30 | 891.5 |
| kernel:gemvx/v7/isolated | 19.28 | 18.05 to 20.41 | 923.0 | 16.973 | 4.1% | 15.09 | 898.1 |
| pooled_M1/eager | 7.24 | 5.86 to 8.45 | 965.0 | 6.664 | 13.5% misfit | 3.23 | 896.3 |
| pooled_M1/graph | 2.54 | 2.29 to 2.82 | 890.2 | 2.156 | 5.9% misfit | 2.45 | 892.5 |
| pooled_M1/isolated | 18.43 | 16.96 to 19.80 | 934.9 | 16.436 | 8.0% misfit | 14.28 | 894.7 |

### Launch-cost split

| K | tau_graph | tau_eager | tau_isolated | eager - graph (exposed launch) | isolated - graph |
|---|---|---|---|---|---|
| K4096 | 2.47 | 7.67 | 18.72 | 5.20 | 16.25 |
| K14336 | 3.09 | 9.05 | 20.69 | 5.95 | 17.59 |
| pooled_M1 | 2.54 | 7.24 | 18.43 | 4.70 | 15.89 |

### M sweep (graph mode): t(M) / t(1) and dispatched kernel family

| model | layer | M=2 | M=4 | M=8 | M=16 | kernels at M=2 / 16 |
|---|---|---|---|---|---|---|
| llama31_8b | down_proj | 1.01 | 1.00 | 1.01 | 1.01 | cutlass_sm80+splitK / cutlass_sm80 |
| llama31_8b | gate_proj | 1.02 | 1.02 | 1.02 | 1.03 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_up_fused | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | k_proj | 1.05 | 1.05 | 1.06 | 1.08 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| llama31_8b | lm_head | 1.00 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | o_proj | 1.00 | 1.00 | 1.00 | 1.01 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | q_proj | 1.00 | 1.00 | 1.00 | 1.01 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | qkv_fused | 1.01 | 1.01 | 1.02 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | up_proj | 1.02 | 1.02 | 1.02 | 1.03 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | v_proj | 1.05 | 1.05 | 1.06 | 1.08 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | down_proj | 1.02 | 1.02 | 1.02 | 1.02 | cutlass_sm80+splitK / cutlass_sm80 |
| qwen25_7b | gate_proj | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_up_fused | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | k_proj | 1.23 | 1.24 | 1.25 | 1.27 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | lm_head | 1.00 | 1.01 | 1.02 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | o_proj | 0.98 | 0.98 | 0.98 | 0.99 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | q_proj | 0.98 | 0.98 | 0.98 | 0.99 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | qkv_fused | 0.98 | 0.98 | 0.98 | 0.99 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | up_proj | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | v_proj | 1.23 | 1.24 | 1.25 | 1.27 | cutlass_sm80+splitK / cutlass_sm80+splitK |

Kernel names: 212 cells filled from `dispatch_table_e2_e4.json` (profiler capture lost on this stack, findings.md); 6 recorded cells matched it exactly, 0 partial recordings were a subset of it and were completed.

## Canary

- reference 2026-10-26T03:49:16+05:30: 825.17 GB/s, CV 0.02%
- per-cell canary over clean cells: median 825.36 GB/s, CV 0.17%, range 824.49 to 828.50 (n=436)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2490.0, 'median': 2580.0, 'max': 2670.0}
- max temperature: 69 C
- clock-event reasons seen: ['gpu_idle', 'sw_power_cap']

## Cross-session comparison

218 matching clean cells; |diff| median 0.08%, max 2.17%; 0 differ by more than 3% (listed in compare.csv).

## Plots

- [e2_bandwidth](e2_bandwidth.png)
- [e2_efficiency](e2_efficiency.png)
- [e2_msweep](e2_msweep.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
