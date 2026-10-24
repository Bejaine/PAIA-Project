# Summary: e2v2_s2_20261024

Cells: 218 total; clean 218

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.08%, p90 0.82%, max 3.31%.

Clean cells flagged noisy (CV > 3%): 20. With the tail-insensitive robust CV (IQR/1.349/median): 0 above 3%, median 0.06%.

## E2 (B1) cuBLAS BF16 GEMV

B0 = 891.73 GB/s (from the E1 session). `%E1` compares with the cold E1 Triton read of the same byte size and mode (bandwidth on GB10 depends slightly on footprint).

### Per real shape, M=1

| model | layer | N x K | MiB | isolated us | eager us | graph us | graph GB/s | %B0 | %E1 same size | kernel (M=1) | model eff BF16 | model eff 4-bit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | k_proj | 1024 x 4096 | 8.0 | 23.90 | 12.45 | 11.81 | 711.0 | 79.7 | 90.1 | gemvx/v6 | 79.1% | 48.7% |
| llama31_8b | v_proj | 1024 x 4096 | 8.0 | 23.90 | 12.45 | 11.85 | 708.5 | 79.5 | 89.7 | gemvx/v6 | 79.1% | 48.7% |
| llama31_8b | q_proj | 4096 x 4096 | 32.0 | 52.54 | 40.96 | 40.30 | 833.0 | 93.4 | 96.8 | gemvx/v6 | 93.8% | 79.1% |
| llama31_8b | o_proj | 4096 x 4096 | 32.0 | 52.45 | 40.94 | 40.33 | 832.4 | 93.4 | 96.7 | gemvx/v6 | 93.8% | 79.1% |
| llama31_8b | qkv_fused | 6144 x 4096 | 48.0 | 70.78 | 59.30 | 58.66 | 858.4 | 96.3 | 98.7 | gemvx/v7 | 95.9% | 85.5% |
| llama31_8b | gate_proj | 14336 x 4096 | 112.0 | 146.27 | 134.69 | 134.05 | 876.3 | 98.3 | 99.3 | gemvx/v7 | 98.2% | 93.2% |
| llama31_8b | up_proj | 14336 x 4096 | 112.0 | 146.22 | 134.64 | 134.05 | 876.4 | 98.3 | 99.3 | gemvx/v7 | 98.2% | 93.2% |
| llama31_8b | down_proj | 4096 x 14336 | 112.0 | 147.01 | 135.55 | 134.85 | 871.2 | 97.7 | 98.7 | gemvx/v7 | 98.2% | 93.2% |
| llama31_8b | gate_up_fused | 28672 x 4096 | 224.0 | 278.30 | 266.34 | 265.83 | 883.8 | 99.1 | 99.6 | gemvx/v7 | 99.1% | 96.5% |
| llama31_8b | lm_head | 128256 x 4096 | 1002.0 | 1197.41 | 1182.59 | 1179.82 | 890.8 | 99.9 | 100.0 | gemvx/v6 | 99.8% | 99.2% |
| qwen25_7b | v_proj | 512 x 3584 | 3.5 | 22.02 | 10.01 | 5.94 | 618.9 | 69.4 | 90.6 | gemvx/v7 | 63.2% | 30.2% |
| qwen25_7b | k_proj | 512 x 3584 | 3.5 | 21.98 | 10.06 | 5.95 | 618.6 | 69.4 | 90.6 | gemvx/v7 | 63.2% | 30.2% |
| qwen25_7b | q_proj | 3584 x 3584 | 24.5 | 43.49 | 32.48 | 32.20 | 798.3 | 89.5 | 93.8 | gemvx/v6 | 92.1% | 74.4% |
| qwen25_7b | o_proj | 3584 x 3584 | 24.5 | 43.49 | 32.74 | 32.16 | 799.2 | 89.6 | 93.9 | gemvx/v6 | 92.1% | 74.4% |
| qwen25_7b | qkv_fused | 4608 x 3584 | 31.5 | 52.42 | 40.84 | 40.21 | 821.8 | 92.2 | 95.6 | gemvx/v6 | 93.7% | 78.9% |
| qwen25_7b | gate_proj | 18944 x 3584 | 129.5 | 166.75 | 155.17 | 154.45 | 879.5 | 98.6 | 99.5 | gemvx/v7 | 98.4% | 94.1% |
| qwen25_7b | up_proj | 18944 x 3584 | 129.5 | 166.75 | 155.15 | 154.46 | 879.4 | 98.6 | 99.5 | gemvx/v7 | 98.4% | 94.1% |
| qwen25_7b | down_proj | 3584 x 18944 | 129.5 | 168.29 | 156.79 | 156.15 | 869.9 | 97.6 | 98.4 | gemvx/v7 | 98.4% | 94.1% |
| qwen25_7b | gate_up_fused | 37888 x 3584 | 259.0 | 318.94 | 307.42 | 306.40 | 886.6 | 99.4 | 99.9 | gemvx/v7 | 99.2% | 96.9% |
| qwen25_7b | lm_head | 152064 x 3584 | 1039.5 | 1241.60 | 1227.01 | 1223.88 | 890.9 | 99.9 | 100.0 | gemvx/v6 | 99.8% | 99.2% |

Model efficiency uses the graph-mode fit of the kernel cuBLAS dispatched for that shape (only fits with <= 5% residual): eff = B / (B + B_half), i.e. the tau-only loss relative to that kernel's own R_inf. 4-bit replaces weight bytes by a quarter, activations and output unchanged (no scale metadata). The per-byte loss of a kernel shows up in %B0, not here.

cuBLAS dispatched a GEMV kernel (`gemvx`) for every real shape at M=1.

### Fits t = tau + B / R (M=1)

| fit | tau us | tau CI95 | R_inf GB/s | B_half MiB | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|
| K14336/eager | 9.13 | 5.45 to 10.90 | 975.1 | 8.488 | 12.8% misfit | 5.50 | 899.5 |
| K14336/graph | 3.09 | 2.64 to 3.99 | 878.2 | 2.588 | 6.3% misfit | 3.89 | 892.8 |
| K14336/isolated | 21.45 | 18.48 to 23.51 | 943.7 | 19.309 | 8.0% misfit | 17.33 | 901.9 |
| K4096/eager | 7.70 | 5.64 to 8.95 | 1005.4 | 7.386 | 15.0% misfit | 3.03 | 904.8 |
| K4096/graph | 2.46 | 2.12 to 2.65 | 903.2 | 2.122 | 4.2% | 2.37 | 892.6 |
| K4096/isolated | 19.04 | 16.81 to 20.55 | 949.0 | 17.230 | 8.5% misfit | 14.48 | 904.9 |
| kernel:gemvx/v6/eager | 3.07 | 2.84 to 3.25 | 887.8 | 2.601 | 1.2% | 3.81 | 890.8 |
| kernel:gemvx/v6/graph | 2.50 | 2.29 to 2.69 | 887.6 | 2.115 | 1.2% | 3.30 | 892.6 |
| kernel:gemvx/v6/isolated | 14.52 | 14.27 to 14.74 | 887.9 | 12.297 | 0.8% | 14.89 | 888.4 |
| kernel:gemvx/v7/eager | 7.82 | 6.57 to 8.96 | 930.9 | 6.938 | 8.1% misfit | 4.10 | 899.6 |
| kernel:gemvx/v7/graph | 2.38 | 1.97 to 2.56 | 900.4 | 2.040 | 4.1% | 2.27 | 891.4 |
| kernel:gemvx/v7/isolated | 19.63 | 18.40 to 20.75 | 923.3 | 17.284 | 4.1% | 15.70 | 899.7 |
| pooled_M1/eager | 7.27 | 5.89 to 8.48 | 965.2 | 6.694 | 13.5% misfit | 3.22 | 896.1 |
| pooled_M1/graph | 2.53 | 2.28 to 2.81 | 890.3 | 2.151 | 5.8% misfit | 2.43 | 892.6 |
| pooled_M1/isolated | 18.84 | 17.33 to 20.24 | 935.2 | 16.800 | 8.2% misfit | 14.80 | 894.8 |

### Launch-cost split

| K | tau_graph | tau_eager | tau_isolated | eager - graph (exposed launch) | isolated - graph |
|---|---|---|---|---|---|
| K4096 | 2.46 | 7.70 | 19.04 | 5.24 | 16.57 |
| K14336 | 3.09 | 9.13 | 21.45 | 6.04 | 18.36 |
| pooled_M1 | 2.53 | 7.27 | 18.84 | 4.74 | 16.30 |

### M sweep (graph mode): t(M) / t(1) and dispatched kernel family

| model | layer | M=2 | M=4 | M=8 | M=16 | kernels at M=2 / 16 |
|---|---|---|---|---|---|---|
| llama31_8b | down_proj | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80+splitK / cutlass_sm80 |
| llama31_8b | gate_proj | 1.02 | 1.02 | 1.02 | 1.03 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_up_fused | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | k_proj | 1.05 | 1.06 | 1.06 | 1.09 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| llama31_8b | lm_head | 1.00 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | o_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | q_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | qkv_fused | 1.01 | 1.02 | 1.02 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | up_proj | 1.02 | 1.02 | 1.02 | 1.03 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | v_proj | 1.05 | 1.05 | 1.06 | 1.08 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | down_proj | 1.01 | 1.02 | 1.03 | 1.03 | cutlass_sm80+splitK / cutlass_sm80 |
| qwen25_7b | gate_proj | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_up_fused | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | k_proj | 1.23 | 1.23 | 1.25 | 1.27 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | lm_head | 1.01 | 1.01 | 1.02 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | o_proj | 0.98 | 0.98 | 0.98 | 0.99 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | q_proj | 0.98 | 0.98 | 0.98 | 0.99 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | qkv_fused | 0.98 | 0.98 | 0.98 | 0.99 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | up_proj | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | v_proj | 1.23 | 1.23 | 1.25 | 1.27 | cutlass_sm80+splitK / cutlass_sm80+splitK |

Kernel names: 210 cells filled from `dispatch_table.json` (profiler capture lost on this stack, findings.md); 8 recorded cells matched it exactly.

## Canary

- reference 2026-10-24T05:24:48+05:30: 824.78 GB/s, CV 0.12%
- per-cell canary over clean cells: median 824.88 GB/s, CV 0.18%, range 823.79 to 828.08 (n=436)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2460.0, 'median': 2595.0, 'max': 2685.0}
- max temperature: 67 C
- clock-event reasons seen: ['gpu_idle', 'sw_power_cap']

## Cross-session comparison

218 matching clean cells; |diff| median 0.08%, max 2.91%; 0 differ by more than 3% (listed in compare.csv).

## Plots

- [e2_bandwidth](e2_bandwidth.png)
- [e2_efficiency](e2_efficiency.png)
- [e2_msweep](e2_msweep.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
