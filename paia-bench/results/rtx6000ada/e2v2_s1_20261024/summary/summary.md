# Summary: e2v2_s1_20261024

Cells: 218 total; clean 218

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.08%, p90 0.72%, max 3.70%.

Clean cells flagged noisy (CV > 3%): 19. With the tail-insensitive robust CV (IQR/1.349/median): 0 above 3%, median 0.06%.

## E2 (B1) cuBLAS BF16 GEMV

B0 = 891.73 GB/s (from the E1 session). `%E1` compares with the cold E1 Triton read of the same byte size and mode (bandwidth on GB10 depends slightly on footprint).

### Per real shape, M=1

| model | layer | N x K | MiB | isolated us | eager us | graph us | graph GB/s | %B0 | %E1 same size | kernel (M=1) | model eff BF16 | model eff 4-bit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | k_proj | 1024 x 4096 | 8.0 | 23.90 | 12.40 | 11.82 | 710.5 | 79.7 | 90.0 | gemvx/v6 | 79.1% | 48.7% |
| llama31_8b | v_proj | 1024 x 4096 | 8.0 | 23.87 | 12.42 | 11.80 | 711.5 | 79.8 | 90.1 | gemvx/v6 | 79.1% | 48.7% |
| llama31_8b | o_proj | 4096 x 4096 | 32.0 | 52.19 | 40.92 | 40.32 | 832.6 | 93.4 | 96.7 | gemvx/v6 | 93.8% | 79.1% |
| llama31_8b | q_proj | 4096 x 4096 | 32.0 | 52.26 | 40.96 | 40.34 | 832.2 | 93.3 | 96.7 | gemvx/v6 | 93.8% | 79.1% |
| llama31_8b | qkv_fused | 6144 x 4096 | 48.0 | 70.62 | 59.24 | 58.63 | 858.9 | 96.3 | 98.8 | gemvx/v7 | 95.9% | 85.5% |
| llama31_8b | gate_proj | 14336 x 4096 | 112.0 | 146.05 | 134.66 | 134.03 | 876.5 | 98.3 | 99.3 | gemvx/v7 | 98.2% | 93.2% |
| llama31_8b | up_proj | 14336 x 4096 | 112.0 | 145.98 | 134.63 | 134.07 | 876.3 | 98.3 | 99.3 | gemvx/v7 | 98.2% | 93.2% |
| llama31_8b | down_proj | 4096 x 14336 | 112.0 | 146.77 | 135.50 | 134.93 | 870.6 | 97.6 | 98.7 | gemvx/v7 | 98.2% | 93.2% |
| llama31_8b | gate_up_fused | 28672 x 4096 | 224.0 | 278.16 | 266.37 | 265.62 | 884.5 | 99.2 | 99.7 | gemvx/v7 | 99.1% | 96.5% |
| llama31_8b | lm_head | 128256 x 4096 | 1002.0 | 1197.04 | 1182.58 | 1179.75 | 890.8 | 99.9 | 100.0 | gemvx/v6 | 99.8% | 99.2% |
| qwen25_7b | k_proj | 512 x 3584 | 3.5 | 21.86 | 10.08 | 5.93 | 620.8 | 69.6 | 90.9 | gemvx/v7 | 63.2% | 30.2% |
| qwen25_7b | v_proj | 512 x 3584 | 3.5 | 21.63 | 10.10 | 5.93 | 620.7 | 69.6 | 90.9 | gemvx/v7 | 63.2% | 30.2% |
| qwen25_7b | o_proj | 3584 x 3584 | 24.5 | 43.87 | 32.70 | 32.14 | 799.9 | 89.7 | 94.0 | gemvx/v6 | 92.0% | 74.3% |
| qwen25_7b | q_proj | 3584 x 3584 | 24.5 | 44.03 | 32.73 | 32.17 | 799.1 | 89.6 | 93.9 | gemvx/v6 | 92.0% | 74.3% |
| qwen25_7b | qkv_fused | 4608 x 3584 | 31.5 | 52.06 | 40.76 | 40.19 | 822.3 | 92.2 | 95.6 | gemvx/v6 | 93.7% | 78.8% |
| qwen25_7b | gate_proj | 18944 x 3584 | 129.5 | 166.53 | 155.07 | 154.48 | 879.3 | 98.6 | 99.5 | gemvx/v7 | 98.4% | 94.1% |
| qwen25_7b | up_proj | 18944 x 3584 | 129.5 | 166.59 | 155.05 | 154.51 | 879.1 | 98.6 | 99.5 | gemvx/v7 | 98.4% | 94.1% |
| qwen25_7b | down_proj | 3584 x 18944 | 129.5 | 168.10 | 156.83 | 156.17 | 869.8 | 97.5 | 98.4 | gemvx/v7 | 98.4% | 94.1% |
| qwen25_7b | gate_up_fused | 37888 x 3584 | 259.0 | 319.17 | 307.43 | 306.33 | 886.8 | 99.4 | 99.9 | gemvx/v7 | 99.2% | 96.9% |
| qwen25_7b | lm_head | 152064 x 3584 | 1039.5 | 1241.70 | 1227.09 | 1223.81 | 890.9 | 99.9 | 100.0 | gemvx/v6 | 99.8% | 99.2% |

Model efficiency uses the graph-mode fit of the kernel cuBLAS dispatched for that shape (only fits with <= 5% residual): eff = B / (B + B_half), i.e. the tau-only loss relative to that kernel's own R_inf. 4-bit replaces weight bytes by a quarter, activations and output unchanged (no scale metadata). The per-byte loss of a kernel shows up in %B0, not here.

cuBLAS dispatched a GEMV kernel (`gemvx`) for every real shape at M=1.

### Fits t = tau + B / R (M=1)

| fit | tau us | tau CI95 | R_inf GB/s | B_half MiB | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|
| K14336/eager | 9.05 | 5.17 to 10.92 | 976.6 | 8.432 | 13.3% misfit | 5.45 | 898.7 |
| K14336/graph | 3.09 | 2.64 to 3.99 | 878.3 | 2.591 | 6.3% misfit | 3.92 | 893.0 |
| K14336/isolated | 20.97 | 18.05 to 22.95 | 942.8 | 18.855 | 7.8% misfit | 16.85 | 901.5 |
| K4096/eager | 7.70 | 5.64 to 8.93 | 1006.0 | 7.391 | 15.0% misfit | 3.02 | 905.0 |
| K4096/graph | 2.47 | 2.12 to 2.67 | 903.3 | 2.127 | 4.3% | 2.37 | 892.6 |
| K4096/isolated | 18.94 | 16.79 to 20.39 | 948.2 | 17.123 | 8.3% misfit | 14.47 | 907.3 |
| kernel:gemvx/v6/eager | 3.03 | 2.79 to 3.22 | 887.0 | 2.563 | 1.3% | 3.78 | 890.9 |
| kernel:gemvx/v6/graph | 2.50 | 2.28 to 2.72 | 887.8 | 2.118 | 1.3% | 3.27 | 892.7 |
| kernel:gemvx/v6/isolated | 14.56 | 14.24 to 14.86 | 888.7 | 12.339 | 0.9% | 14.99 | 888.7 |
| kernel:gemvx/v7/eager | 7.85 | 6.62 to 8.95 | 931.0 | 6.967 | 7.9% misfit | 4.08 | 899.7 |
| kernel:gemvx/v7/graph | 2.38 | 1.96 to 2.57 | 900.8 | 2.041 | 4.3% | 2.31 | 891.6 |
| kernel:gemvx/v7/isolated | 19.43 | 18.21 to 20.55 | 923.0 | 17.104 | 4.2% | 15.09 | 897.0 |
| pooled_M1/eager | 7.26 | 5.88 to 8.47 | 965.8 | 6.691 | 13.6% misfit | 3.22 | 896.3 |
| pooled_M1/graph | 2.54 | 2.28 to 2.81 | 890.5 | 2.153 | 5.9% misfit | 2.45 | 892.6 |
| pooled_M1/isolated | 18.66 | 17.20 to 20.01 | 934.6 | 16.632 | 7.8% misfit | 14.52 | 895.1 |

### Launch-cost split

| K | tau_graph | tau_eager | tau_isolated | eager - graph (exposed launch) | isolated - graph |
|---|---|---|---|---|---|
| K4096 | 2.47 | 7.70 | 18.94 | 5.23 | 16.47 |
| K14336 | 3.09 | 9.05 | 20.97 | 5.96 | 17.88 |
| pooled_M1 | 2.54 | 7.26 | 18.66 | 4.73 | 16.13 |

### M sweep (graph mode): t(M) / t(1) and dispatched kernel family

| model | layer | M=2 | M=4 | M=8 | M=16 | kernels at M=2 / 16 |
|---|---|---|---|---|---|---|
| llama31_8b | down_proj | 1.01 | 1.01 | 1.01 | 1.01 | cutlass_sm80+splitK / cutlass_sm80 |
| llama31_8b | gate_proj | 1.02 | 1.02 | 1.02 | 1.03 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_up_fused | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | k_proj | 1.05 | 1.05 | 1.06 | 1.08 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| llama31_8b | lm_head | 1.00 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | o_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | q_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | qkv_fused | 1.02 | 1.02 | 1.02 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | up_proj | 1.02 | 1.02 | 1.02 | 1.03 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | v_proj | 1.05 | 1.06 | 1.07 | 1.08 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | down_proj | 1.01 | 1.02 | 1.02 | 1.03 | cutlass_sm80+splitK / cutlass_sm80 |
| qwen25_7b | gate_proj | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_up_fused | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | k_proj | 1.24 | 1.24 | 1.25 | 1.27 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | lm_head | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | o_proj | 0.98 | 0.98 | 0.98 | 0.99 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | q_proj | 0.98 | 0.98 | 0.98 | 0.99 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | qkv_fused | 0.98 | 0.98 | 0.99 | 0.99 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | up_proj | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | v_proj | 1.24 | 1.24 | 1.25 | 1.27 | cutlass_sm80+splitK / cutlass_sm80+splitK |

Kernel names: 197 cells filled from `dispatch_table.json` (profiler capture lost on this stack, findings.md); 21 recorded cells matched it exactly.

## Canary

- reference 2026-10-24T04:11:39+05:30: 825.09 GB/s, CV 0.02%
- per-cell canary over clean cells: median 825.39 GB/s, CV 0.18%, range 824.37 to 828.38 (n=436)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2460.0, 'median': 2595.0, 'max': 2685.0}
- max temperature: 66 C
- clock-event reasons seen: ['gpu_idle', 'sw_power_cap']

## Plots

- [e2_bandwidth](e2_bandwidth.png)
- [e2_efficiency](e2_efficiency.png)
- [e2_msweep](e2_msweep.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
