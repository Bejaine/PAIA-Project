# Summary: e2v2_s1

Cells: 218 total; clean 218

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.44%, p90 1.69%, max 4.46%.

Clean cells flagged noisy (CV > 3%): 148. With the tail-insensitive robust CV (IQR/1.349/median): 14 above 3%, median 0.46%.

## E2 (B1) cuBLAS BF16 GEMV

B0 = 239.49 GB/s (from the E1 session). `%E1` compares with the cold E1 Triton read of the same byte size and mode (bandwidth on GB10 depends slightly on footprint).

### Per real shape, M=1

| model | layer | N x K | MiB | isolated us | eager us | graph us | graph GB/s | %B0 | %E1 same size | kernel (M=1) | model eff BF16 | model eff 4-bit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | k_proj | 1024 x 4096 | 8.0 | 62.02 | 53.11 | 50.51 | 166.3 | 69.4 | 71.3 | gemvx/v6 | 98.4% | 93.9% |
| llama31_8b | v_proj | 1024 x 4096 | 8.0 | 62.14 | 53.81 | 50.99 | 164.7 | 68.8 | 70.6 | gemvx/v6 | 98.4% | 93.9% |
| llama31_8b | q_proj | 4096 x 4096 | 32.0 | 215.94 | 204.17 | 198.87 | 168.8 | 70.5 | 70.8 | gemvx/v6 | 99.6% | 98.4% |
| llama31_8b | o_proj | 4096 x 4096 | 32.0 | 211.15 | 200.45 | 202.89 | 165.5 | 69.1 | 69.4 | gemvx/v6 | 99.6% | 98.4% |
| llama31_8b | qkv_fused | 6144 x 4096 | 48.0 | 307.07 | 300.74 | 299.46 | 168.1 | 70.2 | 70.5 | gemvx/v6 | 99.7% | 98.9% |
| llama31_8b | gate_proj | 14336 x 4096 | 112.0 | 508.99 | 502.29 | 500.46 | 234.7 | 98.0 | 98.1 | gemvx/v7 | - | - |
| llama31_8b | up_proj | 14336 x 4096 | 112.0 | 509.74 | 504.63 | 501.46 | 234.3 | 97.8 | 97.9 | gemvx/v7 | - | - |
| llama31_8b | down_proj | 4096 x 14336 | 112.0 | 516.54 | 510.06 | 508.22 | 231.2 | 96.5 | 96.6 | gemvx/v7 | - | - |
| llama31_8b | gate_up_fused | 28672 x 4096 | 224.0 | 1004.43 | 995.52 | 994.53 | 236.2 | 98.6 | 98.5 | gemvx/v7 | - | - |
| llama31_8b | lm_head | 128256 x 4096 | 1002.0 | 5998.40 | 5988.70 | 6005.84 | 175.0 | 73.1 | 73.2 | gemvx/v7 | - | - |
| qwen25_7b | k_proj | 512 x 3584 | 3.5 | 26.53 | 19.23 | 17.70 | 207.8 | 86.8 | 93.0 | gemvx/v7 | - | - |
| qwen25_7b | v_proj | 512 x 3584 | 3.5 | 26.53 | 19.66 | 17.57 | 209.3 | 87.4 | 93.7 | gemvx/v7 | - | - |
| qwen25_7b | q_proj | 3584 x 3584 | 24.5 | 164.80 | 158.32 | 159.75 | 160.9 | 67.2 | 67.8 | gemvx/v6 | 99.5% | 97.9% |
| qwen25_7b | o_proj | 3584 x 3584 | 24.5 | 165.98 | 158.00 | 155.96 | 164.8 | 68.8 | 69.5 | gemvx/v6 | 99.5% | 97.9% |
| qwen25_7b | qkv_fused | 4608 x 3584 | 31.5 | 207.65 | 201.52 | 198.21 | 166.7 | 69.6 | 69.9 | gemvx/v6 | 99.6% | 98.4% |
| qwen25_7b | gate_proj | 18944 x 3584 | 129.5 | 808.03 | 801.66 | 798.54 | 170.1 | 71.0 | 71.0 | gemvx/v6 | 99.9% | 99.6% |
| qwen25_7b | up_proj | 18944 x 3584 | 129.5 | 808.51 | 800.63 | 798.43 | 170.1 | 71.0 | 71.0 | gemvx/v6 | 99.9% | 99.6% |
| qwen25_7b | down_proj | 3584 x 18944 | 129.5 | 597.47 | 590.76 | 589.01 | 230.6 | 96.3 | 96.3 | gemvx/v7 | - | - |
| qwen25_7b | gate_up_fused | 37888 x 3584 | 259.0 | 1590.54 | 1589.50 | 1588.48 | 171.0 | 71.4 | 71.3 | gemvx/v6 | 99.9% | 99.8% |
| qwen25_7b | lm_head | 152064 x 3584 | 1039.5 | 6284.70 | 6285.79 | 6300.70 | 173.0 | 72.3 | 72.4 | gemvx/v7 | - | - |

Model efficiency uses the graph-mode fit of the kernel cuBLAS dispatched for that shape (only fits with <= 5% residual): eff = B / (B + B_half), i.e. the tau-only loss relative to that kernel's own R_inf. 4-bit replaces weight bytes by a quarter, activations and output unchanged (no scale metadata). The per-byte loss of a kernel shows up in %B0, not here.

cuBLAS dispatched a GEMV kernel (`gemvx`) for every real shape at M=1.

### Fits t = tau + B / R (M=1)

| fit | tau us | tau CI95 | R_inf GB/s | B_half MiB | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|
| K14336/eager | 8.24 | 7.53 to 11.62 | 227.0 | 1.783 | 8.1% misfit | 26.73 | 235.0 |
| K14336/graph | 5.16 | 4.29 to 8.58 | 225.7 | 1.110 | 9.1% misfit | 24.78 | 234.8 |
| K14336/isolated | 16.11 | 15.37 to 19.93 | 227.1 | 3.489 | 8.0% misfit | 35.03 | 235.0 |
| K4096/eager | 3.91 | 0.63 to 5.11 | 191.1 | 0.713 | 15.0% misfit | 9.18 | 173.0 |
| K4096/graph | 2.02 | -0.94 to 2.63 | 192.3 | 0.370 | 14.2% misfit | 5.68 | 172.4 |
| K4096/isolated | 10.33 | 7.95 to 11.68 | 184.1 | 1.813 | 11.2% misfit | 15.61 | 172.5 |
| kernel:cutlass_sm80+splitK/eager | 7.94 | 7.23 to 8.37 | 225.9 | 1.710 | 2.0% | 7.40 | 221.5 |
| kernel:cutlass_sm80+splitK/graph | 5.08 | 4.08 to 5.57 | 230.5 | 1.117 | 3.2% | 4.45 | 224.8 |
| kernel:cutlass_sm80+splitK/isolated | 15.45 | 14.80 to 15.98 | 224.4 | 3.306 | 1.4% | 14.99 | 222.6 |
| kernel:gemvx/v6/eager | 3.21 | 1.77 to 8.33 | 168.0 | 0.514 | 2.1% | 6.57 | 171.3 |
| kernel:gemvx/v6/graph | 0.82 | -0.25 to 5.05 | 167.8 | 0.131 | 2.0% | 4.10 | 171.3 |
| kernel:gemvx/v6/isolated | 12.29 | 11.04 to 15.46 | 169.2 | 1.982 | 1.7% | 14.99 | 171.3 |
| kernel:gemvx/v7/eager | 4.44 | 2.81 to 5.28 | 226.2 | 0.959 | 11.5% misfit | 4.21 | 232.2 |
| kernel:gemvx/v7/graph | 2.47 | 1.20 to 2.85 | 226.1 | 0.532 | 10.8% misfit | 2.48 | 232.3 |
| kernel:gemvx/v7/isolated | 10.94 | 9.72 to 11.77 | 222.7 | 2.323 | 10.1% misfit | 10.33 | 232.1 |
| pooled_M1/eager | 4.54 | 3.06 to 5.87 | 201.8 | 0.875 | 15.5% misfit | -18.12 | 175.7 |
| pooled_M1/graph | 2.39 | 1.29 to 3.12 | 201.8 | 0.460 | 15.3% misfit | -20.84 | 175.1 |
| pooled_M1/isolated | 11.62 | 10.22 to 13.44 | 200.0 | 2.216 | 14.2% misfit | -7.49 | 175.6 |

### Launch-cost split

| K | tau_graph | tau_eager | tau_isolated | eager - graph (exposed launch) | isolated - graph |
|---|---|---|---|---|---|
| K4096 | 2.02 | 3.91 | 10.33 | 1.90 | 8.31 |
| K14336 | 5.16 | 8.24 | 16.11 | 3.08 | 10.95 |
| pooled_M1 | 2.39 | 4.54 | 11.62 | 2.15 | 9.23 |

### M sweep (graph mode): t(M) / t(1) and dispatched kernel family

| model | layer | M=2 | M=4 | M=8 | M=16 | kernels at M=2 / 16 |
|---|---|---|---|---|---|---|
| llama31_8b | down_proj | 1.00 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_proj | 1.05 | 1.02 | 1.02 | 1.03 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_up_fused | 1.02 | 1.02 | 1.03 | 1.05 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | k_proj | 0.74 | 0.75 | 0.89 | 0.88 | cutlass_sm80+splitK / cutlass_sm80 |
| llama31_8b | lm_head | 0.75 | 0.75 | 0.75 | 0.75 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | o_proj | 0.73 | 0.73 | 0.73 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | q_proj | 0.74 | 0.73 | 0.75 | 0.75 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | qkv_fused | 0.73 | 0.76 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | up_proj | 1.05 | 1.02 | 1.02 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | v_proj | 0.74 | 0.78 | 0.86 | 0.88 | cutlass_sm80+splitK / cutlass_sm80 |
| qwen25_7b | down_proj | 1.00 | 0.99 | 1.00 | 1.00 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_proj | 0.73 | 0.74 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_up_fused | 0.73 | 0.73 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | k_proj | 1.22 | 1.19 | 1.23 | 1.25 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | lm_head | 0.74 | 0.74 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | o_proj | 0.73 | 0.72 | 0.72 | 0.73 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | q_proj | 0.70 | 0.70 | 0.70 | 0.71 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | qkv_fused | 0.75 | 0.73 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | up_proj | 0.73 | 0.73 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | v_proj | 1.23 | 1.24 | 1.23 | 1.25 | cutlass_sm80+splitK / cutlass_sm80+splitK |

## Canary

- reference 2026-10-18T22:24:13+00:00: 236.39 GB/s, CV 0.67%
- per-cell canary over clean cells: median 233.68 GB/s, CV 0.87%, range 225.76 to 237.68 (n=436)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2184.0, 'median': 2184.0, 'max': 2190.0}
- max temperature: 59 C
- clock-event reasons seen: none

## Cross-session comparison

218 matching clean cells; |diff| median 0.27%, max 5.92%; 8 differ by more than 3% (listed in compare.csv).

## Plots

- [e2_bandwidth](e2_bandwidth.png)
- [e2_efficiency](e2_efficiency.png)
- [e2_msweep](e2_msweep.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
