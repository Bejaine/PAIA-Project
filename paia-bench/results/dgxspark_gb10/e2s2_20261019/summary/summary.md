# Summary: e2s2_20261019

Cells: 218 total; clean 218

Clean cells flagged noisy (CV > 3%): 161. With the tail-insensitive robust CV (IQR/1.349/median): 11 above 3%, median 0.43%.

## E2 (B1) cuBLAS BF16 GEMV

B0 = 242.10 GB/s (from the E1 session). `%E1` compares with the cold E1 Triton read of the same byte size and mode (bandwidth on GB10 depends slightly on footprint).

### Per real shape, M=1

| model | layer | N x K | MiB | isolated us | eager us | graph us | graph GB/s | %B0 | %E1 same size | kernel (M=1) | model eff BF16 | model eff 4-bit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | v_proj | 1024 x 4096 | 8.0 | 62.32 | 55.98 | 50.04 | 167.8 | 69.3 | 71.6 | gemvx/v6 | 94.3% | 80.7% |
| llama31_8b | k_proj | 1024 x 4096 | 8.0 | 61.41 | 52.60 | 52.41 | 160.3 | 66.2 | 68.4 | gemvx/v6 | 94.3% | 80.7% |
| llama31_8b | o_proj | 4096 x 4096 | 32.0 | 208.75 | 199.86 | 202.92 | 165.4 | 68.3 | 68.9 | gemvx/v6 | 98.5% | 94.3% |
| llama31_8b | q_proj | 4096 x 4096 | 32.0 | 208.83 | 202.63 | 198.17 | 169.4 | 70.0 | 70.5 | gemvx/v6 | 98.5% | 94.3% |
| llama31_8b | qkv_fused | 6144 x 4096 | 48.0 | 308.00 | 298.38 | 293.60 | 171.5 | 70.8 | 71.3 | gemvx/v6 | 99.0% | 96.1% |
| llama31_8b | gate_proj | 14336 x 4096 | 112.0 | 482.70 | 473.65 | 475.43 | 247.1 | 102.1 | 102.4 | gemvx/v7 | - | - |
| llama31_8b | up_proj | 14336 x 4096 | 112.0 | 480.13 | 494.10 | 476.84 | 246.4 | 101.8 | 102.1 | gemvx/v7 | - | - |
| llama31_8b | down_proj | 4096 x 14336 | 112.0 | 515.98 | 511.75 | 507.45 | 231.5 | 95.6 | 95.9 | gemvx/v7 | - | - |
| llama31_8b | gate_up_fused | 28672 x 4096 | 224.0 | 993.76 | 986.81 | 997.21 | 235.6 | 97.3 | 97.6 | gemvx/v7 | - | - |
| llama31_8b | lm_head | 128256 x 4096 | 1002.0 | 5851.34 | 5993.76 | 5982.22 | 175.7 | 72.6 | 72.9 | gemvx/v7 | - | - |
| qwen25_7b | v_proj | 512 x 3584 | 3.5 | 24.96 | 18.30 | 16.66 | 220.8 | 91.2 | 99.9 | gemvx/v7 | - | - |
| qwen25_7b | k_proj | 512 x 3584 | 3.5 | 25.57 | 18.64 | 17.45 | 210.7 | 87.0 | 95.3 | gemvx/v7 | - | - |
| qwen25_7b | o_proj | 3584 x 3584 | 24.5 | 160.58 | 151.43 | 150.91 | 170.3 | 70.4 | 71.4 | gemvx/v6 | 98.1% | 92.7% |
| qwen25_7b | q_proj | 3584 x 3584 | 24.5 | 159.71 | 155.79 | 151.11 | 170.1 | 70.3 | 71.3 | gemvx/v6 | 98.1% | 92.7% |
| qwen25_7b | qkv_fused | 4608 x 3584 | 31.5 | 206.82 | 207.59 | 191.98 | 172.1 | 71.1 | 71.7 | gemvx/v6 | 98.5% | 94.3% |
| qwen25_7b | up_proj | 18944 x 3584 | 129.5 | 778.66 | 771.53 | 798.45 | 170.1 | 70.3 | 70.4 | gemvx/v6 | 99.6% | 98.5% |
| qwen25_7b | down_proj | 3584 x 18944 | 129.5 | 596.45 | 585.51 | 577.96 | 235.0 | 97.1 | 97.3 | gemvx/v7 | - | - |
| qwen25_7b | gate_proj | 18944 x 3584 | 129.5 | 777.92 | 790.97 | 774.60 | 175.4 | 72.4 | 72.6 | gemvx/v6 | 99.6% | 98.5% |
| qwen25_7b | gate_up_fused | 37888 x 3584 | 259.0 | 1542.99 | 1532.45 | 1546.21 | 175.7 | 72.6 | 72.8 | gemvx/v6 | 99.8% | 99.3% |
| qwen25_7b | lm_head | 152064 x 3584 | 1039.5 | 6126.00 | 6118.72 | 6178.88 | 176.5 | 72.9 | 73.3 | gemvx/v7 | - | - |

Model efficiency uses the graph-mode fit of the kernel cuBLAS dispatched for that shape (only fits with <= 5% residual): eff = B / (B + B_half), i.e. the tau-only loss relative to that kernel's own R_inf. 4-bit replaces weight bytes by a quarter, activations and output unchanged (no scale metadata). The per-byte loss of a kernel shows up in %B0, not here.

cuBLAS dispatched a GEMV kernel (`gemvx`) for every real shape at M=1.

### Fits t = tau + B / R (M=1)

| fit | tau us | tau CI95 | R_inf GB/s | B_half MiB | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|
| K14336/eager | 8.11 | 7.32 to 11.67 | 227.5 | 1.759 | 8.5% misfit | 25.69 | 235.2 |
| K14336/graph | 4.53 | 3.20 to 7.25 | 227.1 | 0.982 | 10.1% misfit | 25.41 | 235.1 |
| K14336/isolated | 15.99 | 15.30 to 18.76 | 228.3 | 3.481 | 8.1% misfit | 34.62 | 234.7 |
| K4096/eager | 4.05 | 0.84 to 5.23 | 197.2 | 0.762 | 15.5% misfit | 9.23 | 178.3 |
| K4096/graph | 1.84 | -1.08 to 2.55 | 194.6 | 0.341 | 15.2% misfit | 5.46 | 173.4 |
| K4096/isolated | 10.22 | 7.67 to 11.60 | 187.1 | 1.824 | 11.6% misfit | 16.75 | 175.8 |
| kernel:cutlass_sm80+splitK/eager | 7.80 | 6.96 to 8.24 | 227.1 | 1.689 | 2.4% | 7.32 | 223.6 |
| kernel:cutlass_sm80+splitK/graph | 4.68 | 3.07 to 5.28 | 238.6 | 1.064 | 6.0% misfit | 4.36 | 236.1 |
| kernel:cutlass_sm80+splitK/isolated | 15.60 | 15.21 to 15.81 | 231.8 | 3.448 | 0.8% | 15.43 | 231.3 |
| kernel:gemvx/v6/eager | 4.94 | 3.25 to 8.97 | 172.6 | 0.813 | 2.7% | 12.35 | 176.3 |
| kernel:gemvx/v6/graph | 2.94 | 1.74 to 5.51 | 171.9 | 0.481 | 2.2% | 2.69 | 173.1 |
| kernel:gemvx/v6/isolated | 13.58 | 12.15 to 17.03 | 173.8 | 2.250 | 2.2% | 19.73 | 177.1 |
| kernel:gemvx/v7/eager | 4.44 | 2.51 to 5.30 | 230.9 | 0.978 | 11.8% misfit | 7.48 | 233.0 |
| kernel:gemvx/v7/graph | 2.28 | 0.89 to 2.75 | 230.0 | 0.500 | 11.1% misfit | 3.99 | 233.3 |
| kernel:gemvx/v7/isolated | 10.69 | 9.18 to 11.55 | 226.6 | 2.310 | 10.3% misfit | 10.04 | 232.2 |
| pooled_M1/eager | 4.56 | 3.00 to 5.79 | 205.4 | 0.893 | 15.9% misfit | -9.96 | 180.5 |
| pooled_M1/graph | 2.17 | 1.04 to 2.91 | 204.9 | 0.424 | 15.7% misfit | -15.19 | 179.9 |
| pooled_M1/isolated | 11.43 | 9.92 to 13.28 | 203.6 | 2.218 | 14.5% misfit | -7.12 | 179.8 |

### Launch-cost split

| K | tau_graph | tau_eager | tau_isolated | eager - graph (exposed launch) | isolated - graph |
|---|---|---|---|---|---|
| K4096 | 1.84 | 4.05 | 10.22 | 2.21 | 8.39 |
| K14336 | 4.53 | 8.11 | 15.99 | 3.57 | 11.46 |
| pooled_M1 | 2.17 | 4.56 | 11.43 | 2.38 | 9.26 |

### M sweep (graph mode): t(M) / t(1) and dispatched kernel family

| model | layer | M=2 | M=4 | M=8 | M=16 | kernels at M=2 / 16 |
|---|---|---|---|---|---|---|
| llama31_8b | down_proj | 1.02 | 1.02 | 1.02 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_proj | 1.12 | 1.06 | 1.07 | 1.09 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_up_fused | 1.06 | 1.02 | 1.03 | 1.04 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | k_proj | 0.86 | 0.86 | 0.83 | 0.85 | cutlass_sm80+splitK / cutlass_sm80 |
| llama31_8b | lm_head | 0.76 | 0.77 | 0.78 | 0.76 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | o_proj | 0.76 | 0.75 | 0.78 | 0.75 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | q_proj | 0.76 | 0.80 | 0.78 | 0.79 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | qkv_fused | 0.77 | 0.76 | 0.75 | 0.75 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | up_proj | 1.11 | 1.13 | 1.07 | 1.10 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | v_proj | 0.83 | 0.75 | 0.95 | 0.89 | cutlass_sm80+splitK / cutlass_sm80 |
| qwen25_7b | down_proj | 1.02 | 1.03 | 1.02 | 1.03 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_proj | 0.75 | 0.75 | 0.77 | 0.76 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_up_fused | 0.75 | 0.76 | 0.75 | 0.77 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | k_proj | 1.19 | 1.18 | 1.18 | 1.26 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | lm_head | 0.75 | 0.75 | 0.77 | 0.76 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | o_proj | 0.74 | 0.74 | 0.75 | 0.75 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | q_proj | 0.74 | 0.75 | 0.75 | 0.76 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | qkv_fused | 0.76 | 0.76 | 0.77 | 0.77 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | up_proj | 0.73 | 0.74 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | v_proj | 1.29 | 1.29 | 1.27 | 1.30 | cutlass_sm80+splitK / cutlass_sm80+splitK |

## Canary

- reference 2026-10-18T20:36:00+00:00: 244.09 GB/s, CV 0.76%
- per-cell canary over clean cells: median 241.05 GB/s, CV 1.12%, range 232.09 to 244.52 (n=436)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2184.0, 'median': 2190.0, 'max': 2190}
- max temperature: 54 C
- clock-event reasons seen: none

## Cross-session comparison

218 matching clean cells; |diff| median 2.08%, max 14.90%; 64 differ by more than 3% (listed in compare.csv).

## Plots

- [e2_bandwidth](e2_bandwidth.png)
- [e2_efficiency](e2_efficiency.png)
- [e2_msweep](e2_msweep.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
