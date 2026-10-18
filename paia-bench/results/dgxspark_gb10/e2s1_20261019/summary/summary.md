# Summary: e2s1_20261019

Cells: 218 total; clean 218

Clean cells flagged noisy (CV > 3%): 166. With the tail-insensitive robust CV (IQR/1.349/median): 12 above 3%, median 0.44%.

## E2 (B1) cuBLAS BF16 GEMV

B0 = 242.10 GB/s (from the E1 session). `%E1` compares with the cold E1 Triton read of the same byte size and mode (bandwidth on GB10 depends slightly on footprint).

### Per real shape, M=1

| model | layer | N x K | MiB | isolated us | eager us | graph us | graph GB/s | %B0 | %E1 same size | kernel (M=1) | model eff BF16 | model eff 4-bit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | k_proj | 1024 x 4096 | 8.0 | 59.42 | 56.32 | 50.26 | 167.1 | 69.0 | 71.3 | gemvx/v6 | 96.2% | 86.4% |
| llama31_8b | v_proj | 1024 x 4096 | 8.0 | 61.23 | 53.49 | 52.19 | 160.9 | 66.5 | 68.7 | gemvx/v6 | 96.2% | 86.4% |
| llama31_8b | o_proj | 4096 x 4096 | 32.0 | 209.49 | 202.11 | 211.79 | 158.5 | 65.5 | 66.0 | gemvx/v6 | 99.0% | 96.2% |
| llama31_8b | q_proj | 4096 x 4096 | 32.0 | 209.09 | 200.43 | 200.33 | 167.6 | 69.2 | 69.8 | gemvx/v6 | 99.0% | 96.2% |
| llama31_8b | qkv_fused | 6144 x 4096 | 48.0 | 297.70 | 295.58 | 291.25 | 172.9 | 71.4 | 71.8 | gemvx/v6 | 99.3% | 97.4% |
| llama31_8b | up_proj | 14336 x 4096 | 112.0 | 506.66 | 489.40 | 488.42 | 240.5 | 99.3 | 99.6 | gemvx/v7 | - | - |
| llama31_8b | gate_proj | 14336 x 4096 | 112.0 | 504.51 | 488.56 | 488.67 | 240.4 | 99.3 | 99.6 | gemvx/v7 | - | - |
| llama31_8b | down_proj | 4096 x 14336 | 112.0 | 508.77 | 499.97 | 501.47 | 234.3 | 96.8 | 97.0 | gemvx/v7 | - | - |
| llama31_8b | gate_up_fused | 28672 x 4096 | 224.0 | 944.62 | 936.33 | 934.66 | 251.4 | 103.8 | 104.1 | gemvx/v7 | - | - |
| llama31_8b | lm_head | 128256 x 4096 | 1002.0 | 5968.29 | 5822.66 | 5802.08 | 181.1 | 74.8 | 75.2 | gemvx/v7 | - | - |
| qwen25_7b | k_proj | 512 x 3584 | 3.5 | 26.38 | 19.91 | 18.12 | 203.0 | 83.8 | 91.8 | gemvx/v7 | - | - |
| qwen25_7b | v_proj | 512 x 3584 | 3.5 | 25.57 | 18.91 | 17.24 | 213.4 | 88.1 | 96.5 | gemvx/v7 | - | - |
| qwen25_7b | o_proj | 3584 x 3584 | 24.5 | 161.50 | 156.57 | 154.86 | 166.0 | 68.6 | 69.6 | gemvx/v6 | 98.7% | 95.1% |
| qwen25_7b | q_proj | 3584 x 3584 | 24.5 | 163.73 | 152.84 | 155.39 | 165.4 | 68.3 | 69.3 | gemvx/v6 | 98.7% | 95.1% |
| qwen25_7b | qkv_fused | 4608 x 3584 | 31.5 | 201.57 | 200.11 | 197.23 | 167.6 | 69.2 | 69.8 | gemvx/v6 | 99.0% | 96.2% |
| qwen25_7b | up_proj | 18944 x 3584 | 129.5 | 800.46 | 793.29 | 772.72 | 175.8 | 72.6 | 72.8 | gemvx/v6 | 99.8% | 99.0% |
| qwen25_7b | gate_proj | 18944 x 3584 | 129.5 | 801.42 | 773.56 | 790.90 | 171.7 | 70.9 | 71.1 | gemvx/v6 | 99.8% | 99.0% |
| qwen25_7b | down_proj | 3584 x 18944 | 129.5 | 585.30 | 576.16 | 582.34 | 233.3 | 96.3 | 96.6 | gemvx/v7 | - | - |
| qwen25_7b | gate_up_fused | 37888 x 3584 | 259.0 | 1596.02 | 1574.97 | 1640.43 | 165.6 | 68.4 | 68.6 | gemvx/v6 | 99.9% | 99.5% |
| qwen25_7b | lm_head | 152064 x 3584 | 1039.5 | 6295.17 | 6295.58 | 6015.15 | 181.3 | 74.9 | 75.2 | gemvx/v7 | - | - |

Model efficiency uses the graph-mode fit of the kernel cuBLAS dispatched for that shape (only fits with <= 5% residual): eff = B / (B + B_half), i.e. the tau-only loss relative to that kernel's own R_inf. 4-bit replaces weight bytes by a quarter, activations and output unchanged (no scale metadata). The per-byte loss of a kernel shows up in %B0, not here.

cuBLAS dispatched a GEMV kernel (`gemvx`) for every real shape at M=1.

### Fits t = tau + B / R (M=1)

| fit | tau us | tau CI95 | R_inf GB/s | B_half MiB | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|
| K14336/eager | 8.13 | 7.40 to 10.21 | 229.0 | 1.775 | 8.4% misfit | 27.57 | 236.3 |
| K14336/graph | 4.97 | 4.09 to 6.18 | 229.2 | 1.087 | 8.7% misfit | 24.97 | 237.1 |
| K14336/isolated | 16.04 | 15.45 to 18.57 | 230.2 | 3.522 | 7.4% misfit | 33.17 | 236.4 |
| K4096/eager | 3.91 | 0.36 to 5.08 | 194.2 | 0.724 | 15.7% misfit | 7.91 | 175.1 |
| K4096/graph | 2.04 | -0.66 to 2.63 | 195.9 | 0.381 | 15.2% misfit | 7.54 | 175.5 |
| K4096/isolated | 10.30 | 7.45 to 11.75 | 189.4 | 1.860 | 13.1% misfit | 14.63 | 174.3 |
| kernel:cutlass_sm80+splitK/eager | 8.16 | 7.58 to 8.59 | 238.9 | 1.859 | 1.7% | 7.84 | 239.0 |
| kernel:cutlass_sm80+splitK/graph | 5.27 | 4.64 to 5.66 | 245.8 | 1.235 | 2.1% | 4.85 | 243.4 |
| kernel:cutlass_sm80+splitK/isolated | 15.52 | 15.31 to 15.67 | 227.9 | 3.373 | 0.4% | 15.36 | 227.0 |
| kernel:gemvx/v6/eager | 4.81 | 3.21 to 8.25 | 171.6 | 0.787 | 2.1% | 8.19 | 173.4 |
| kernel:gemvx/v6/graph | 1.95 | 0.57 to 6.28 | 169.9 | 0.316 | 2.6% | 6.48 | 173.2 |
| kernel:gemvx/v6/isolated | 12.27 | 11.16 to 14.13 | 171.9 | 2.010 | 1.5% | 12.11 | 172.4 |
| kernel:gemvx/v7/eager | 4.46 | 2.68 to 5.28 | 230.4 | 0.981 | 12.0% misfit | -1.22 | 234.4 |
| kernel:gemvx/v7/graph | 2.49 | 1.34 to 2.85 | 230.4 | 0.547 | 10.8% misfit | 0.43 | 234.5 |
| kernel:gemvx/v7/isolated | 10.82 | 9.28 to 11.65 | 227.4 | 2.346 | 10.7% misfit | 8.75 | 234.9 |
| pooled_M1/eager | 4.57 | 3.08 to 5.84 | 205.1 | 0.894 | 16.0% misfit | -17.50 | 177.4 |
| pooled_M1/graph | 2.41 | 1.33 to 3.13 | 205.3 | 0.471 | 15.8% misfit | -8.11 | 181.2 |
| pooled_M1/isolated | 11.52 | 10.00 to 13.30 | 204.0 | 2.241 | 14.6% misfit | -9.91 | 177.1 |

### Launch-cost split

| K | tau_graph | tau_eager | tau_isolated | eager - graph (exposed launch) | isolated - graph |
|---|---|---|---|---|---|
| K4096 | 2.04 | 3.91 | 10.30 | 1.87 | 8.26 |
| K14336 | 4.97 | 8.13 | 16.04 | 3.16 | 11.07 |
| pooled_M1 | 2.41 | 4.57 | 11.52 | 2.16 | 9.11 |

### M sweep (graph mode): t(M) / t(1) and dispatched kernel family

| model | layer | M=2 | M=4 | M=8 | M=16 | kernels at M=2 / 16 |
|---|---|---|---|---|---|---|
| llama31_8b | down_proj | 1.00 | 1.03 | 1.01 | 1.01 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_proj | 1.03 | 1.06 | 1.10 | 1.05 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_up_fused | 1.07 | 1.12 | 1.08 | 1.09 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | k_proj | 0.80 | 0.78 | 0.87 | 0.91 | cutlass_sm80+splitK / cutlass_sm80 |
| llama31_8b | lm_head | 0.77 | 0.77 | 0.78 | 0.81 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | o_proj | 0.69 | 0.69 | 0.69 | 0.82 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | q_proj | 0.74 | 0.73 | 0.73 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | qkv_fused | 0.74 | 0.75 | 0.80 | 0.77 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | up_proj | 1.03 | 1.03 | 1.06 | 1.04 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | v_proj | 0.70 | 0.83 | 0.87 | 0.88 | cutlass_sm80+splitK / cutlass_sm80 |
| qwen25_7b | down_proj | 1.00 | 1.00 | 1.00 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_proj | 0.72 | 0.74 | 0.73 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_up_fused | 0.70 | 0.70 | 0.71 | 0.72 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | k_proj | 1.18 | 1.17 | 1.16 | 1.16 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | lm_head | 0.77 | 0.77 | 0.77 | 0.78 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | o_proj | 0.72 | 0.72 | 0.72 | 0.72 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | q_proj | 0.72 | 0.72 | 0.72 | 0.73 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | qkv_fused | 0.72 | 0.73 | 0.73 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | up_proj | 0.74 | 0.75 | 0.75 | 0.76 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | v_proj | 1.24 | 1.22 | 1.19 | 1.28 | cutlass_sm80+splitK / cutlass_sm80+splitK |

## Canary

- reference 2026-10-18T20:14:33+00:00: 237.59 GB/s, CV 0.36%
- per-cell canary over clean cells: median 234.56 GB/s, CV 1.46%, range 225.72 to 244.53 (n=436)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2184.0, 'median': 2190.0, 'max': 2190}
- max temperature: 61 C
- clock-event reasons seen: none

## Cross-session comparison

218 matching clean cells; |diff| median 2.08%, max 15.12%; 63 differ by more than 3% (listed in compare.csv).

## Plots

- [e2_bandwidth](e2_bandwidth.png)
- [e2_efficiency](e2_efficiency.png)
- [e2_msweep](e2_msweep.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
