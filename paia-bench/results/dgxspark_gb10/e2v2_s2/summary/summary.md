# Summary: e2v2_s2

Cells: 218 total; clean 218

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.45%, p90 1.29%, max 8.76%.

Clean cells flagged noisy (CV > 3%): 143. With the tail-insensitive robust CV (IQR/1.349/median): 10 above 3%, median 0.39%.

## E2 (B1) cuBLAS BF16 GEMV

B0 = 239.49 GB/s (from the E1 session). `%E1` compares with the cold E1 Triton read of the same byte size and mode (bandwidth on GB10 depends slightly on footprint).

### Per real shape, M=1

| model | layer | N x K | MiB | isolated us | eager us | graph us | graph GB/s | %B0 | %E1 same size | kernel (M=1) | model eff BF16 | model eff 4-bit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | k_proj | 1024 x 4096 | 8.0 | 61.92 | 53.18 | 48.81 | 172.1 | 71.9 | 73.8 | gemvx/v6 | 99.3% | 97.1% |
| llama31_8b | v_proj | 1024 x 4096 | 8.0 | 62.06 | 50.62 | 51.38 | 163.5 | 68.3 | 70.1 | gemvx/v6 | 99.3% | 97.1% |
| llama31_8b | o_proj | 4096 x 4096 | 32.0 | 209.89 | 203.57 | 202.49 | 165.8 | 69.2 | 69.5 | gemvx/v6 | 99.8% | 99.3% |
| llama31_8b | q_proj | 4096 x 4096 | 32.0 | 209.92 | 203.91 | 198.37 | 169.2 | 70.7 | 71.0 | gemvx/v6 | 99.8% | 99.3% |
| llama31_8b | qkv_fused | 6144 x 4096 | 48.0 | 309.12 | 300.85 | 299.35 | 168.2 | 70.2 | 70.5 | gemvx/v6 | 99.9% | 99.5% |
| llama31_8b | gate_proj | 14336 x 4096 | 112.0 | 507.68 | 502.99 | 504.13 | 233.0 | 97.3 | 97.3 | gemvx/v7 | - | - |
| llama31_8b | down_proj | 4096 x 14336 | 112.0 | 516.93 | 511.20 | 508.96 | 230.8 | 96.4 | 96.4 | gemvx/v7 | - | - |
| llama31_8b | up_proj | 14336 x 4096 | 112.0 | 509.58 | 504.59 | 501.82 | 234.1 | 97.7 | 97.8 | gemvx/v7 | - | - |
| llama31_8b | gate_up_fused | 28672 x 4096 | 224.0 | 1007.17 | 999.36 | 1000.86 | 234.7 | 98.0 | 97.9 | gemvx/v7 | - | - |
| llama31_8b | lm_head | 128256 x 4096 | 1002.0 | 6012.85 | 6005.90 | 6015.41 | 174.7 | 72.9 | 73.0 | gemvx/v7 | - | - |
| qwen25_7b | k_proj | 512 x 3584 | 3.5 | 26.30 | 19.24 | 17.62 | 208.7 | 87.1 | 93.4 | gemvx/v7 | - | - |
| qwen25_7b | v_proj | 512 x 3584 | 3.5 | 26.53 | 19.25 | 17.72 | 207.5 | 86.7 | 92.9 | gemvx/v7 | - | - |
| qwen25_7b | o_proj | 3584 x 3584 | 24.5 | 165.41 | 157.11 | 156.40 | 164.4 | 68.6 | 69.3 | gemvx/v6 | 99.8% | 99.0% |
| qwen25_7b | q_proj | 3584 x 3584 | 24.5 | 164.80 | 158.49 | 156.55 | 164.2 | 68.6 | 69.2 | gemvx/v6 | 99.8% | 99.0% |
| qwen25_7b | qkv_fused | 4608 x 3584 | 31.5 | 207.55 | 201.91 | 199.02 | 166.0 | 69.3 | 69.6 | gemvx/v6 | 99.8% | 99.2% |
| qwen25_7b | gate_proj | 18944 x 3584 | 129.5 | 809.38 | 800.49 | 799.12 | 170.0 | 71.0 | 71.0 | gemvx/v6 | 100.0% | 99.8% |
| qwen25_7b | down_proj | 3584 x 18944 | 129.5 | 597.57 | 590.85 | 585.46 | 232.0 | 96.9 | 96.9 | gemvx/v7 | - | - |
| qwen25_7b | up_proj | 18944 x 3584 | 129.5 | 810.16 | 801.38 | 801.75 | 169.4 | 70.7 | 70.7 | gemvx/v6 | 100.0% | 99.8% |
| qwen25_7b | gate_up_fused | 37888 x 3584 | 259.0 | 1590.02 | 1586.94 | 1587.72 | 171.1 | 71.4 | 71.4 | gemvx/v6 | 100.0% | 99.9% |
| qwen25_7b | lm_head | 152064 x 3584 | 1039.5 | 6288.96 | 6291.46 | 6311.09 | 172.8 | 72.1 | 72.2 | gemvx/v7 | - | - |

Model efficiency uses the graph-mode fit of the kernel cuBLAS dispatched for that shape (only fits with <= 5% residual): eff = B / (B + B_half), i.e. the tau-only loss relative to that kernel's own R_inf. 4-bit replaces weight bytes by a quarter, activations and output unchanged (no scale metadata). The per-byte loss of a kernel shows up in %B0, not here.

cuBLAS dispatched a GEMV kernel (`gemvx`) for every real shape at M=1.

### Fits t = tau + B / R (M=1)

| fit | tau us | tau CI95 | R_inf GB/s | B_half MiB | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|
| K14336/eager | 8.35 | 7.78 to 11.97 | 226.6 | 1.804 | 7.8% misfit | 26.27 | 234.5 |
| K14336/graph | 4.87 | 3.72 to 8.14 | 225.6 | 1.048 | 9.2% misfit | 24.31 | 234.4 |
| K14336/isolated | 16.35 | 15.74 to 19.28 | 228.6 | 3.565 | 7.2% misfit | 33.32 | 235.0 |
| K4096/eager | 3.94 | 0.65 to 5.14 | 191.8 | 0.721 | 15.0% misfit | 8.40 | 173.3 |
| K4096/graph | 2.01 | -0.91 to 2.62 | 192.7 | 0.369 | 14.2% misfit | 5.40 | 173.4 |
| K4096/isolated | 10.27 | 7.91 to 11.56 | 184.2 | 1.804 | 10.9% misfit | 14.84 | 172.8 |
| kernel:cutlass_sm80+splitK/eager | 7.97 | 7.56 to 8.28 | 223.3 | 1.698 | 1.2% | 7.77 | 224.0 |
| kernel:cutlass_sm80+splitK/graph | 4.82 | 3.37 to 5.44 | 230.5 | 1.059 | 4.9% | 4.24 | 225.5 |
| kernel:cutlass_sm80+splitK/isolated | 15.81 | 15.33 to 16.31 | 226.5 | 3.415 | 1.2% | 15.80 | 232.6 |
| kernel:gemvx/v6/eager | 2.58 | 1.33 to 8.05 | 168.1 | 0.414 | 2.1% | 8.46 | 171.5 |
| kernel:gemvx/v6/graph | 0.38 | -0.73 to 3.70 | 167.8 | 0.060 | 1.9% | 3.96 | 171.0 |
| kernel:gemvx/v6/isolated | 12.11 | 11.04 to 14.19 | 169.7 | 1.959 | 1.2% | 13.80 | 170.9 |
| kernel:gemvx/v7/eager | 4.44 | 2.72 to 5.28 | 226.3 | 0.959 | 11.6% misfit | 5.29 | 232.2 |
| kernel:gemvx/v7/graph | 2.45 | 1.11 to 2.84 | 225.9 | 0.527 | 10.8% misfit | 4.28 | 232.8 |
| kernel:gemvx/v7/isolated | 10.87 | 9.72 to 11.65 | 222.8 | 2.311 | 10.1% misfit | 11.53 | 232.4 |
| pooled_M1/eager | 4.55 | 3.02 to 5.87 | 202.0 | 0.878 | 15.4% misfit | -17.57 | 175.4 |
| pooled_M1/graph | 2.35 | 1.23 to 3.05 | 201.9 | 0.453 | 15.0% misfit | -21.06 | 175.0 |
| pooled_M1/isolated | 11.58 | 10.21 to 13.39 | 200.5 | 2.214 | 14.1% misfit | -11.68 | 175.2 |

### Launch-cost split

| K | tau_graph | tau_eager | tau_isolated | eager - graph (exposed launch) | isolated - graph |
|---|---|---|---|---|---|
| K4096 | 2.01 | 3.94 | 10.27 | 1.94 | 8.26 |
| K14336 | 4.87 | 8.35 | 16.35 | 3.48 | 11.48 |
| pooled_M1 | 2.35 | 4.55 | 11.58 | 2.20 | 9.23 |

### M sweep (graph mode): t(M) / t(1) and dispatched kernel family

| model | layer | M=2 | M=4 | M=8 | M=16 | kernels at M=2 / 16 |
|---|---|---|---|---|---|---|
| llama31_8b | down_proj | 1.00 | 1.01 | 1.00 | 1.01 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_proj | 1.01 | 1.01 | 1.01 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | gate_up_fused | 1.01 | 1.02 | 1.02 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | k_proj | 0.77 | 0.77 | 0.88 | 0.91 | cutlass_sm80+splitK / cutlass_sm80 |
| llama31_8b | lm_head | 0.75 | 0.75 | 0.75 | 0.75 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | o_proj | 0.73 | 0.74 | 0.73 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | q_proj | 0.75 | 0.74 | 0.75 | 0.75 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | qkv_fused | 0.74 | 0.73 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | up_proj | 1.01 | 1.02 | 1.02 | 1.02 | cutlass_sm80 / cutlass_sm80 |
| llama31_8b | v_proj | 0.77 | 0.77 | 0.85 | 0.87 | cutlass_sm80+splitK / cutlass_sm80 |
| qwen25_7b | down_proj | 1.00 | 1.00 | 1.01 | 1.01 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_proj | 0.73 | 0.73 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | gate_up_fused | 0.73 | 0.73 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | k_proj | 1.22 | 1.24 | 1.21 | 1.26 | cutlass_sm80+splitK / cutlass_sm80+splitK |
| qwen25_7b | lm_head | 0.73 | 0.74 | 0.74 | 0.75 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | o_proj | 0.72 | 0.73 | 0.72 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | q_proj | 0.72 | 0.72 | 0.72 | 0.72 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | qkv_fused | 0.73 | 0.73 | 0.74 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | up_proj | 0.73 | 0.73 | 0.73 | 0.74 | cutlass_sm80 / cutlass_sm80 |
| qwen25_7b | v_proj | 1.22 | 1.19 | 1.22 | 1.24 | cutlass_sm80+splitK / cutlass_sm80+splitK |

## Canary

- reference 2026-10-18T23:22:08+00:00: 237.11 GB/s, CV 0.78%
- per-cell canary over clean cells: median 234.14 GB/s, CV 0.80%, range 226.13 to 238.32 (n=436)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2177, 'median': 2184.0, 'max': 2190}
- max temperature: 57 C
- clock-event reasons seen: none

## Cross-session comparison

218 matching clean cells; |diff| median 0.27%, max 6.29%; 8 differ by more than 3% (listed in compare.csv).

## Plots

- [e2_bandwidth](e2_bandwidth.png)
- [e2_efficiency](e2_efficiency.png)
- [e2_msweep](e2_msweep.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
