# Summary: s2_20261018

Cells: 360 total; clean 360

Clean cells flagged noisy (CV > 3%): 189. With the tail-insensitive robust CV (IQR/1.349/median): 18 above 3%, median 0.41%.

## E1 (B0) achievable bandwidth

**R_inf_B0 = 242.10 GB/s** (read/torch_sum/graph; median achieved GB/s of clean, l2-safe, cold, graph-mode reads with >= 240 MiB (10x L2); best read implementation).

**Probe tau = 1.86 us** (95% CI 1.53 to 1.95), fitted R 243.5 GB/s, B_half 0.431 MiB, rms residual 1.8% (read/triton_read/graph).

### Cold plateaus (>= 10x L2)

| op/impl/mode | median GB/s | min | max | n |
|---|---|---|---|---|
| copy/torch_copy_/eager | 226.82 | 222.04 | 233.33 | 6 |
| copy/torch_copy_/graph | 239.88 | 222.19 | 245.75 | 6 |
| copy/torch_copy_/isolated | 226.33 | 221.28 | 234.95 | 6 |
| read/torch_sum/eager | 242.08 | 240.49 | 242.30 | 5 |
| read/torch_sum/graph | 242.10 | 240.58 | 242.33 | 5 |
| read/torch_sum/isolated | 241.74 | 239.00 | 242.31 | 5 |
| read/triton_read/eager | 240.73 | 240.20 | 241.83 | 5 |
| read/triton_read/graph | 240.91 | 239.79 | 241.49 | 5 |
| read/triton_read/isolated | 240.17 | 239.62 | 240.23 | 5 |
| write/torch_fill_/eager | 196.77 | 196.43 | 196.80 | 5 |
| write/torch_fill_/graph | 196.80 | 196.74 | 196.88 | 5 |
| write/torch_fill_/isolated | 196.69 | 195.95 | 196.78 | 5 |

### Fits t = tau + B / R (cold, l2-safe; weighted LS, Theil-Sen cross-check, bootstrap 95% CI)

Fits with rms relative residual above 5% are marked `misfit`: the single-tau model does not describe that op.

| op/impl/mode | tau us | tau CI95 | R_inf GB/s | R CI95 | B_half MiB | R^2 | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|---|---|
| copy/torch_copy_/eager | 3.24 | 3.17 to 3.51 | 225.3 | 222.8 to 228.1 | 0.697 | 0.99851 | 1.7% | 8.05 | 225.6 |
| copy/torch_copy_/graph | 1.37 | 1.27 to 2.10 | 230.1 | 224.7 to 236.1 | 0.301 | 0.99474 | 3.6% | 5.97 | 226.1 |
| copy/torch_copy_/isolated | 8.05 | 6.53 to 8.69 | 228.2 | 224.4 to 232.1 | 1.753 | 0.99906 | 3.6% | 10.92 | 224.9 |
| read/torch_sum/eager | 7.02 | 5.65 to 12.06 | 246.8 | 241.7 to 262.4 | 1.653 | 0.99959 | 15.3% misfit | 6.29 | 242.3 |
| read/torch_sum/graph | 7.41 | 6.00 to 11.79 | 247.5 | 242.3 to 261.8 | 1.749 | 0.99948 | 13.7% misfit | 5.91 | 242.4 |
| read/torch_sum/isolated | 13.50 | 11.93 to 18.05 | 244.8 | 240.9 to 255.0 | 3.151 | 0.99988 | 10.8% misfit | 13.69 | 242.2 |
| read/triton_read/eager | 4.70 | 3.11 to 5.75 | 249.3 | 242.3 to 257.9 | 1.117 | 0.99841 | 7.7% misfit | 2.39 | 241.5 |
| read/triton_read/graph | 1.86 | 1.53 to 1.95 | 243.5 | 241.4 to 245.7 | 0.431 | 0.99973 | 1.8% | 0.80 | 241.5 |
| read/triton_read/isolated | 11.37 | 9.80 to 12.59 | 244.4 | 240.5 to 248.9 | 2.649 | 0.99968 | 5.1% misfit | 10.09 | 240.9 |
| write/torch_fill_/eager | 3.00 | 2.78 to 3.47 | 195.9 | 194.7 to 197.4 | 0.560 | 0.99997 | 2.5% | 3.31 | 196.9 |
| write/torch_fill_/graph | 0.66 | 0.62 to 0.87 | 196.1 | 195.2 to 197.1 | 0.123 | 0.99998 | 1.1% | 0.40 | 196.8 |
| write/torch_fill_/isolated | 7.22 | 5.96 to 8.20 | 198.5 | 195.7 to 201.3 | 1.367 | 0.99991 | 4.5% | 6.86 | 196.9 |

### Hot (L2) behaviour, graph mode

| op/impl | peak GB/s | at buffer size | falls off at |
|---|---|---|---|
| copy/torch_copy_ | 1276.8 | 4 MiB | 32 MiB |
| read/torch_sum | 1024.8 | 16 MiB | 32 MiB |
| read/triton_read | 1525.5 | 16 MiB | 64 MiB |
| write/torch_fill_ | 767.0 | 4 MiB | 32 MiB |

## Canary

- reference 2026-10-18T14:58:06+00:00: 237.94 GB/s, CV 0.76%
- per-cell canary over clean cells: median 235.35 GB/s, CV 0.41%, range 226.63 to 239.00 (n=720)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2177.5, 'median': 2190.0, 'max': 2190}
- max temperature: 53 C
- clock-event reasons seen: none

## Cross-session comparison

360 matching clean cells; |diff| median 0.26%, max 29.76%; 21 differ by more than 3% (listed in compare.csv).

## Plots

- [e1_cold_bandwidth](e1_cold_bandwidth.png)
- [e1_hot_vs_cold](e1_hot_vs_cold.png)
- [e1_fit](e1_fit_residuals.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
