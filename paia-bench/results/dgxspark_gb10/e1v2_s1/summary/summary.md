# Summary: e1v2_s1

Cells: 360 total; clean 360

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.38%, p90 2.85%, max 36.38%.

Clean cells flagged noisy (CV > 3%): 226. With the tail-insensitive robust CV (IQR/1.349/median): 34 above 3%, median 0.58%.

## E1 (B0) achievable bandwidth

**R_inf_B0 = 239.00 GB/s** (read/torch_sum/graph; median achieved GB/s of clean, l2-safe, cold, graph-mode reads with >= 240 MiB (10x L2); best read implementation).

**Probe tau = 1.70 us** (95% CI 1.50 to 1.80), fitted R 241.7 GB/s, B_half 0.391 MiB, rms residual 1.6% (read/triton_read/graph).

### Cold plateaus (>= 10x L2)

| op/impl/mode | median GB/s | min | max | n |
|---|---|---|---|---|
| copy/torch_copy_/eager | 228.57 | 221.21 | 236.35 | 6 |
| copy/torch_copy_/graph | 240.16 | 221.90 | 246.26 | 6 |
| copy/torch_copy_/isolated | 227.80 | 221.44 | 235.26 | 6 |
| read/torch_sum/eager | 239.19 | 238.70 | 240.04 | 5 |
| read/torch_sum/graph | 239.00 | 237.56 | 240.71 | 5 |
| read/torch_sum/isolated | 238.54 | 235.99 | 240.18 | 5 |
| read/triton_read/eager | 240.08 | 239.24 | 240.47 | 5 |
| read/triton_read/graph | 238.90 | 237.56 | 240.42 | 5 |
| read/triton_read/isolated | 237.99 | 236.85 | 238.72 | 5 |
| write/torch_fill_/eager | 196.25 | 195.92 | 196.61 | 5 |
| write/torch_fill_/graph | 196.39 | 195.77 | 196.55 | 5 |
| write/torch_fill_/isolated | 196.41 | 195.59 | 196.52 | 5 |

### Fits t = tau + B / R (cold, l2-safe; weighted LS, Theil-Sen cross-check, bootstrap 95% CI)

Fits with rms relative residual above 5% are marked `misfit`: the single-tau model does not describe that op.

| op/impl/mode | tau us | tau CI95 | R_inf GB/s | R CI95 | B_half MiB | R^2 | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|---|---|
| copy/torch_copy_/eager | 3.19 | 3.05 to 3.58 | 225.6 | 222.3 to 229.3 | 0.686 | 0.99749 | 2.2% | 11.93 | 226.9 |
| copy/torch_copy_/graph | 1.39 | 1.27 to 2.03 | 229.8 | 223.9 to 236.0 | 0.305 | 0.99410 | 3.9% | 11.42 | 228.1 |
| copy/torch_copy_/isolated | 7.91 | 6.68 to 8.31 | 227.6 | 223.9 to 231.5 | 1.716 | 0.99881 | 3.1% | 15.59 | 226.9 |
| read/torch_sum/eager | 6.93 | 5.36 to 11.79 | 245.1 | 239.4 to 261.4 | 1.619 | 0.99945 | 15.5% misfit | 6.61 | 239.9 |
| read/torch_sum/graph | 7.94 | 5.98 to 12.02 | 246.6 | 240.1 to 261.1 | 1.868 | 0.99921 | 13.4% misfit | 7.64 | 240.3 |
| read/torch_sum/isolated | 13.34 | 11.68 to 17.87 | 242.7 | 238.6 to 253.7 | 3.088 | 0.99987 | 11.0% misfit | 12.36 | 239.4 |
| read/triton_read/eager | 4.85 | 3.30 to 5.79 | 248.7 | 241.7 to 257.3 | 1.150 | 0.99839 | 7.2% misfit | -0.99 | 240.7 |
| read/triton_read/graph | 1.70 | 1.50 to 1.80 | 241.7 | 239.8 to 244.2 | 0.391 | 0.99985 | 1.6% | -2.99 | 239.7 |
| read/triton_read/isolated | 11.31 | 9.79 to 12.30 | 242.3 | 238.8 to 246.3 | 2.612 | 0.99973 | 4.5% | 9.10 | 239.0 |
| write/torch_fill_/eager | 2.99 | 2.78 to 3.47 | 195.5 | 194.3 to 197.0 | 0.558 | 0.99997 | 2.5% | 3.34 | 196.4 |
| write/torch_fill_/graph | 0.65 | 0.62 to 0.82 | 195.5 | 194.8 to 196.4 | 0.122 | 0.99997 | 0.9% | 0.82 | 196.4 |
| write/torch_fill_/isolated | 7.18 | 5.90 to 8.30 | 198.5 | 195.4 to 201.8 | 1.358 | 0.99989 | 5.0% | 7.26 | 196.6 |

### Hot (L2) behaviour, graph mode

| op/impl | peak GB/s | at buffer size | falls off at |
|---|---|---|---|
| copy/torch_copy_ | 1278.2 | 4 MiB | 32 MiB |
| read/torch_sum | 1093.9 | 16 MiB | 64 MiB |
| read/triton_read | 1407.1 | 16 MiB | 64 MiB |
| write/torch_fill_ | 762.2 | 4 MiB | 32 MiB |

## Canary

- reference 2026-10-18T21:48:02+00:00: 237.98 GB/s, CV 0.29%
- per-cell canary over clean cells: median 233.89 GB/s, CV 0.54%, range 226.33 to 237.11 (n=720)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2164.0, 'median': 2184.0, 'max': 2190.0}
- max temperature: 60 C
- clock-event reasons seen: none

## Cross-session comparison

360 matching clean cells; |diff| median 0.38%, max 24.77%; 17 differ by more than 3% (listed in compare.csv).

## Plots

- [e1_cold_bandwidth](e1_cold_bandwidth.png)
- [e1_hot_vs_cold](e1_hot_vs_cold.png)
- [e1_fit](e1_fit_residuals.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
