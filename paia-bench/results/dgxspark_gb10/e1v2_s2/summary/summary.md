# Summary: e1v2_s2

Cells: 360 total; clean 360

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.37%, p90 3.18%, max 84.70%.

Clean cells flagged noisy (CV > 3%): 229. With the tail-insensitive robust CV (IQR/1.349/median): 41 above 3%, median 0.49%.

## E1 (B0) achievable bandwidth

**R_inf_B0 = 239.49 GB/s** (read/torch_sum/graph; median achieved GB/s of clean, l2-safe, cold, graph-mode reads with >= 240 MiB (10x L2); best read implementation).

**Probe tau = 1.66 us** (95% CI 1.40 to 1.76), fitted R 241.1 GB/s, B_half 0.381 MiB, rms residual 1.6% (read/triton_read/graph).

### Cold plateaus (>= 10x L2)

| op/impl/mode | median GB/s | min | max | n |
|---|---|---|---|---|
| copy/torch_copy_/eager | 227.95 | 223.71 | 235.71 | 6 |
| copy/torch_copy_/graph | 241.80 | 224.06 | 245.87 | 6 |
| copy/torch_copy_/isolated | 227.33 | 222.45 | 235.20 | 6 |
| read/torch_sum/eager | 240.56 | 238.64 | 241.28 | 5 |
| read/torch_sum/graph | 239.49 | 238.30 | 240.69 | 5 |
| read/torch_sum/isolated | 239.64 | 237.48 | 240.54 | 5 |
| read/triton_read/eager | 239.11 | 238.83 | 239.47 | 5 |
| read/triton_read/graph | 239.05 | 238.59 | 239.82 | 5 |
| read/triton_read/isolated | 238.73 | 238.15 | 238.90 | 5 |
| write/torch_fill_/eager | 196.77 | 196.43 | 196.85 | 5 |
| write/torch_fill_/graph | 196.80 | 196.69 | 196.84 | 5 |
| write/torch_fill_/isolated | 196.74 | 196.05 | 196.80 | 5 |

### Fits t = tau + B / R (cold, l2-safe; weighted LS, Theil-Sen cross-check, bootstrap 95% CI)

Fits with rms relative residual above 5% are marked `misfit`: the single-tau model does not describe that op.

| op/impl/mode | tau us | tau CI95 | R_inf GB/s | R CI95 | B_half MiB | R^2 | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|---|---|
| copy/torch_copy_/eager | 3.26 | 3.18 to 3.68 | 225.7 | 222.8 to 228.9 | 0.701 | 0.99790 | 1.9% | 10.18 | 225.6 |
| copy/torch_copy_/graph | 1.41 | 1.29 to 2.33 | 230.0 | 224.0 to 236.4 | 0.308 | 0.99444 | 3.9% | 8.55 | 225.7 |
| copy/torch_copy_/isolated | 7.79 | 6.68 to 8.21 | 227.4 | 224.1 to 230.8 | 1.691 | 0.99874 | 2.9% | 14.17 | 226.1 |
| read/torch_sum/eager | 7.03 | 5.64 to 12.06 | 244.7 | 239.6 to 260.2 | 1.641 | 0.99974 | 15.2% misfit | 7.06 | 240.8 |
| read/torch_sum/graph | 7.90 | 6.12 to 11.98 | 246.0 | 240.0 to 260.2 | 1.854 | 0.99941 | 13.1% misfit | 7.11 | 240.3 |
| read/torch_sum/isolated | 13.36 | 11.81 to 17.91 | 243.1 | 239.1 to 253.5 | 3.097 | 0.99988 | 10.9% misfit | 13.62 | 240.4 |
| read/triton_read/eager | 4.66 | 3.08 to 5.67 | 246.8 | 239.9 to 255.4 | 1.097 | 0.99875 | 7.6% misfit | 2.76 | 239.7 |
| read/triton_read/graph | 1.66 | 1.40 to 1.76 | 241.1 | 239.5 to 243.3 | 0.381 | 0.99987 | 1.6% | 0.44 | 239.2 |
| read/triton_read/isolated | 11.43 | 10.04 to 12.43 | 243.0 | 239.6 to 247.0 | 2.648 | 0.99968 | 4.4% | 9.59 | 239.7 |
| write/torch_fill_/eager | 3.00 | 2.78 to 3.47 | 195.8 | 194.6 to 197.4 | 0.559 | 0.99996 | 2.5% | 3.36 | 196.9 |
| write/torch_fill_/graph | 0.66 | 0.63 to 0.87 | 196.0 | 195.1 to 197.0 | 0.123 | 0.99998 | 1.0% | 0.76 | 196.8 |
| write/torch_fill_/isolated | 7.21 | 6.02 to 8.27 | 198.1 | 195.4 to 200.6 | 1.363 | 0.99995 | 4.6% | 6.92 | 197.0 |

### Hot (L2) behaviour, graph mode

| op/impl | peak GB/s | at buffer size | falls off at |
|---|---|---|---|
| copy/torch_copy_ | 1275.6 | 4 MiB | 32 MiB |
| read/torch_sum | 1026.1 | 16 MiB | 64 MiB |
| read/triton_read | 1520.4 | 16 MiB | 32 MiB |
| write/torch_fill_ | 759.7 | 4 MiB | 32 MiB |

## Canary

- reference 2026-10-18T22:45:45+00:00: 237.51 GB/s, CV 0.20%
- per-cell canary over clean cells: median 234.39 GB/s, CV 0.51%, range 225.75 to 238.18 (n=720)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2164.0, 'median': 2184.0, 'max': 2190.0}
- max temperature: 59 C
- clock-event reasons seen: none

## Cross-session comparison

360 matching clean cells; |diff| median 0.39%, max 19.85%; 18 differ by more than 3% (listed in compare.csv).

## Plots

- [e1_cold_bandwidth](e1_cold_bandwidth.png)
- [e1_hot_vs_cold](e1_hot_vs_cold.png)
- [e1_fit](e1_fit_residuals.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
