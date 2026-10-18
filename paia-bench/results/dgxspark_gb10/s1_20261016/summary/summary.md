# Summary: s1_20261016

Cells: 360 total; clean 360

Clean cells flagged noisy (CV > 3%): 181. With the tail-insensitive robust CV (IQR/1.349/median): 17 above 3%, median 0.40%.

## E1 (B0) achievable bandwidth

**R_inf_B0 = 241.74 GB/s** (read/torch_sum/graph; median achieved GB/s of clean, l2-safe, cold, graph-mode reads with >= 240 MiB (10x L2); best read implementation).

**Probe tau = 1.85 us** (95% CI 1.14 to 1.97), fitted R 245.3 GB/s, B_half 0.433 MiB, rms residual 2.9% (read/triton_read/graph).

### Cold plateaus (>= 10x L2)

| op/impl/mode | median GB/s | min | max | n |
|---|---|---|---|---|
| copy/torch_copy_/eager | 225.50 | 221.40 | 231.81 | 6 |
| copy/torch_copy_/graph | 238.73 | 221.91 | 244.99 | 6 |
| copy/torch_copy_/isolated | 225.31 | 220.71 | 232.24 | 6 |
| read/torch_sum/eager | 242.08 | 241.42 | 242.56 | 5 |
| read/torch_sum/graph | 241.74 | 240.84 | 242.22 | 5 |
| read/torch_sum/isolated | 241.74 | 239.84 | 242.08 | 5 |
| read/triton_read/eager | 241.02 | 239.80 | 241.79 | 5 |
| read/triton_read/graph | 240.61 | 239.35 | 241.40 | 5 |
| read/triton_read/isolated | 239.98 | 239.58 | 240.54 | 5 |
| write/torch_fill_/eager | 196.84 | 196.43 | 196.88 | 5 |
| write/torch_fill_/graph | 196.83 | 196.74 | 196.90 | 5 |
| write/torch_fill_/isolated | 196.74 | 196.00 | 196.84 | 5 |

### Fits t = tau + B / R (cold, l2-safe; weighted LS, Theil-Sen cross-check, bootstrap 95% CI)

Fits with rms relative residual above 5% are marked `misfit`: the single-tau model does not describe that op.

| op/impl/mode | tau us | tau CI95 | R_inf GB/s | R CI95 | B_half MiB | R^2 | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|---|---|
| copy/torch_copy_/eager | 3.19 | 3.11 to 3.39 | 225.0 | 223.0 to 227.3 | 0.685 | 0.99894 | 1.4% | 7.70 | 225.0 |
| copy/torch_copy_/graph | 1.33 | 1.23 to 2.01 | 229.7 | 224.5 to 235.4 | 0.291 | 0.99493 | 3.4% | 2.98 | 223.9 |
| copy/torch_copy_/isolated | 7.87 | 6.75 to 8.28 | 227.0 | 223.9 to 230.2 | 1.704 | 0.99938 | 2.7% | 12.11 | 224.3 |
| read/torch_sum/eager | 6.99 | 5.52 to 11.89 | 247.3 | 242.1 to 263.2 | 1.648 | 0.99954 | 15.3% misfit | 6.77 | 242.4 |
| read/torch_sum/graph | 7.99 | 5.95 to 11.93 | 248.6 | 242.2 to 263.4 | 1.893 | 0.99919 | 13.3% misfit | 6.99 | 242.3 |
| read/torch_sum/isolated | 13.37 | 11.76 to 17.87 | 245.4 | 241.6 to 256.3 | 3.129 | 0.99979 | 10.9% misfit | 12.86 | 242.3 |
| read/triton_read/eager | 4.74 | 3.13 to 5.84 | 249.3 | 241.9 to 258.2 | 1.127 | 0.99830 | 8.3% misfit | 2.28 | 241.8 |
| read/triton_read/graph | 1.85 | 1.14 to 1.97 | 245.3 | 241.5 to 249.2 | 0.433 | 0.99935 | 2.9% | -0.19 | 241.5 |
| read/triton_read/isolated | 11.48 | 10.02 to 12.59 | 244.7 | 241.0 to 249.0 | 2.680 | 0.99957 | 4.7% | 9.87 | 241.4 |
| write/torch_fill_/eager | 2.99 | 2.78 to 3.47 | 195.9 | 194.7 to 197.5 | 0.560 | 0.99997 | 2.5% | 3.35 | 196.9 |
| write/torch_fill_/graph | 0.22 | 0.12 to 0.86 | 193.6 | 190.4 to 197.1 | 0.041 | 0.99963 | 4.9% | 0.35 | 196.9 |
| write/torch_fill_/isolated | 7.23 | 5.88 to 8.33 | 199.0 | 195.7 to 202.4 | 1.372 | 0.99987 | 5.0% misfit | 6.92 | 197.0 |

### Hot (L2) behaviour, graph mode

| op/impl | peak GB/s | at buffer size | falls off at |
|---|---|---|---|
| copy/torch_copy_ | 1281.3 | 4 MiB | 32 MiB |
| read/torch_sum | 1041.5 | 16 MiB | 64 MiB |
| read/triton_read | 1505.0 | 16 MiB | 64 MiB |
| write/torch_fill_ | 762.4 | 4 MiB | 32 MiB |

## Canary

- reference 2026-10-16T20:02:04+00:00: 240.15 GB/s, CV 0.62%
- reference 2026-10-16T21:02:22+00:00: 240.27 GB/s, CV 0.83%, drift +0.05%
- reference 2026-10-16T22:02:33+00:00: 239.88 GB/s, CV 0.22%, drift -0.16%
- reference 2026-10-16T23:02:51+00:00: 240.23 GB/s, CV 0.53%, drift +0.15%
- per-cell canary over clean cells: median 238.03 GB/s, CV 0.78%, range 229.90 to 241.65 (n=720)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2177.0, 'median': 2190.0, 'max': 2190.0}
- max temperature: 49 C
- clock-event reasons seen: none

## Plots

- [e1_cold_bandwidth](e1_cold_bandwidth.png)
- [e1_hot_vs_cold](e1_hot_vs_cold.png)
- [e1_fit](e1_fit_residuals.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
