# Summary: e1v2_s1_20261023

Cells: 360 total; clean 320, unclean_exhausted 40

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.14%, p90 1.33%, max 149.76%.

Clean cells flagged noisy (CV > 3%): 63. With the tail-insensitive robust CV (IQR/1.349/median): 9 above 3%, median 0.18%.

## E1 (B0) achievable bandwidth

**R_inf_B0 = 891.19 GB/s** (read/triton_read/graph; median achieved GB/s of clean, l2-safe, cold, graph-mode reads with >= 960 MiB (10x L2); best read implementation).

**Probe tau = 1.31 us** (95% CI 1.26 to 1.38), fitted R 895.4 GB/s, B_half 1.121 MiB, rms residual 1.6% (read/triton_read/graph).

### Cold plateaus (>= 10x L2)

| op/impl/mode | median GB/s | min | max | n |
|---|---|---|---|---|
| copy/torch_copy_/eager | 799.70 | 799.70 | 799.70 | 1 |
| copy/torch_copy_/graph | 807.85 | 807.85 | 807.85 | 1 |
| copy/torch_copy_/isolated | 797.08 | 796.27 | 797.89 | 2 |
| read/torch_sum/eager | 887.73 | 887.73 | 887.73 | 1 |
| read/torch_sum/graph | 888.68 | 887.55 | 889.81 | 2 |
| read/torch_sum/isolated | 880.16 | 880.16 | 880.16 | 1 |
| read/triton_read/eager | 890.60 | 890.60 | 890.60 | 1 |
| read/triton_read/graph | 891.19 | 891.19 | 891.19 | 1 |
| write/torch_fill_/eager | 891.81 | 891.81 | 891.81 | 1 |
| write/torch_fill_/graph | 891.69 | 891.67 | 891.71 | 2 |
| write/torch_fill_/isolated | 888.73 | 888.73 | 888.73 | 1 |

### Fits t = tau + B / R (cold, l2-safe; weighted LS, Theil-Sen cross-check, bootstrap 95% CI)

Fits with rms relative residual above 5% are marked `misfit`: the single-tau model does not describe that op.

| op/impl/mode | tau us | tau CI95 | R_inf GB/s | R CI95 | B_half MiB | R^2 | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|---|---|
| copy/torch_copy_/eager | 3.38 | 0.75 to 4.58 | 921.5 | 823.1 to 1083.4 | 2.970 | 0.97814 | 18.3% misfit | -0.51 | 804.7 |
| copy/torch_copy_/graph | 0.93 | 0.23 to 1.00 | 845.2 | 812.6 to 886.0 | 0.752 | 0.99736 | 5.6% misfit | -0.07 | 808.1 |
| copy/torch_copy_/isolated | 10.83 | 7.20 to 12.95 | 894.2 | 826.5 to 1015.6 | 9.239 | 0.98624 | 15.2% misfit | 3.36 | 805.7 |
| read/torch_sum/eager | 8.19 | 6.43 to 9.46 | 958.2 | 910.7 to 1053.5 | 7.480 | 0.99428 | 12.0% misfit | 5.41 | 895.3 |
| read/torch_sum/graph | 6.04 | 5.37 to 7.62 | 903.3 | 886.2 to 944.7 | 5.202 | 0.99980 | 10.8% misfit | 5.84 | 891.7 |
| read/torch_sum/isolated | 17.41 | 15.65 to 18.73 | 934.9 | 906.3 to 1004.0 | 15.523 | 0.99749 | 7.2% misfit | 13.89 | 896.2 |
| read/triton_read/eager | 7.61 | 5.30 to 9.01 | 1032.8 | 934.9 to 1240.4 | 7.492 | 0.97796 | 17.1% misfit | 2.11 | 905.8 |
| read/triton_read/graph | 1.31 | 1.26 to 1.38 | 895.4 | 891.0 to 900.9 | 1.121 | 0.99998 | 1.6% | 1.24 | 892.1 |
| read/triton_read/isolated | 16.74 | 14.74 to 18.15 | 968.3 | 922.2 to 1113.6 | 15.459 | 0.99335 | 8.9% misfit | 12.44 | 924.8 |
| write/torch_fill_/eager | 3.15 | 1.43 to 4.08 | 1036.0 | 924.9 to 1239.2 | 3.113 | 0.97608 | 20.2% misfit | 0.09 | 896.3 |
| write/torch_fill_/graph | 0.36 | 0.05 to 0.49 | 925.8 | 892.7 to 963.7 | 0.318 | 0.99829 | 8.0% misfit | -0.11 | 891.7 |
| write/torch_fill_/isolated | 10.74 | 7.99 to 12.44 | 1060.1 | 944.6 to 1327.1 | 10.858 | 0.97035 | 16.8% misfit | 2.06 | 923.5 |

### Hot (L2) behaviour, graph mode

| op/impl | peak GB/s | at buffer size | falls off at |
|---|---|---|---|
| copy/torch_copy_ | 4716.5 | 32 MiB | 64 MiB |
| read/torch_sum | 3922.0 | 64 MiB | 128 MiB |
| read/triton_read | 4277.0 | 64 MiB | 128 MiB |
| write/torch_fill_ | 3040.6 | 32 MiB | 128 MiB |

## Canary

- reference 2026-10-23T11:27:48+05:30: 825.20 GB/s, CV 0.19%
- reference 2026-10-23T14:08:35+05:30: 825.00 GB/s, CV 0.02%, drift -0.02%
- reference 2026-10-23T16:43:35+05:30: 825.21 GB/s, CV 0.02%, drift +0.03%
- per-cell canary over clean cells: median 825.16 GB/s, CV 0.17%, range 824.09 to 828.42 (n=640)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2475, 'median': 2655.0, 'max': 2685.0}
- max temperature: 74 C
- clock-event reasons seen: ['gpu_idle', 'sw_power_cap']

## Plots

- [e1_cold_bandwidth](e1_cold_bandwidth.png)
- [e1_hot_vs_cold](e1_hot_vs_cold.png)
- [e1_fit](e1_fit_residuals.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
