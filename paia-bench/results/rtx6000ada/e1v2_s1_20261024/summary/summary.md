# Summary: e1v2_s1_20261024

Cells: 360 total; clean 360

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.10%, p90 1.28%, max 6.92%.

Clean cells flagged noisy (CV > 3%): 59. With the tail-insensitive robust CV (IQR/1.349/median): 3 above 3%, median 0.19%.

## E1 (B0) achievable bandwidth

**R_inf_B0 = 891.73 GB/s** (read/triton_read/graph; median achieved GB/s of clean, l2-safe, cold, graph-mode reads with >= 960 MiB (10x L2); best read implementation).

**Probe tau = 1.31 us** (95% CI 1.21 to 1.36), fitted R 895.6 GB/s, B_half 1.115 MiB, rms residual 1.6% (read/triton_read/graph).

### Cold plateaus (>= 10x L2)

| op/impl/mode | median GB/s | min | max | n |
|---|---|---|---|---|
| copy/torch_copy_/eager | 800.98 | 800.02 | 804.64 | 4 |
| copy/torch_copy_/graph | 809.48 | 807.45 | 814.64 | 4 |
| copy/torch_copy_/isolated | 798.54 | 796.13 | 806.43 | 4 |
| read/torch_sum/eager | 889.91 | 887.74 | 889.96 | 3 |
| read/torch_sum/graph | 889.84 | 887.58 | 889.86 | 3 |
| read/torch_sum/isolated | 885.14 | 880.04 | 886.64 | 3 |
| read/triton_read/eager | 891.44 | 890.59 | 892.05 | 3 |
| read/triton_read/graph | 891.73 | 891.16 | 892.15 | 3 |
| read/triton_read/isolated | 886.31 | 881.88 | 888.78 | 3 |
| write/torch_fill_/eager | 891.79 | 891.79 | 891.81 | 3 |
| write/torch_fill_/graph | 891.68 | 891.68 | 891.71 | 3 |
| write/torch_fill_/isolated | 889.74 | 888.67 | 890.00 | 3 |

### Fits t = tau + B / R (cold, l2-safe; weighted LS, Theil-Sen cross-check, bootstrap 95% CI)

Fits with rms relative residual above 5% are marked `misfit`: the single-tau model does not describe that op.

| op/impl/mode | tau us | tau CI95 | R_inf GB/s | R CI95 | B_half MiB | R^2 | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|---|---|
| copy/torch_copy_/eager | 3.28 | 0.63 to 4.56 | 883.8 | 813.2 to 978.7 | 2.762 | 0.98985 | 17.0% misfit | -0.55 | 802.6 |
| copy/torch_copy_/graph | 0.92 | 0.16 to 0.98 | 835.5 | 811.0 to 863.4 | 0.734 | 0.99913 | 5.0% | -0.66 | 810.5 |
| copy/torch_copy_/isolated | 10.82 | 7.04 to 13.14 | 873.6 | 819.3 to 966.0 | 9.016 | 0.99243 | 14.9% misfit | 4.41 | 806.7 |
| read/torch_sum/eager | 8.12 | 6.31 to 9.42 | 940.5 | 904.9 to 1007.3 | 7.285 | 0.99666 | 11.3% misfit | 5.35 | 892.1 |
| read/torch_sum/graph | 6.04 | 5.33 to 7.46 | 903.4 | 887.4 to 940.8 | 5.204 | 0.99977 | 10.4% misfit | 5.81 | 891.5 |
| read/torch_sum/isolated | 17.11 | 15.41 to 18.24 | 919.6 | 900.3 to 957.0 | 15.007 | 0.99873 | 6.3% misfit | 13.94 | 891.4 |
| read/triton_read/eager | 7.52 | 5.16 to 9.04 | 995.3 | 921.5 to 1128.6 | 7.142 | 0.98704 | 16.4% misfit | 2.12 | 894.5 |
| read/triton_read/graph | 1.31 | 1.21 to 1.36 | 895.6 | 890.4 to 900.7 | 1.115 | 0.99998 | 1.6% | 1.37 | 892.0 |
| read/triton_read/isolated | 16.86 | 14.60 to 18.38 | 935.1 | 906.3 to 990.0 | 15.035 | 0.99738 | 8.5% misfit | 12.45 | 893.1 |
| write/torch_fill_/eager | 3.10 | 1.37 to 4.06 | 1002.2 | 915.4 to 1144.2 | 2.959 | 0.98508 | 19.2% misfit | 0.04 | 892.5 |
| write/torch_fill_/graph | 0.35 | 0.05 to 0.48 | 922.6 | 892.5 to 956.1 | 0.311 | 0.99861 | 7.8% misfit | 0.03 | 891.7 |
| write/torch_fill_/isolated | 10.63 | 7.82 to 12.41 | 1013.6 | 929.3 to 1177.5 | 10.280 | 0.98227 | 16.4% misfit | 1.84 | 898.1 |

### Hot (L2) behaviour, graph mode

| op/impl | peak GB/s | at buffer size | falls off at |
|---|---|---|---|
| copy/torch_copy_ | 4684.4 | 32 MiB | 64 MiB |
| read/torch_sum | 3981.4 | 64 MiB | 128 MiB |
| read/triton_read | 5026.3 | 64 MiB | 128 MiB |
| write/torch_fill_ | 3040.7 | 32 MiB | 128 MiB |

## Canary

- reference 2026-10-24T03:19:57+05:30: 824.83 GB/s, CV 0.02%
- per-cell canary over clean cells: median 824.86 GB/s, CV 0.18%, range 823.91 to 828.18 (n=720)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2460.0, 'median': 2670.0, 'max': 2685}
- max temperature: 63 C
- clock-event reasons seen: ['gpu_idle', 'sw_power_cap']

## Plots

- [e1_cold_bandwidth](e1_cold_bandwidth.png)
- [e1_hot_vs_cold](e1_hot_vs_cold.png)
- [e1_fit](e1_fit_residuals.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
