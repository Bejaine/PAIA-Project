# Summary: e1v2_s2_20261024

Cells: 360 total; clean 360

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.12%, p90 1.08%, max 8.00%.

Clean cells flagged noisy (CV > 3%): 59. With the tail-insensitive robust CV (IQR/1.349/median): 2 above 3%, median 0.18%.

## E1 (B0) achievable bandwidth

**R_inf_B0 = 891.77 GB/s** (read/triton_read/graph; median achieved GB/s of clean, l2-safe, cold, graph-mode reads with >= 960 MiB (10x L2); best read implementation).

**Probe tau = 1.21 us** (95% CI 1.19 to 1.23), fitted R 893.0 GB/s, B_half 1.028 MiB, rms residual 0.5% (read/triton_read/graph).

### Cold plateaus (>= 10x L2)

| op/impl/mode | median GB/s | min | max | n |
|---|---|---|---|---|
| copy/torch_copy_/eager | 800.45 | 799.81 | 806.97 | 4 |
| copy/torch_copy_/graph | 809.42 | 808.80 | 814.67 | 4 |
| copy/torch_copy_/isolated | 799.21 | 795.85 | 803.51 | 4 |
| read/torch_sum/eager | 889.92 | 887.70 | 889.96 | 3 |
| read/torch_sum/graph | 889.86 | 887.66 | 889.87 | 3 |
| read/torch_sum/isolated | 885.34 | 880.21 | 886.82 | 3 |
| read/triton_read/eager | 891.51 | 890.59 | 892.02 | 3 |
| read/triton_read/graph | 891.77 | 891.14 | 892.09 | 3 |
| read/triton_read/isolated | 886.52 | 882.02 | 888.85 | 3 |
| write/torch_fill_/eager | 891.80 | 891.78 | 891.81 | 3 |
| write/torch_fill_/graph | 891.72 | 891.66 | 891.73 | 3 |
| write/torch_fill_/isolated | 889.77 | 888.74 | 890.04 | 3 |

### Fits t = tau + B / R (cold, l2-safe; weighted LS, Theil-Sen cross-check, bootstrap 95% CI)

Fits with rms relative residual above 5% are marked `misfit`: the single-tau model does not describe that op.

| op/impl/mode | tau us | tau CI95 | R_inf GB/s | R CI95 | B_half MiB | R^2 | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|---|---|
| copy/torch_copy_/eager | 3.26 | 0.62 to 4.55 | 884.2 | 813.3 to 980.0 | 2.753 | 0.99014 | 17.0% misfit | -0.53 | 802.8 |
| copy/torch_copy_/graph | 0.92 | 0.14 to 0.98 | 836.2 | 811.2 to 864.3 | 0.731 | 0.99907 | 5.1% misfit | 0.43 | 809.7 |
| copy/torch_copy_/isolated | 10.68 | 6.95 to 12.95 | 873.4 | 818.9 to 966.3 | 8.895 | 0.99204 | 14.8% misfit | 3.85 | 803.8 |
| read/torch_sum/eager | 8.23 | 6.44 to 9.50 | 941.2 | 905.4 to 1008.1 | 7.387 | 0.99657 | 11.1% misfit | 5.35 | 892.1 |
| read/torch_sum/graph | 6.00 | 5.33 to 7.47 | 902.7 | 887.5 to 939.6 | 5.169 | 0.99980 | 10.5% misfit | 5.81 | 891.5 |
| read/torch_sum/isolated | 16.86 | 15.08 to 18.03 | 920.7 | 900.6 to 959.2 | 14.802 | 0.99864 | 6.6% misfit | 13.62 | 891.4 |
| read/triton_read/eager | 7.57 | 5.17 to 9.11 | 997.1 | 921.9 to 1134.4 | 7.195 | 0.98662 | 16.6% misfit | 2.05 | 894.8 |
| read/triton_read/graph | 1.21 | 1.19 to 1.23 | 893.0 | 891.2 to 895.5 | 1.028 | 1.00000 | 0.5% | 1.22 | 892.0 |
| read/triton_read/isolated | 16.63 | 14.40 to 18.14 | 935.9 | 906.6 to 992.1 | 14.846 | 0.99728 | 8.6% misfit | 12.10 | 893.3 |
| write/torch_fill_/eager | 3.17 | 1.44 to 4.13 | 1001.7 | 915.6 to 1141.3 | 3.024 | 0.98521 | 19.0% misfit | 0.04 | 892.6 |
| write/torch_fill_/graph | 0.35 | 0.05 to 0.48 | 922.6 | 892.5 to 956.0 | 0.311 | 0.99861 | 7.8% misfit | 0.03 | 891.7 |
| write/torch_fill_/isolated | 10.51 | 7.70 to 12.27 | 1013.5 | 929.3 to 1175.8 | 10.160 | 0.98228 | 16.4% misfit | 1.77 | 898.1 |

### Hot (L2) behaviour, graph mode

| op/impl | peak GB/s | at buffer size | falls off at |
|---|---|---|---|
| copy/torch_copy_ | 4679.4 | 32 MiB | 64 MiB |
| read/torch_sum | 3996.5 | 64 MiB | 128 MiB |
| read/triton_read | 4999.0 | 64 MiB | 128 MiB |
| write/torch_fill_ | 3039.6 | 32 MiB | 128 MiB |

## Canary

- reference 2026-10-24T04:40:20+05:30: 824.73 GB/s, CV 0.07%
- per-cell canary over clean cells: median 824.92 GB/s, CV 0.18%, range 824.13 to 828.34 (n=720)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2460.0, 'median': 2670.0, 'max': 2685.0}
- max temperature: 66 C
- clock-event reasons seen: ['gpu_idle', 'sw_power_cap']

## Cross-session comparison

360 matching clean cells; |diff| median 0.09%, max 11.16%; 5 differ by more than 3% (listed in compare.csv).

## Plots

- [e1_cold_bandwidth](e1_cold_bandwidth.png)
- [e1_hot_vs_cold](e1_hot_vs_cold.png)
- [e1_fit](e1_fit_residuals.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
