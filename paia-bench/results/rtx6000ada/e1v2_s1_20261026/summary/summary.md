# Summary: e1v2_s1_20261026

Cells: 360 total; clean 360

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.12%, p90 1.12%, max 11.86%.

Clean cells flagged noisy (CV > 3%): 60. With the tail-insensitive robust CV (IQR/1.349/median): 2 above 3%, median 0.18%.

## E1 (B0) achievable bandwidth

**R_inf_B0 = 891.69 GB/s** (read/triton_read/graph; median achieved GB/s of clean, l2-safe, cold, graph-mode reads with >= 960 MiB (10x L2); best read implementation).

**Probe tau = 1.50 us** (95% CI 1.33 to 1.61), fitted R 902.7 GB/s, B_half 1.293 MiB, rms residual 2.8% (read/triton_read/graph).

### Cold plateaus (>= 10x L2)

| op/impl/mode | median GB/s | min | max | n |
|---|---|---|---|---|
| copy/torch_copy_/eager | 800.51 | 799.78 | 801.16 | 4 |
| copy/torch_copy_/graph | 809.26 | 805.94 | 813.78 | 4 |
| copy/torch_copy_/isolated | 798.85 | 796.24 | 802.20 | 4 |
| read/torch_sum/eager | 889.89 | 887.70 | 889.93 | 3 |
| read/torch_sum/graph | 889.81 | 887.59 | 889.83 | 3 |
| read/torch_sum/isolated | 885.36 | 880.35 | 886.70 | 3 |
| read/triton_read/eager | 891.37 | 890.48 | 891.91 | 3 |
| read/triton_read/graph | 891.69 | 891.07 | 892.03 | 3 |
| read/triton_read/isolated | 886.33 | 881.82 | 888.70 | 3 |
| write/torch_fill_/eager | 891.81 | 891.77 | 891.81 | 3 |
| write/torch_fill_/graph | 891.70 | 891.68 | 891.72 | 3 |
| write/torch_fill_/isolated | 889.75 | 888.81 | 890.16 | 3 |

### Fits t = tau + B / R (cold, l2-safe; weighted LS, Theil-Sen cross-check, bootstrap 95% CI)

Fits with rms relative residual above 5% are marked `misfit`: the single-tau model does not describe that op.

| op/impl/mode | tau us | tau CI95 | R_inf GB/s | R CI95 | B_half MiB | R^2 | rms rel resid | Theil-Sen tau | Theil-Sen R |
|---|---|---|---|---|---|---|---|---|---|
| copy/torch_copy_/eager | 3.20 | 0.61 to 4.46 | 882.4 | 812.7 to 976.2 | 2.694 | 0.98946 | 16.8% misfit | -0.65 | 801.4 |
| copy/torch_copy_/graph | 0.92 | 0.15 to 0.98 | 836.3 | 811.4 to 864.6 | 0.731 | 0.99900 | 5.0% | -0.88 | 809.8 |
| copy/torch_copy_/isolated | 10.40 | 6.73 to 12.64 | 870.2 | 818.1 to 957.0 | 8.635 | 0.99246 | 14.6% misfit | 3.66 | 802.5 |
| read/torch_sum/eager | 8.22 | 6.40 to 9.48 | 941.5 | 905.4 to 1008.4 | 7.377 | 0.99654 | 11.2% misfit | 5.32 | 892.1 |
| read/torch_sum/graph | 6.07 | 5.39 to 7.54 | 902.4 | 887.1 to 938.6 | 5.227 | 0.99981 | 10.3% misfit | 5.80 | 891.5 |
| read/torch_sum/isolated | 16.52 | 14.72 to 17.70 | 921.5 | 900.4 to 961.1 | 14.521 | 0.99857 | 6.8% misfit | 13.32 | 891.1 |
| read/triton_read/eager | 7.66 | 5.25 to 9.21 | 996.5 | 921.9 to 1131.6 | 7.280 | 0.98673 | 16.5% misfit | 2.16 | 894.7 |
| read/triton_read/graph | 1.50 | 1.33 to 1.61 | 902.7 | 893.1 to 914.0 | 1.293 | 0.99983 | 2.8% | 1.46 | 892.2 |
| read/triton_read/isolated | 16.71 | 14.56 to 18.15 | 933.9 | 905.9 to 987.2 | 14.886 | 0.99750 | 8.2% misfit | 12.45 | 893.2 |
| write/torch_fill_/eager | 3.03 | 1.31 to 4.01 | 1003.0 | 915.1 to 1146.3 | 2.902 | 0.98487 | 19.5% misfit | 0.03 | 892.3 |
| write/torch_fill_/graph | 0.36 | 0.06 to 0.48 | 924.5 | 894.2 to 957.9 | 0.314 | 0.99843 | 7.7% misfit | -0.25 | 891.7 |
| write/torch_fill_/isolated | 10.19 | 7.41 to 11.89 | 1014.2 | 929.2 to 1175.8 | 9.851 | 0.98211 | 16.5% misfit | 1.60 | 897.8 |

### Hot (L2) behaviour, graph mode

| op/impl | peak GB/s | at buffer size | falls off at |
|---|---|---|---|
| copy/torch_copy_ | 4712.7 | 32 MiB | 64 MiB |
| read/torch_sum | 3978.2 | 64 MiB | 128 MiB |
| read/triton_read | 4523.0 | 64 MiB | 128 MiB |
| write/torch_fill_ | 3039.9 | 32 MiB | 128 MiB |

## Canary

- reference 2026-10-26T03:01:59+05:30: 825.01 GB/s, CV 0.18%
- per-cell canary over clean cells: median 825.02 GB/s, CV 0.18%, range 824.07 to 828.52 (n=720)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2490, 'median': 2670.0, 'max': 2685}
- max temperature: 63 C
- clock-event reasons seen: ['gpu_idle', 'sw_power_cap']

## Plots

- [e1_cold_bandwidth](e1_cold_bandwidth.png)
- [e1_hot_vs_cold](e1_hot_vs_cold.png)
- [e1_fit](e1_fit_residuals.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
