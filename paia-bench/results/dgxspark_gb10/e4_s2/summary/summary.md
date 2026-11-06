# Summary: e4_s2

Cells: 1146 total; clean 1146

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.68%, p90 1.97%, max 10.98%.

Clean cells flagged noisy (CV > 3%): 855. With the tail-insensitive robust CV (IQR/1.349/median): 49 above 3%, median 0.53%.

## E4 library FP8 / INT4 vs cuBLAS BF16

B0 = 239.49 GB/s. BF16 reference: E2 session(s) e2v2_s1, e2v2_s2 (mean of medians). Theoretical speedup = stored bytes ratio (weights + metadata + x + y). Identity check, max |efficiency - BW ratio| over all rows: 4.44e-16.

### Per real shape, M=1, graph mode

| model | layer | impl | kernel | graph us | BF16 us | GB/s | %B0 | S measured | S theory | efficiency | eager S | isolated S | excess us | per-op us | per-byte us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | v_proj | FP8 pt (_scaled_mm) | nvjet_64x32 | 21.71 | 51.19 | 193.5 | 80.8 | 2.36 | 2.00 | 117.9% | 2.48 | 1.74 | - | - | - |
| llama31_8b | v_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 37.30 | 51.19 | 112.7 | 47.1 | 1.37 | 2.00 | 68.7% | 1.34 | 1.20 | 19.74 | 18.51 | 1.23 |
| llama31_8b | v_proj | INT4 tinygemm | tinygemm | 12.90 | 51.19 | 173.5 | 72.5 | 3.97 | 3.75 | 105.8% | 3.55 | 3.06 | 3.55 | 3.34 | 0.22 |
| llama31_8b | v_proj | INT4 Marlin | marlin | 16.10 | 51.19 | 134.9 | 56.3 | 3.18 | 3.87 | 82.2% | 2.93 | 1.89 | 7.03 | 6.71 | 0.32 |
| llama31_8b | v_proj | FP8 pt + act quant | nvjet_64x32 | 35.74 | 51.19 | 117.6 | 49.1 | 1.43 | 2.00 | 71.7% | 1.49 | 1.31 | - | - | - |
| llama31_8b | v_proj | FP8 rw + act quant | cutlass_sm120 | 39.97 | 51.19 | 105.3 | 44.0 | 1.28 | 2.00 | 64.2% | 1.21 | 1.04 | - | - | - |
| llama31_8b | k_proj | FP8 pt (_scaled_mm) | nvjet_64x32 | 21.21 | 49.66 | 198.0 | 82.7 | 2.34 | 2.00 | 117.1% | 2.50 | 1.77 | - | - | - |
| llama31_8b | k_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 37.55 | 49.66 | 112.0 | 46.8 | 1.32 | 2.00 | 66.2% | 1.36 | 1.20 | 19.99 | 18.76 | 1.23 |
| llama31_8b | k_proj | INT4 tinygemm | tinygemm | 13.07 | 49.66 | 171.3 | 71.5 | 3.80 | 3.75 | 101.3% | 3.65 | 3.05 | 3.72 | 3.51 | 0.22 |
| llama31_8b | k_proj | INT4 Marlin | marlin | 16.09 | 49.66 | 135.1 | 56.4 | 3.09 | 3.87 | 79.9% | 2.99 | 1.85 | 7.02 | 6.69 | 0.32 |
| llama31_8b | k_proj | FP8 pt + act quant | nvjet_64x32 | 35.55 | 49.66 | 118.3 | 49.4 | 1.40 | 2.00 | 69.9% | 1.52 | 1.29 | - | - | - |
| llama31_8b | k_proj | FP8 rw + act quant | cutlass_sm120 | 40.02 | 49.66 | 105.2 | 43.9 | 1.24 | 2.00 | 62.2% | 1.23 | 1.04 | - | - | - |
| llama31_8b | o_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 82.42 | 202.69 | 203.7 | 85.1 | 2.46 | 2.00 | 123.0% | 2.46 | 2.21 | - | - | - |
| llama31_8b | o_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 82.43 | 202.69 | 203.9 | 85.1 | 2.46 | 2.00 | 123.1% | 2.41 | 2.18 | 12.26 | 7.33 | 4.92 |
| llama31_8b | o_proj | INT4 tinygemm | tinygemm | 41.01 | 202.69 | 217.7 | 90.9 | 4.94 | 3.76 | 131.5% | 4.70 | 4.35 | 3.73 | 2.87 | 0.86 |
| llama31_8b | o_proj | INT4 Marlin | marlin | 40.16 | 202.69 | 215.8 | 90.1 | 5.05 | 3.87 | 130.3% | 4.75 | 3.82 | 3.97 | 2.68 | 1.29 |
| llama31_8b | o_proj | FP8 pt + act quant | nvjet_64x64 | 95.50 | 202.69 | 175.9 | 73.4 | 2.12 | 2.00 | 106.2% | 2.17 | 2.00 | - | - | - |
| llama31_8b | o_proj | FP8 rw + act quant | cutlass_sm120 | 84.36 | 202.69 | 199.3 | 83.2 | 2.40 | 2.00 | 120.3% | 2.31 | 2.01 | - | - | - |
| llama31_8b | qkv_fused | FP8 pt (_scaled_mm) | nvjet_64x64 | 115.27 | 299.40 | 218.5 | 91.2 | 2.60 | 2.00 | 129.9% | 2.60 | 2.38 | - | - | - |
| llama31_8b | qkv_fused | FP8 rw (CUTLASS) | cutlass_sm120 | 119.61 | 299.40 | 210.7 | 88.0 | 2.50 | 2.00 | 125.3% | 2.47 | 2.30 | 14.35 | 6.97 | 7.38 |
| llama31_8b | qkv_fused | INT4 tinygemm | tinygemm | 60.57 | 299.40 | 221.0 | 92.3 | 4.94 | 3.76 | 131.4% | 4.81 | 4.55 | 4.66 | 3.37 | 1.29 |
| llama31_8b | qkv_fused | INT4 Marlin | marlin | 57.28 | 299.40 | 226.9 | 94.7 | 5.23 | 3.87 | 134.9% | 5.06 | 4.25 | 3.01 | 1.07 | 1.94 |
| llama31_8b | qkv_fused | FP8 pt + act quant | nvjet_64x64 | 131.11 | 299.40 | 192.1 | 80.2 | 2.28 | 2.00 | 114.2% | 2.34 | 2.19 | - | - | - |
| llama31_8b | qkv_fused | FP8 rw + act quant | cutlass_sm120 | 121.77 | 299.40 | 207.0 | 86.5 | 2.46 | 2.00 | 123.1% | 2.41 | 2.17 | - | - | - |
| llama31_8b | q_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 82.08 | 198.62 | 204.6 | 85.4 | 2.42 | 2.00 | 121.0% | 2.48 | 2.23 | - | - | - |
| llama31_8b | q_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 81.99 | 198.62 | 205.0 | 85.6 | 2.42 | 2.00 | 121.3% | 2.45 | 2.21 | 11.82 | 6.89 | 4.92 |
| llama31_8b | q_proj | INT4 tinygemm | tinygemm | 41.02 | 198.62 | 217.7 | 90.9 | 4.84 | 3.76 | 128.8% | 4.70 | 4.41 | 3.74 | 2.88 | 0.86 |
| llama31_8b | q_proj | INT4 Marlin | marlin | 39.88 | 198.62 | 217.3 | 90.7 | 4.98 | 3.87 | 128.6% | 4.85 | 3.86 | 3.69 | 2.40 | 1.29 |
| llama31_8b | q_proj | FP8 pt + act quant | nvjet_64x64 | 95.48 | 198.62 | 175.9 | 73.4 | 2.08 | 2.00 | 104.1% | 2.20 | 2.02 | - | - | - |
| llama31_8b | q_proj | FP8 rw + act quant | cutlass_sm120 | 84.45 | 198.62 | 199.0 | 83.1 | 2.35 | 2.00 | 117.8% | 2.33 | 2.04 | - | - | - |
| llama31_8b | down_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 264.95 | 508.59 | 221.7 | 92.6 | 1.92 | 2.00 | 96.0% | 1.92 | 1.85 | - | - | - |
| llama31_8b | down_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 271.25 | 508.59 | 216.6 | 90.5 | 1.87 | 2.00 | 93.8% | 1.87 | 1.81 | 25.90 | 8.69 | 17.21 |
| llama31_8b | down_proj | INT4 tinygemm | tinygemm | 139.60 | 508.59 | 223.7 | 93.4 | 3.64 | 3.76 | 96.9% | 3.61 | 3.52 | 9.19 | 6.18 | 3.01 |
| llama31_8b | down_proj | INT4 Marlin | marlin | 133.12 | 508.59 | 227.7 | 95.1 | 3.82 | 3.88 | 98.6% | 3.80 | 3.50 | 6.55 | 2.02 | 4.52 |
| llama31_8b | down_proj | FP8 pt + act quant | nvjet_64x64 | 297.93 | 508.59 | 197.2 | 82.3 | 1.71 | 2.00 | 85.4% | 1.71 | 1.70 | - | - | - |
| llama31_8b | down_proj | FP8 rw + act quant | cutlass_sm120 | 275.22 | 508.59 | 213.5 | 89.2 | 1.85 | 2.00 | 92.5% | 1.83 | 1.77 | - | - | - |
| llama31_8b | gate_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 269.26 | 502.30 | 218.2 | 91.1 | 1.87 | 2.00 | 93.3% | 1.89 | 1.65 | 23.94 | -23.06 | 47.00 |
| llama31_8b | gate_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 272.15 | 502.30 | 216.1 | 90.2 | 1.85 | 2.00 | 92.4% | 1.85 | 1.77 | 26.59 | 9.36 | 17.23 |
| llama31_8b | gate_proj | INT4 tinygemm | tinygemm | 137.92 | 502.30 | 226.4 | 94.6 | 3.64 | 3.76 | 96.8% | 3.59 | 3.49 | 7.51 | 4.50 | 3.01 |
| llama31_8b | gate_proj | INT4 Marlin | marlin | 133.57 | 502.30 | 227.0 | 94.8 | 3.76 | 3.88 | 97.0% | 3.70 | 3.42 | 6.99 | 2.47 | 4.52 |
| llama31_8b | gate_proj | FP8 pt + act quant | xmma_sm89 | 283.86 | 502.30 | 207.0 | 86.4 | 1.77 | 2.00 | 88.5% | 1.76 | 1.58 | - | - | - |
| llama31_8b | gate_proj | FP8 rw + act quant | cutlass_sm120 | 277.51 | 502.30 | 211.9 | 88.5 | 1.81 | 2.00 | 90.6% | 1.80 | 1.71 | - | - | - |
| llama31_8b | up_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 262.91 | 501.64 | 223.5 | 93.3 | 1.91 | 2.00 | 95.4% | 1.89 | 1.65 | 17.59 | -29.41 | 47.00 |
| llama31_8b | up_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 272.92 | 501.64 | 215.5 | 90.0 | 1.84 | 2.00 | 92.0% | 1.85 | 1.79 | 27.36 | 10.13 | 17.23 |
| llama31_8b | up_proj | INT4 tinygemm | tinygemm | 137.94 | 501.64 | 226.4 | 94.5 | 3.64 | 3.76 | 96.7% | 3.61 | 3.51 | 7.53 | 4.52 | 3.01 |
| llama31_8b | up_proj | INT4 Marlin | marlin | 133.79 | 501.64 | 226.6 | 94.6 | 3.75 | 3.88 | 96.8% | 3.73 | 3.43 | 7.21 | 2.69 | 4.52 |
| llama31_8b | up_proj | FP8 pt + act quant | xmma_sm89 | 278.93 | 501.64 | 210.7 | 88.0 | 1.80 | 2.00 | 90.0% | 1.79 | 1.58 | - | - | - |
| llama31_8b | up_proj | FP8 rw + act quant | cutlass_sm120 | 277.40 | 501.64 | 212.0 | 88.5 | 1.81 | 2.00 | 90.5% | 1.81 | 1.72 | - | - | - |
| llama31_8b | gate_up_fused | FP8 pt (_scaled_mm) | nvjet_32x32 | 499.00 | 997.70 | 235.5 | 98.3 | 2.00 | 2.00 | 100.0% | 2.03 | 1.92 | - | - | - |
| llama31_8b | gate_up_fused | FP8 rw (CUTLASS) | cutlass_sm120 | 523.50 | 997.70 | 224.7 | 93.8 | 1.91 | 2.00 | 95.4% | 1.91 | 1.86 | 32.39 | -2.06 | 34.46 |
| llama31_8b | gate_up_fused | INT4 tinygemm | tinygemm | 272.30 | 997.70 | 229.4 | 95.8 | 3.66 | 3.76 | 97.4% | 3.62 | 3.58 | 11.52 | 5.49 | 6.02 |
| llama31_8b | gate_up_fused | INT4 Marlin | marlin | 263.96 | 997.70 | 229.7 | 95.9 | 3.78 | 3.88 | 97.5% | 3.76 | 3.60 | 10.84 | 1.80 | 9.04 |
| llama31_8b | gate_up_fused | FP8 pt + act quant | nvjet_32x32 | 555.22 | 997.70 | 211.6 | 88.4 | 1.80 | 2.00 | 89.9% | 1.96 | 1.87 | - | - | - |
| llama31_8b | gate_up_fused | FP8 rw + act quant | cutlass_sm120 | 536.86 | 997.70 | 219.1 | 91.5 | 1.86 | 2.00 | 93.0% | 1.85 | 1.81 | - | - | - |
| llama31_8b | lm_head | FP8 pt (_scaled_mm) | xmma_sm89 | 2618.82 | 6010.62 | 200.7 | 83.8 | 2.30 | 2.00 | 114.8% | 2.29 | 2.29 | 424.18 | 3.72 | 420.46 |
| llama31_8b | lm_head | FP8 rw (CUTLASS) | cutlass_sm120 | 2347.73 | 6010.62 | 224.1 | 93.6 | 2.56 | 2.00 | 128.2% | 2.57 | 2.56 | 150.95 | -3.19 | 154.14 |
| llama31_8b | lm_head | INT4 tinygemm | tinygemm | 1199.95 | 6010.62 | 232.8 | 97.2 | 5.01 | 3.76 | 133.1% | 4.99 | 4.95 | 33.53 | 6.59 | 26.94 |
| llama31_8b | lm_head | INT4 Marlin | marlin | 1177.02 | 6010.62 | 230.4 | 96.2 | 5.11 | 3.88 | 131.8% | 5.08 | 5.05 | 44.87 | 4.43 | 40.44 |
| llama31_8b | lm_head | FP8 pt + act quant | xmma_sm89 | 2624.86 | 6010.62 | 200.2 | 83.6 | 2.29 | 2.00 | 114.5% | 2.30 | 2.29 | - | - | - |
| llama31_8b | lm_head | FP8 rw + act quant | cutlass_sm120 | 2365.11 | 6010.62 | 222.4 | 92.9 | 2.54 | 2.00 | 127.2% | 2.56 | 2.54 | - | - | - |
| qwen25_7b | v_proj | FP8 pt (_scaled_mm) | nvjet_32x32 | 11.37 | 17.65 | 161.8 | 67.6 | 1.55 | 2.00 | 77.6% | 1.61 | 1.04 | - | - | - |
| qwen25_7b | v_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 31.69 | 17.65 | 58.1 | 24.3 | 0.56 | 2.00 | 27.9% | 0.59 | 0.58 | 24.00 | 23.46 | 0.54 |
| qwen25_7b | v_proj | INT4 tinygemm | tinygemm | 8.22 | 17.65 | 119.5 | 49.9 | 2.15 | 3.74 | 57.4% | 1.99 | 1.75 | 4.12 | 4.02 | 0.09 |
| qwen25_7b | v_proj | INT4 Marlin | marlin | 14.43 | 17.65 | 66.1 | 27.6 | 1.22 | 3.85 | 31.7% | 1.19 | 0.83 | 10.45 | 10.31 | 0.14 |
| qwen25_7b | v_proj | FP8 pt + act quant | nvjet_32x32 | 24.67 | 17.65 | 74.7 | 31.2 | 0.72 | 2.00 | 35.8% | 0.79 | 0.71 | - | - | - |
| qwen25_7b | v_proj | FP8 rw + act quant | cutlass_sm120 | 34.31 | 17.65 | 53.8 | 22.5 | 0.51 | 1.99 | 25.8% | 0.53 | 0.49 | - | - | - |
| qwen25_7b | k_proj | FP8 pt (_scaled_mm) | nvjet_32x32 | 11.38 | 17.66 | 161.7 | 67.5 | 1.55 | 2.00 | 77.7% | 1.56 | 1.05 | - | - | - |
| qwen25_7b | k_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 31.69 | 17.66 | 58.1 | 24.3 | 0.56 | 2.00 | 27.9% | 0.59 | 0.58 | 24.00 | 23.46 | 0.54 |
| qwen25_7b | k_proj | INT4 tinygemm | tinygemm | 8.30 | 17.66 | 118.4 | 49.5 | 2.13 | 3.74 | 56.9% | 1.96 | 1.71 | 4.20 | 4.10 | 0.09 |
| qwen25_7b | k_proj | INT4 Marlin | marlin | 14.54 | 17.66 | 65.6 | 27.4 | 1.21 | 3.85 | 31.5% | 1.15 | 0.83 | 10.56 | 10.41 | 0.14 |
| qwen25_7b | k_proj | FP8 pt + act quant | nvjet_32x32 | 24.67 | 17.66 | 74.7 | 31.2 | 0.72 | 2.00 | 35.9% | 0.78 | 0.71 | - | - | - |
| qwen25_7b | k_proj | FP8 rw + act quant | cutlass_sm120 | 34.32 | 17.66 | 53.8 | 22.4 | 0.51 | 1.99 | 25.8% | 0.52 | 0.49 | - | - | - |
| qwen25_7b | qkv_fused | FP8 pt (_scaled_mm) | nvjet_64x64 | 76.12 | 198.61 | 217.1 | 90.7 | 2.61 | 2.00 | 130.5% | 2.65 | 2.31 | - | - | - |
| qwen25_7b | qkv_fused | FP8 rw (CUTLASS) | cutlass_sm120 | 80.13 | 198.61 | 206.5 | 86.2 | 2.48 | 2.00 | 124.1% | 2.46 | 2.20 | 11.04 | 6.19 | 4.85 |
| qwen25_7b | qkv_fused | INT4 tinygemm | tinygemm | 40.51 | 198.61 | 217.0 | 90.6 | 4.90 | 3.76 | 130.4% | 4.69 | 4.31 | 3.81 | 2.96 | 0.85 |
| qwen25_7b | qkv_fused | INT4 Marlin | marlin | 39.44 | 198.61 | 216.3 | 90.3 | 5.04 | 3.87 | 130.0% | 4.87 | 3.82 | 3.82 | 2.55 | 1.27 |
| qwen25_7b | qkv_fused | FP8 pt + act quant | nvjet_64x64 | 92.14 | 198.61 | 179.4 | 74.9 | 2.16 | 2.00 | 107.8% | 2.25 | 2.03 | - | - | - |
| qwen25_7b | qkv_fused | FP8 rw + act quant | cutlass_sm120 | 82.98 | 198.61 | 199.5 | 83.3 | 2.39 | 2.00 | 119.9% | 2.34 | 2.01 | - | - | - |
| qwen25_7b | q_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 65.65 | 158.15 | 195.8 | 81.8 | 2.41 | 2.00 | 120.5% | 2.41 | 2.09 | - | - | - |
| qwen25_7b | q_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 63.85 | 158.15 | 201.6 | 84.2 | 2.48 | 2.00 | 124.0% | 2.40 | 2.07 | 10.11 | 6.34 | 3.77 |
| qwen25_7b | q_proj | INT4 tinygemm | tinygemm | 32.58 | 158.15 | 209.9 | 87.6 | 4.85 | 3.76 | 129.1% | 4.59 | 4.09 | 4.03 | 3.37 | 0.66 |
| qwen25_7b | q_proj | INT4 Marlin | marlin | 31.63 | 158.15 | 209.9 | 87.6 | 5.00 | 3.87 | 129.1% | 4.75 | 3.52 | 3.91 | 2.92 | 0.99 |
| qwen25_7b | q_proj | FP8 pt + act quant | nvjet_64x64 | 77.21 | 158.15 | 166.6 | 69.5 | 2.05 | 2.00 | 102.5% | 2.05 | 1.83 | - | - | - |
| qwen25_7b | q_proj | FP8 rw + act quant | cutlass_sm120 | 66.82 | 158.15 | 192.7 | 80.5 | 2.37 | 2.00 | 118.5% | 2.27 | 1.90 | - | - | - |
| qwen25_7b | o_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 65.66 | 156.18 | 195.8 | 81.8 | 2.38 | 2.00 | 119.0% | 2.40 | 2.10 | - | - | - |
| qwen25_7b | o_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 64.06 | 156.18 | 200.9 | 83.9 | 2.44 | 2.00 | 122.1% | 2.39 | 2.10 | 10.32 | 6.55 | 3.77 |
| qwen25_7b | o_proj | INT4 tinygemm | tinygemm | 32.77 | 156.18 | 208.7 | 87.1 | 4.77 | 3.76 | 126.8% | 4.55 | 4.16 | 4.22 | 3.56 | 0.66 |
| qwen25_7b | o_proj | INT4 Marlin | marlin | 31.45 | 156.18 | 211.1 | 88.1 | 4.97 | 3.87 | 128.3% | 4.71 | 3.52 | 3.73 | 2.74 | 0.99 |
| qwen25_7b | o_proj | FP8 pt + act quant | nvjet_64x64 | 77.12 | 156.18 | 166.7 | 69.6 | 2.03 | 2.00 | 101.3% | 2.04 | 1.84 | - | - | - |
| qwen25_7b | o_proj | FP8 rw + act quant | cutlass_sm120 | 66.79 | 156.18 | 192.7 | 80.5 | 2.34 | 2.00 | 117.1% | 2.25 | 1.90 | - | - | - |
| qwen25_7b | down_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 317.07 | 587.24 | 214.2 | 89.4 | 1.85 | 2.00 | 92.6% | 1.86 | 1.81 | - | - | - |
| qwen25_7b | down_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 314.79 | 587.24 | 215.8 | 90.1 | 1.87 | 2.00 | 93.3% | 1.84 | 1.81 | 31.13 | 11.22 | 19.90 |
| qwen25_7b | down_proj | INT4 tinygemm | tinygemm | 161.86 | 587.24 | 223.1 | 93.2 | 3.63 | 3.76 | 96.5% | 3.58 | 3.51 | 11.07 | 7.59 | 3.48 |
| qwen25_7b | down_proj | INT4 Marlin | marlin | 153.77 | 587.24 | 228.0 | 95.2 | 3.82 | 3.88 | 98.6% | 3.81 | 3.54 | 7.40 | 2.18 | 5.23 |
| qwen25_7b | down_proj | FP8 pt + act quant | nvjet_64x64 | 355.51 | 587.24 | 191.1 | 79.8 | 1.65 | 2.00 | 82.6% | 1.66 | 1.65 | - | - | - |
| qwen25_7b | down_proj | FP8 rw + act quant | cutlass_sm120 | 319.60 | 587.24 | 212.6 | 88.8 | 1.84 | 2.00 | 91.9% | 1.81 | 1.76 | - | - | - |
| qwen25_7b | gate_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 315.58 | 798.83 | 215.3 | 89.9 | 2.53 | 2.00 | 126.6% | 2.62 | 2.35 | 31.91 | -22.44 | 54.35 |
| qwen25_7b | gate_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 329.70 | 798.83 | 206.3 | 86.1 | 2.42 | 2.00 | 121.3% | 2.41 | 2.34 | 45.72 | 25.79 | 19.93 |
| qwen25_7b | gate_proj | INT4 tinygemm | tinygemm | 158.38 | 798.83 | 228.0 | 95.2 | 5.04 | 3.76 | 134.1% | 4.99 | 4.87 | 7.58 | 4.10 | 3.48 |
| qwen25_7b | gate_proj | INT4 Marlin | marlin | 154.33 | 798.83 | 227.1 | 94.8 | 5.18 | 3.88 | 133.6% | 5.13 | 4.79 | 7.96 | 2.73 | 5.23 |
| qwen25_7b | gate_proj | FP8 pt + act quant | xmma_sm89 | 344.49 | 798.83 | 197.2 | 82.3 | 2.32 | 2.00 | 116.0% | 2.42 | 2.26 | - | - | - |
| qwen25_7b | gate_proj | FP8 rw + act quant | cutlass_sm120 | 329.40 | 798.83 | 206.5 | 86.2 | 2.43 | 2.00 | 121.4% | 2.41 | 2.31 | - | - | - |
| qwen25_7b | up_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 317.52 | 800.09 | 214.0 | 89.3 | 2.52 | 2.00 | 126.0% | 2.51 | 2.37 | 33.85 | -20.50 | 54.35 |
| qwen25_7b | up_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 329.86 | 800.09 | 206.2 | 86.1 | 2.43 | 2.00 | 121.4% | 2.41 | 2.34 | 45.87 | 25.95 | 19.93 |
| qwen25_7b | up_proj | INT4 tinygemm | tinygemm | 158.21 | 800.09 | 228.3 | 95.3 | 5.06 | 3.76 | 134.5% | 4.98 | 4.88 | 7.41 | 3.93 | 3.48 |
| qwen25_7b | up_proj | INT4 Marlin | marlin | 154.35 | 800.09 | 227.1 | 94.8 | 5.18 | 3.88 | 133.8% | 5.14 | 4.79 | 7.99 | 2.76 | 5.23 |
| qwen25_7b | up_proj | FP8 pt + act quant | xmma_sm89 | 328.30 | 800.09 | 206.9 | 86.4 | 2.44 | 2.00 | 121.9% | 2.40 | 2.27 | - | - | - |
| qwen25_7b | up_proj | FP8 rw + act quant | cutlass_sm120 | 329.47 | 800.09 | 206.4 | 86.2 | 2.43 | 2.00 | 121.6% | 2.41 | 2.32 | - | - | - |
| qwen25_7b | gate_up_fused | FP8 pt (_scaled_mm) | nvjet_128x72 | 592.12 | 1588.10 | 229.5 | 95.8 | 2.68 | 2.00 | 134.1% | 2.69 | 2.58 | - | - | - |
| qwen25_7b | gate_up_fused | FP8 rw (CUTLASS) | cutlass_sm120 | 608.41 | 1588.10 | 223.6 | 93.4 | 2.61 | 2.00 | 130.7% | 2.60 | 2.53 | 40.46 | 0.60 | 39.85 |
| qwen25_7b | gate_up_fused | INT4 tinygemm | tinygemm | 314.18 | 1588.10 | 229.9 | 96.0 | 5.05 | 3.76 | 134.4% | 5.00 | 4.95 | 12.62 | 5.66 | 6.96 |
| qwen25_7b | gate_up_fused | INT4 Marlin | marlin | 305.30 | 1588.10 | 229.6 | 95.9 | 5.20 | 3.88 | 134.2% | 5.17 | 4.98 | 12.60 | 2.14 | 10.46 |
| qwen25_7b | gate_up_fused | FP8 pt + act quant | nvjet_128x72 | 606.10 | 1588.10 | 224.2 | 93.6 | 2.62 | 2.00 | 131.0% | 2.62 | 2.52 | - | - | - |
| qwen25_7b | gate_up_fused | FP8 rw + act quant | cutlass_sm120 | 627.18 | 1588.10 | 216.9 | 90.6 | 2.53 | 2.00 | 126.8% | 2.59 | 2.49 | - | - | - |
| qwen25_7b | lm_head | FP8 pt (_scaled_mm) | xmma_sm89 | 2613.76 | 6305.90 | 208.6 | 87.1 | 2.41 | 2.00 | 120.7% | 2.41 | 2.39 | 336.83 | -99.39 | 436.22 |
| qwen25_7b | lm_head | FP8 rw (CUTLASS) | cutlass_sm120 | 2454.08 | 6305.90 | 222.5 | 92.9 | 2.57 | 2.00 | 128.7% | 2.56 | 2.55 | 174.62 | 14.68 | 159.94 |
| qwen25_7b | lm_head | INT4 tinygemm | tinygemm | 1240.56 | 6305.90 | 233.6 | 97.6 | 5.08 | 3.76 | 135.1% | 5.08 | 5.02 | 30.33 | 2.38 | 27.95 |
| qwen25_7b | lm_head | INT4 Marlin | marlin | 1219.02 | 6305.90 | 230.8 | 96.4 | 5.17 | 3.88 | 133.5% | 5.18 | 5.11 | 44.34 | 2.39 | 41.96 |
| qwen25_7b | lm_head | FP8 pt + act quant | xmma_sm89 | 2630.42 | 6305.90 | 207.3 | 86.6 | 2.40 | 2.00 | 119.9% | 2.40 | 2.39 | - | - | - |
| qwen25_7b | lm_head | FP8 rw + act quant | cutlass_sm120 | 2453.57 | 6305.90 | 222.5 | 92.9 | 2.57 | 2.00 | 128.7% | 2.62 | 2.59 | - | - | - |

%B0 is also the efficiency against an ideal BF16 kernel at B0 (cuBLAS BF16 itself runs at about 70% of B0 on 13 of 20 shapes, so efficiency against cuBLAS can exceed 100%). excess = t - B / B0; per-byte = B (1/R_asym - 1/B0) with the asymptotic rate of the kernel dispatched for that shape (below; '-' when that kernel has fewer than 3 ops >= 64 MiB); per-op = excess - per-byte.

### Asymptotic streaming rate per implementation (graph, M=1, ops >= 64 MiB)

| impl | R_asym GB/s | CI95 | %B0 | intercept us | points | kernels |
|---|---|---|---|---|---|---|
| INT4 Marlin (marlin) | 231.2 | 229.4 to 234.0 | 96.6 | 4.99 | 9 | marlin |
| INT4 tinygemm (tinygemm) | 234.1 | 227.6 to 237.1 | 97.7 | 18.30 | 9 | tinygemm |
| FP8 pt (_scaled_mm) (nvjet_192x112) | 198.4 | 173.6 to unbounded | 82.9 | 17.16 | 3 | nvjet_192x112 |
| FP8 pt (_scaled_mm) (xmma_sm89) | 201.0 | 198.7 to unbounded | 83.9 | -10.96 | 6 | xmma_sm89 |
| FP8 rw (CUTLASS) (cutlass_sm120) | 223.8 | 220.4 to 227.0 | 93.4 | 3.70 | 15 | cutlass_sm120 |

### Per-kernel fits t = tau + B / R (M=1; informational, misfits are expected for tiled kernels)

| impl | kernel | mode | points | tau us | tau CI95 | R_inf GB/s | R_inf %B0 | B_half MiB | rms rel resid |
|---|---|---|---|---|---|---|---|---|---|
| INT4 Marlin | marlin | eager | 44 | 14.56 | 11.92 to 18.16 | 262.8 | 109.7 | 3.649 | 20.7% misfit |
| INT4 Marlin | marlin | graph | 44 | 12.58 | 10.09 to 16.10 | 264.2 | 110.3 | 3.169 | 22.0% misfit |
| INT4 Marlin | marlin | isolated | 44 | 29.73 | 26.52 to 33.61 | 259.6 | 108.4 | 7.361 | 14.8% misfit |
| INT4 tinygemm | tinygemm | eager | 46 | 6.57 | 5.95 to 7.53 | 235.2 | 98.2 | 1.474 | 10.2% misfit |
| INT4 tinygemm | tinygemm | graph | 46 | 4.54 | 3.93 to 5.88 | 233.9 | 97.6 | 1.013 | 12.5% misfit |
| INT4 tinygemm | tinygemm | isolated | 46 | 12.38 | 11.67 to 13.25 | 235.8 | 98.5 | 2.784 | 6.4% misfit |
| FP8 pt (_scaled_mm) | nvjet_16x16 | eager | 5 | 11.85 | 11.76 to 11.92 | 6577.8 | 2746.6 | 74.312 | 0.4% |
| FP8 pt (_scaled_mm) | nvjet_16x16 | graph | 5 | 5.93 | 5.08 to 6.42 | 391.1 | 163.3 | 2.213 | 4.3% |
| FP8 pt (_scaled_mm) | nvjet_16x16 | isolated | 5 | 20.04 | 19.19 to 20.68 | 432.2 | 180.5 | 8.261 | 1.4% |
| FP8 pt (_scaled_mm) | nvjet_64x64 | eager | 12 | 5.93 | 2.81 to 9.58 | 225.0 | 93.9 | 1.273 | 2.8% |
| FP8 pt (_scaled_mm) | nvjet_64x64 | graph | 12 | 5.65 | 2.33 to 9.52 | 224.7 | 93.8 | 1.212 | 3.0% |
| FP8 pt (_scaled_mm) | nvjet_64x64 | isolated | 12 | 19.55 | 16.88 to 22.48 | 225.3 | 94.1 | 4.201 | 2.0% |
| FP8 pt (_scaled_mm) | xmma_sm89 | eager | 8 | -23.84 | -32.94 to -13.73 | 201.5 | 84.1 | -4.581 | 1.8% |
| FP8 pt (_scaled_mm) | xmma_sm89 | graph | 8 | -22.50 | -31.25 to -13.87 | 201.8 | 84.3 | -4.331 | 1.4% |
| FP8 pt (_scaled_mm) | xmma_sm89 | isolated | 8 | 17.17 | 9.05 to 26.61 | 204.7 | 85.5 | 3.351 | 1.5% |
| FP8 rw (CUTLASS) | cutlass_sm120 | eager | 46 | 30.29 | 26.22 to 35.40 | 248.8 | 103.9 | 7.187 | 25.1% misfit |
| FP8 rw (CUTLASS) | cutlass_sm120 | graph | 46 | 28.63 | 24.73 to 33.47 | 249.6 | 104.2 | 6.817 | 25.5% misfit |
| FP8 rw (CUTLASS) | cutlass_sm120 | isolated | 46 | 43.52 | 38.84 to 49.81 | 244.7 | 102.2 | 10.157 | 22.1% misfit |

### Activation quantization cost (full op minus GEMM only), M=1

| impl | mode | median us | min us | max us | shapes |
|---|---|---|---|---|---|
| FP8 pt + act quant | isolated | 12.21 | -0.69 | 33.39 | 20 |
| FP8 pt + act quant | eager | 13.64 | -13.02 | 38.79 | 20 |
| FP8 pt + act quant | graph | 14.18 | 6.05 | 56.23 | 20 |
| FP8 rw + act quant | isolated | 8.14 | -40.35 | 17.49 | 20 |
| FP8 rw + act quant | eager | 4.14 | -57.14 | 18.50 | 20 |
| FP8 rw + act quant | graph | 2.70 | -0.51 | 18.76 | 20 |

### Per-token GEMV cost (graph, M=1; all layers + lm_head)

| model | layers | impl | ms/token | GiB/token | ideal ms at B0 | % of ideal | speedup vs BF16 | theory |
|---|---|---|---|---|---|---|---|---|
| llama31_8b | unfused | cuBLAS BF16 | 70.48 | 13.98 | 62.69 | 89.0 | 1.00 | 1.00 |
| llama31_8b | unfused | FP8 pt (_scaled_mm) | 34.76 | 6.99 | 31.35 | 90.2 | 2.03 | 2.00 |
| llama31_8b | unfused | FP8 rw (CUTLASS) | 36.13 | 7.00 | 31.38 | 86.9 | 1.95 | 2.00 |
| llama31_8b | unfused | INT4 tinygemm | 17.95 | 3.72 | 16.67 | 92.9 | 3.93 | 3.76 |
| llama31_8b | unfused | INT4 Marlin | 17.58 | 3.61 | 16.18 | 92.0 | 4.01 | 3.87 |
| llama31_8b | unfused | FP8 pt + act quant | 38.56 | 6.99 | 31.36 | 81.3 | 1.83 | 2.00 |
| llama31_8b | unfused | FP8 rw + act quant | 36.89 | 7.00 | 31.38 | 85.1 | 1.91 | 2.00 |
| llama31_8b | fused | cuBLAS BF16 | 70.28 | 13.98 | 62.69 | 89.2 | 1.00 | 1.00 |
| llama31_8b | fused | FP8 pt (_scaled_mm) | 33.39 | 6.99 | 31.35 | 93.9 | 2.10 | 2.00 |
| llama31_8b | fused | FP8 rw (CUTLASS) | 34.25 | 7.00 | 31.38 | 91.6 | 2.05 | 2.00 |
| llama31_8b | fused | INT4 tinygemm | 17.63 | 3.72 | 16.67 | 94.5 | 3.99 | 3.76 |
| llama31_8b | fused | INT4 Marlin | 17.00 | 3.61 | 16.18 | 95.2 | 4.13 | 3.88 |
| llama31_8b | fused | FP8 pt + act quant | 37.18 | 6.99 | 31.36 | 84.3 | 1.89 | 2.00 |
| llama31_8b | fused | FP8 rw + act quant | 34.95 | 7.00 | 31.38 | 89.8 | 2.01 | 2.00 |
| qwen25_7b | unfused | cuBLAS BF16 | 77.31 | 13.17 | 59.07 | 76.4 | 1.00 | 1.00 |
| qwen25_7b | unfused | FP8 pt (_scaled_mm) | 33.53 | 6.59 | 29.54 | 88.1 | 2.31 | 2.00 |
| qwen25_7b | unfused | FP8 rw (CUTLASS) | 35.09 | 6.59 | 29.57 | 84.3 | 2.20 | 2.00 |
| qwen25_7b | unfused | INT4 tinygemm | 16.93 | 3.50 | 15.71 | 92.8 | 4.57 | 3.76 |
| qwen25_7b | unfused | INT4 Marlin | 16.74 | 3.40 | 15.24 | 91.0 | 4.62 | 3.87 |
| qwen25_7b | unfused | FP8 pt + act quant | 37.13 | 6.59 | 29.54 | 79.6 | 2.08 | 2.00 |
| qwen25_7b | unfused | FP8 rw + act quant | 35.51 | 6.60 | 29.57 | 83.3 | 2.18 | 2.00 |
| qwen25_7b | fused | cuBLAS BF16 | 77.15 | 13.17 | 59.06 | 76.6 | 1.00 | 1.00 |
| qwen25_7b | fused | FP8 pt (_scaled_mm) | 32.04 | 6.59 | 29.54 | 92.2 | 2.41 | 2.00 |
| qwen25_7b | fused | FP8 rw (CUTLASS) | 32.34 | 6.59 | 29.56 | 91.4 | 2.39 | 2.00 |
| qwen25_7b | fused | INT4 tinygemm | 16.62 | 3.50 | 15.70 | 94.5 | 4.64 | 3.76 |
| qwen25_7b | fused | INT4 Marlin | 16.06 | 3.40 | 15.24 | 94.9 | 4.80 | 3.88 |
| qwen25_7b | fused | FP8 pt + act quant | 34.29 | 6.59 | 29.54 | 86.1 | 2.25 | 2.00 |
| qwen25_7b | fused | FP8 rw + act quant | 33.16 | 6.59 | 29.57 | 89.2 | 2.33 | 2.00 |

### M sweep (graph): t(M) / t(1)

| impl | model | layer | M=2 | M=4 | M=8 | M=16 | kernels M=1 / 16 |
|---|---|---|---|---|---|---|---|
| FP8 pt (_scaled_mm) | llama31_8b | down_proj | 1.01 | 1.01 | 1.02 | 1.02 | nvjet_64x64 / nvjet_64x64 |
| FP8 pt (_scaled_mm) | llama31_8b | gate_proj | 0.98 | 0.99 | 1.02 | 1.03 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | llama31_8b | gate_up_fused | 0.99 | 1.01 | 1.09 | 1.12 | nvjet_32x32 / nvjet_128x160 |
| FP8 pt (_scaled_mm) | llama31_8b | k_proj | 1.01 | 1.03 | 1.00 | 1.00 | nvjet_64x32 / nvjet_64x32 |
| FP8 pt (_scaled_mm) | llama31_8b | lm_head | 0.89 | 0.90 | 0.90 | 0.91 | xmma_sm89 / nvjet_192x160 |
| FP8 pt (_scaled_mm) | llama31_8b | o_proj | 1.00 | 0.93 | 0.94 | 0.93 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt (_scaled_mm) | llama31_8b | q_proj | 1.00 | 0.94 | 0.94 | 0.94 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt (_scaled_mm) | llama31_8b | qkv_fused | 1.01 | 1.00 | 1.06 | 1.04 | nvjet_64x64 / nvjet_64x88 |
| FP8 pt (_scaled_mm) | llama31_8b | up_proj | 1.01 | 1.04 | 1.05 | 1.06 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | llama31_8b | v_proj | 0.98 | 0.98 | 1.00 | 0.97 | nvjet_64x32 / nvjet_64x32 |
| FP8 pt (_scaled_mm) | qwen25_7b | down_proj | 1.00 | 1.00 | 1.00 | 1.01 | nvjet_64x64 / nvjet_64x64 |
| FP8 pt (_scaled_mm) | qwen25_7b | gate_proj | 1.01 | 0.98 | 0.98 | 0.99 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | qwen25_7b | gate_up_fused | 1.00 | 1.01 | 1.01 | 1.05 | nvjet_128x72 / nvjet_128x96 |
| FP8 pt (_scaled_mm) | qwen25_7b | k_proj | 0.94 | 0.94 | 0.92 | 1.01 | nvjet_32x32 / nvjet_32x16 |
| FP8 pt (_scaled_mm) | qwen25_7b | lm_head | 0.92 | 0.93 | 0.94 | 0.94 | xmma_sm89 / nvjet_192x160 |
| FP8 pt (_scaled_mm) | qwen25_7b | o_proj | 1.01 | 0.90 | 0.90 | 0.89 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt (_scaled_mm) | qwen25_7b | q_proj | 1.01 | 0.90 | 0.90 | 0.91 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt (_scaled_mm) | qwen25_7b | qkv_fused | 1.02 | 1.01 | 1.01 | 1.01 | nvjet_64x64 / nvjet_96x64 |
| FP8 pt (_scaled_mm) | qwen25_7b | up_proj | 1.00 | 1.00 | 0.97 | 1.01 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | qwen25_7b | v_proj | 0.94 | 0.94 | 0.95 | 1.01 | nvjet_32x32 / nvjet_32x16 |
| FP8 rw (CUTLASS) | llama31_8b | down_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | gate_proj | 1.01 | 1.00 | 0.99 | 0.99 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | gate_up_fused | 1.01 | 1.01 | 1.00 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | k_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | lm_head | 1.00 | 1.00 | 1.01 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | o_proj | 1.00 | 1.00 | 1.01 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | q_proj | 1.01 | 1.01 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | qkv_fused | 1.02 | 1.01 | 1.00 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | up_proj | 1.00 | 0.99 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | v_proj | 1.01 | 1.01 | 1.01 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | down_proj | 1.00 | 1.00 | 1.01 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | gate_proj | 0.96 | 0.99 | 0.97 | 0.97 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | gate_up_fused | 1.01 | 1.01 | 1.01 | 1.03 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | k_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | lm_head | 0.98 | 0.98 | 0.98 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | o_proj | 1.01 | 1.00 | 1.00 | 0.99 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | q_proj | 1.01 | 1.00 | 1.00 | 0.99 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | qkv_fused | 1.01 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | up_proj | 0.96 | 0.99 | 0.97 | 0.96 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | v_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| INT4 tinygemm | llama31_8b | down_proj | 1.01 | 1.02 | 1.06 | 2.27 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | gate_proj | 1.00 | 1.01 | 1.02 | 2.33 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | gate_up_fused | 1.01 | 1.01 | 1.02 | 2.31 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | k_proj | 0.99 | 0.98 | 1.06 | 1.80 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | lm_head | 1.00 | 1.01 | 1.01 | 2.16 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | o_proj | 0.99 | 0.99 | 1.02 | 2.33 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | q_proj | 0.99 | 1.00 | 1.02 | 2.27 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | qkv_fused | 1.00 | 1.00 | 1.03 | 2.18 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | up_proj | 1.00 | 1.01 | 1.01 | 2.33 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | v_proj | 1.01 | 0.99 | 1.06 | 1.81 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | down_proj | 1.01 | 1.04 | 1.08 | 2.42 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | gate_proj | 1.00 | 1.01 | 1.02 | 2.01 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | gate_up_fused | 1.00 | 1.01 | 1.02 | 2.00 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | k_proj | 0.99 | 0.97 | 1.06 | 1.63 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | lm_head | 1.00 | 1.00 | 1.02 | 1.99 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | o_proj | 0.99 | 1.00 | 1.03 | 1.99 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | q_proj | 1.00 | 1.00 | 1.04 | 2.00 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | qkv_fused | 1.00 | 1.01 | 1.04 | 2.02 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | up_proj | 1.00 | 1.01 | 1.02 | 2.02 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | v_proj | 1.04 | 0.97 | 1.08 | 1.65 | tinygemm / tinygemm |
| INT4 Marlin | llama31_8b | down_proj | 1.00 | 1.00 | 1.01 | 1.03 | marlin / marlin |
| INT4 Marlin | llama31_8b | gate_proj | 1.01 | 1.01 | 1.02 | 1.03 | marlin / marlin |
| INT4 Marlin | llama31_8b | gate_up_fused | 1.01 | 1.01 | 1.02 | 1.04 | marlin / marlin |
| INT4 Marlin | llama31_8b | k_proj | 1.00 | 1.01 | 1.02 | 1.09 | marlin / marlin |
| INT4 Marlin | llama31_8b | lm_head | 1.00 | 1.01 | 1.02 | 1.03 | marlin / marlin |
| INT4 Marlin | llama31_8b | o_proj | 1.00 | 1.00 | 1.00 | 1.01 | marlin / marlin |
| INT4 Marlin | llama31_8b | q_proj | 1.01 | 1.01 | 1.01 | 1.02 | marlin / marlin |
| INT4 Marlin | llama31_8b | qkv_fused | 1.00 | 1.01 | 1.00 | 1.01 | marlin / marlin |
| INT4 Marlin | llama31_8b | up_proj | 1.01 | 1.00 | 1.01 | 1.03 | marlin / marlin |
| INT4 Marlin | llama31_8b | v_proj | 1.00 | 1.01 | 1.01 | 1.09 | marlin / marlin |
| INT4 Marlin | qwen25_7b | down_proj | 1.00 | 1.00 | 1.01 | 1.02 | marlin / marlin |
| INT4 Marlin | qwen25_7b | gate_proj | 1.00 | 1.01 | 1.02 | 1.04 | marlin / marlin |
| INT4 Marlin | qwen25_7b | gate_up_fused | 1.00 | 1.01 | 1.02 | 1.04 | marlin / marlin |
| INT4 Marlin | qwen25_7b | k_proj | 1.00 | 1.01 | 1.02 | 1.12 | marlin / marlin |
| INT4 Marlin | qwen25_7b | lm_head | 1.00 | 1.01 | 1.02 | 1.03 | marlin / marlin |
| INT4 Marlin | qwen25_7b | o_proj | 1.00 | 1.00 | 1.02 | 1.02 | marlin / marlin |
| INT4 Marlin | qwen25_7b | q_proj | 1.00 | 1.00 | 1.01 | 1.02 | marlin / marlin |
| INT4 Marlin | qwen25_7b | qkv_fused | 1.00 | 1.01 | 1.01 | 1.02 | marlin / marlin |
| INT4 Marlin | qwen25_7b | up_proj | 1.00 | 1.01 | 1.01 | 1.04 | marlin / marlin |
| INT4 Marlin | qwen25_7b | v_proj | 1.01 | 1.01 | 1.02 | 1.14 | marlin / marlin |
| FP8 pt + act quant | llama31_8b | down_proj | 1.00 | 1.00 | 1.01 | 1.02 | nvjet_64x64 / nvjet_64x64 |
| FP8 pt + act quant | llama31_8b | gate_proj | 1.01 | 1.02 | 1.01 | 1.02 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | llama31_8b | gate_up_fused | 1.00 | 0.94 | 1.15 | 1.10 | nvjet_32x32 / nvjet_128x160 |
| FP8 pt + act quant | llama31_8b | k_proj | 1.00 | 1.00 | 1.01 | 1.01 | nvjet_64x32 / nvjet_64x32 |
| FP8 pt + act quant | llama31_8b | lm_head | 0.90 | 0.90 | 0.91 | 0.92 | xmma_sm89 / nvjet_192x160 |
| FP8 pt + act quant | llama31_8b | o_proj | 1.00 | 0.95 | 0.96 | 0.95 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt + act quant | llama31_8b | q_proj | 1.00 | 0.95 | 0.95 | 0.95 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt + act quant | llama31_8b | qkv_fused | 1.00 | 0.98 | 1.19 | 1.11 | nvjet_64x64 / nvjet_64x88 |
| FP8 pt + act quant | llama31_8b | up_proj | 1.03 | 1.04 | 1.01 | 1.05 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | llama31_8b | v_proj | 1.00 | 1.00 | 1.00 | 1.00 | nvjet_64x32 / nvjet_64x32 |
| FP8 pt + act quant | qwen25_7b | down_proj | 1.00 | 1.01 | 1.00 | 1.02 | nvjet_64x64 / nvjet_64x64 |
| FP8 pt + act quant | qwen25_7b | gate_proj | 0.95 | 0.96 | 0.93 | 0.95 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | qwen25_7b | gate_up_fused | 1.00 | 1.01 | 1.02 | 1.05 | nvjet_128x72 / nvjet_128x96 |
| FP8 pt + act quant | qwen25_7b | k_proj | 0.97 | 0.96 | 0.97 | 0.98 | nvjet_32x32 / nvjet_32x16 |
| FP8 pt + act quant | qwen25_7b | lm_head | 0.92 | 0.93 | 0.93 | 0.94 | xmma_sm89 / nvjet_192x160 |
| FP8 pt + act quant | qwen25_7b | o_proj | 0.98 | 0.94 | 0.94 | 0.94 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt + act quant | qwen25_7b | q_proj | 0.99 | 0.93 | 0.94 | 0.94 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt + act quant | qwen25_7b | qkv_fused | 0.96 | 0.97 | 0.98 | 0.98 | nvjet_64x64 / nvjet_96x64 |
| FP8 pt + act quant | qwen25_7b | up_proj | 1.01 | 1.01 | 0.98 | 1.02 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | qwen25_7b | v_proj | 0.97 | 0.96 | 0.96 | 0.99 | nvjet_32x32 / nvjet_32x16 |
| FP8 rw + act quant | llama31_8b | down_proj | 1.00 | 1.01 | 1.02 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | gate_proj | 1.00 | 0.98 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | gate_up_fused | 1.00 | 0.99 | 0.99 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | k_proj | 1.00 | 1.00 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | lm_head | 1.00 | 1.00 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | o_proj | 1.00 | 1.01 | 1.01 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | q_proj | 1.00 | 1.01 | 1.01 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | qkv_fused | 1.00 | 1.00 | 1.02 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | up_proj | 1.00 | 0.99 | 0.98 | 0.99 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | v_proj | 1.00 | 1.00 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | down_proj | 1.00 | 1.01 | 1.01 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | gate_proj | 0.97 | 0.96 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | gate_up_fused | 0.98 | 0.98 | 0.99 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | k_proj | 0.99 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | lm_head | 0.98 | 0.98 | 0.99 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | o_proj | 0.99 | 0.99 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | q_proj | 0.99 | 0.98 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | qkv_fused | 0.99 | 0.98 | 0.99 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | up_proj | 0.97 | 0.97 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | v_proj | 1.00 | 1.00 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |

## Canary

- reference 2026-10-23T06:44:21+00:00: 236.91 GB/s, CV 0.20%
- reference 2026-10-23T07:44:31+00:00: 233.39 GB/s, CV 0.14%, drift -1.49%
- per-cell canary over clean cells: median 233.18 GB/s, CV 0.71%, range 223.18 to 236.69 (n=2292)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2171.0, 'median': 2184.0, 'max': 2184.0}
- max temperature: 65 C
- clock-event reasons seen: none

## Cross-session comparison

1146 matching clean cells; |diff| median 0.32%, max 22.76%; 35 differ by more than 3% (listed in compare.csv).

## Plots

- [e4_bandwidth](e4_bandwidth.png)
- [e4_pct_b0](e4_pct_b0.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
