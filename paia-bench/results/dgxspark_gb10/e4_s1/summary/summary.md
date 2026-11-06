# Summary: e4_s1

Cells: 1146 total; clean 1146

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.71%, p90 1.97%, max 16.98%.

Clean cells flagged noisy (CV > 3%): 862. With the tail-insensitive robust CV (IQR/1.349/median): 57 above 3%, median 0.53%.

## E4 library FP8 / INT4 vs cuBLAS BF16

B0 = 239.49 GB/s. BF16 reference: E2 session(s) e2v2_s1, e2v2_s2 (mean of medians). Theoretical speedup = stored bytes ratio (weights + metadata + x + y). Identity check, max |efficiency - BW ratio| over all rows: 4.44e-16.

### Per real shape, M=1, graph mode

| model | layer | impl | kernel | graph us | BF16 us | GB/s | %B0 | S measured | S theory | efficiency | eager S | isolated S | excess us | per-op us | per-byte us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | v_proj | FP8 pt (_scaled_mm) | nvjet_64x32 | 21.29 | 51.19 | 197.3 | 82.4 | 2.40 | 2.00 | 120.2% | 2.46 | 1.74 | - | - | - |
| llama31_8b | v_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 37.55 | 51.19 | 112.0 | 46.8 | 1.36 | 2.00 | 68.2% | 1.32 | 1.21 | 20.00 | 18.77 | 1.22 |
| llama31_8b | v_proj | INT4 tinygemm | tinygemm | 12.96 | 51.19 | 172.8 | 72.1 | 3.95 | 3.75 | 105.3% | 3.60 | 3.06 | 3.61 | 3.43 | 0.17 |
| llama31_8b | v_proj | INT4 Marlin | marlin | 16.18 | 51.19 | 134.3 | 56.1 | 3.16 | 3.87 | 81.9% | 2.90 | 1.88 | 7.11 | 6.80 | 0.30 |
| llama31_8b | v_proj | FP8 pt + act quant | nvjet_64x32 | 35.50 | 51.19 | 118.4 | 49.4 | 1.44 | 2.00 | 72.2% | 1.50 | 1.31 | - | - | - |
| llama31_8b | v_proj | FP8 rw + act quant | cutlass_sm120 | 40.13 | 51.19 | 104.9 | 43.8 | 1.28 | 2.00 | 63.9% | 1.21 | 1.04 | - | - | - |
| llama31_8b | k_proj | FP8 pt (_scaled_mm) | nvjet_64x32 | 21.28 | 49.66 | 197.4 | 82.4 | 2.33 | 2.00 | 116.7% | 2.50 | 1.74 | - | - | - |
| llama31_8b | k_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 37.37 | 49.66 | 112.5 | 47.0 | 1.33 | 2.00 | 66.5% | 1.36 | 1.20 | 19.82 | 18.59 | 1.22 |
| llama31_8b | k_proj | INT4 tinygemm | tinygemm | 12.87 | 49.66 | 173.9 | 72.6 | 3.86 | 3.75 | 102.8% | 3.68 | 3.14 | 3.53 | 3.35 | 0.17 |
| llama31_8b | k_proj | INT4 Marlin | marlin | 16.20 | 49.66 | 134.1 | 56.0 | 3.07 | 3.87 | 79.3% | 2.97 | 1.88 | 7.13 | 6.82 | 0.30 |
| llama31_8b | k_proj | FP8 pt + act quant | nvjet_64x32 | 34.13 | 49.66 | 123.2 | 51.4 | 1.45 | 2.00 | 72.8% | 1.52 | 1.32 | - | - | - |
| llama31_8b | k_proj | FP8 rw + act quant | cutlass_sm120 | 39.97 | 49.66 | 105.3 | 44.0 | 1.24 | 2.00 | 62.3% | 1.23 | 1.04 | - | - | - |
| llama31_8b | o_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 82.21 | 202.69 | 204.2 | 85.3 | 2.47 | 2.00 | 123.3% | 2.46 | 2.22 | - | - | - |
| llama31_8b | o_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 82.34 | 202.69 | 204.1 | 85.2 | 2.46 | 2.00 | 123.2% | 2.41 | 2.18 | 12.17 | 7.28 | 4.89 |
| llama31_8b | o_proj | INT4 tinygemm | tinygemm | 40.41 | 202.69 | 221.0 | 92.3 | 5.02 | 3.76 | 133.4% | 4.70 | 4.35 | 3.13 | 2.43 | 0.70 |
| llama31_8b | o_proj | INT4 Marlin | marlin | 40.04 | 202.69 | 216.5 | 90.4 | 5.06 | 3.87 | 130.7% | 4.82 | 3.80 | 3.85 | 2.64 | 1.22 |
| llama31_8b | o_proj | FP8 pt + act quant | nvjet_64x64 | 95.48 | 202.69 | 175.9 | 73.4 | 2.12 | 2.00 | 106.2% | 2.17 | 2.00 | - | - | - |
| llama31_8b | o_proj | FP8 rw + act quant | cutlass_sm120 | 84.62 | 202.69 | 198.6 | 82.9 | 2.40 | 2.00 | 119.9% | 2.29 | 2.00 | - | - | - |
| llama31_8b | qkv_fused | FP8 pt (_scaled_mm) | nvjet_64x64 | 115.75 | 299.40 | 217.6 | 90.8 | 2.59 | 2.00 | 129.4% | 2.60 | 2.39 | - | - | - |
| llama31_8b | qkv_fused | FP8 rw (CUTLASS) | cutlass_sm120 | 119.57 | 299.40 | 210.8 | 88.0 | 2.50 | 2.00 | 125.3% | 2.47 | 2.31 | 14.32 | 6.98 | 7.34 |
| llama31_8b | qkv_fused | INT4 tinygemm | tinygemm | 60.00 | 299.40 | 223.2 | 93.2 | 4.99 | 3.76 | 132.7% | 4.81 | 4.56 | 4.09 | 3.04 | 1.05 |
| llama31_8b | qkv_fused | INT4 Marlin | marlin | 56.73 | 299.40 | 229.1 | 95.7 | 5.28 | 3.87 | 136.2% | 5.08 | 4.29 | 2.46 | 0.64 | 1.82 |
| llama31_8b | qkv_fused | FP8 pt + act quant | nvjet_64x64 | 131.13 | 299.40 | 192.1 | 80.2 | 2.28 | 2.00 | 114.2% | 2.33 | 2.18 | - | - | - |
| llama31_8b | qkv_fused | FP8 rw + act quant | cutlass_sm120 | 121.88 | 299.40 | 206.9 | 86.4 | 2.46 | 2.00 | 123.0% | 2.40 | 2.17 | - | - | - |
| llama31_8b | q_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 81.85 | 198.62 | 205.1 | 85.6 | 2.43 | 2.00 | 121.4% | 2.50 | 2.24 | - | - | - |
| llama31_8b | q_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 82.49 | 198.62 | 203.7 | 85.1 | 2.41 | 2.00 | 120.5% | 2.42 | 2.19 | 12.32 | 7.43 | 4.89 |
| llama31_8b | q_proj | INT4 tinygemm | tinygemm | 40.57 | 198.62 | 220.1 | 91.9 | 4.90 | 3.76 | 130.2% | 4.80 | 4.45 | 3.29 | 2.59 | 0.70 |
| llama31_8b | q_proj | INT4 Marlin | marlin | 39.94 | 198.62 | 217.0 | 90.6 | 4.97 | 3.87 | 128.4% | 4.89 | 3.86 | 3.75 | 2.54 | 1.22 |
| llama31_8b | q_proj | FP8 pt + act quant | nvjet_64x64 | 95.50 | 198.62 | 175.8 | 73.4 | 2.08 | 2.00 | 104.0% | 2.20 | 2.01 | - | - | - |
| llama31_8b | q_proj | FP8 rw + act quant | cutlass_sm120 | 84.87 | 198.62 | 198.1 | 82.7 | 2.34 | 2.00 | 117.2% | 2.34 | 2.02 | - | - | - |
| llama31_8b | down_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 266.65 | 508.59 | 220.3 | 92.0 | 1.91 | 2.00 | 95.4% | 1.93 | 1.84 | - | - | - |
| llama31_8b | down_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 270.74 | 508.59 | 217.0 | 90.6 | 1.88 | 2.00 | 94.0% | 1.86 | 1.80 | 25.39 | 8.28 | 17.11 |
| llama31_8b | down_proj | INT4 tinygemm | tinygemm | 139.60 | 508.59 | 223.7 | 93.4 | 3.64 | 3.76 | 96.9% | 3.61 | 3.51 | 9.19 | 6.75 | 2.44 |
| llama31_8b | down_proj | INT4 Marlin | marlin | 133.25 | 508.59 | 227.5 | 95.0 | 3.82 | 3.88 | 98.5% | 3.79 | 3.50 | 6.67 | 2.42 | 4.25 |
| llama31_8b | down_proj | FP8 pt + act quant | nvjet_64x64 | 296.84 | 508.59 | 197.9 | 82.7 | 1.71 | 2.00 | 85.7% | 1.72 | 1.69 | - | - | - |
| llama31_8b | down_proj | FP8 rw + act quant | cutlass_sm120 | 274.38 | 508.59 | 214.2 | 89.4 | 1.85 | 2.00 | 92.7% | 1.82 | 1.77 | - | - | - |
| llama31_8b | gate_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 264.53 | 502.30 | 222.1 | 92.7 | 1.90 | 2.00 | 95.0% | 1.87 | 1.64 | 19.20 | -28.59 | 47.79 |
| llama31_8b | gate_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 271.36 | 502.30 | 216.7 | 90.5 | 1.85 | 2.00 | 92.7% | 1.85 | 1.76 | 25.80 | 8.67 | 17.13 |
| llama31_8b | gate_proj | INT4 tinygemm | tinygemm | 136.85 | 502.30 | 228.2 | 95.3 | 3.67 | 3.76 | 97.6% | 3.61 | 3.50 | 6.44 | 4.00 | 2.44 |
| llama31_8b | gate_proj | INT4 Marlin | marlin | 133.49 | 502.30 | 227.1 | 94.8 | 3.76 | 3.88 | 97.1% | 3.72 | 3.42 | 6.91 | 2.66 | 4.25 |
| llama31_8b | gate_proj | FP8 pt + act quant | xmma_sm89 | 302.99 | 502.30 | 193.9 | 81.0 | 1.66 | 2.00 | 82.9% | 1.77 | 1.58 | - | - | - |
| llama31_8b | gate_proj | FP8 rw + act quant | cutlass_sm120 | 275.58 | 502.30 | 213.4 | 89.1 | 1.82 | 2.00 | 91.3% | 1.78 | 1.71 | - | - | - |
| llama31_8b | up_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 268.46 | 501.64 | 218.9 | 91.4 | 1.87 | 2.00 | 93.5% | 1.87 | 1.63 | 23.14 | -24.66 | 47.79 |
| llama31_8b | up_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 270.19 | 501.64 | 217.7 | 90.9 | 1.86 | 2.00 | 92.9% | 1.85 | 1.77 | 24.63 | 7.50 | 17.13 |
| llama31_8b | up_proj | INT4 tinygemm | tinygemm | 137.11 | 501.64 | 227.8 | 95.1 | 3.66 | 3.76 | 97.3% | 3.61 | 3.52 | 6.70 | 4.26 | 2.44 |
| llama31_8b | up_proj | INT4 Marlin | marlin | 133.85 | 501.64 | 226.5 | 94.6 | 3.75 | 3.88 | 96.7% | 3.71 | 3.43 | 7.27 | 3.02 | 4.25 |
| llama31_8b | up_proj | FP8 pt + act quant | xmma_sm89 | 280.40 | 501.64 | 209.5 | 87.5 | 1.79 | 2.00 | 89.5% | 1.77 | 1.60 | - | - | - |
| llama31_8b | up_proj | FP8 rw + act quant | cutlass_sm120 | 275.33 | 501.64 | 213.6 | 89.2 | 1.82 | 2.00 | 91.2% | 1.79 | 1.72 | - | - | - |
| llama31_8b | gate_up_fused | FP8 pt (_scaled_mm) | nvjet_32x32 | 495.53 | 997.70 | 237.1 | 99.0 | 2.01 | 2.00 | 100.7% | 1.99 | 1.93 | - | - | - |
| llama31_8b | gate_up_fused | FP8 rw (CUTLASS) | cutlass_sm120 | 523.53 | 997.70 | 224.7 | 93.8 | 1.91 | 2.00 | 95.4% | 1.92 | 1.86 | 32.42 | -1.83 | 34.25 |
| llama31_8b | gate_up_fused | INT4 tinygemm | tinygemm | 273.02 | 997.70 | 228.8 | 95.5 | 3.65 | 3.76 | 97.1% | 3.63 | 3.60 | 12.23 | 7.35 | 4.88 |
| llama31_8b | gate_up_fused | INT4 Marlin | marlin | 263.91 | 997.70 | 229.7 | 95.9 | 3.78 | 3.88 | 97.5% | 3.75 | 3.60 | 10.79 | 2.29 | 8.50 |
| llama31_8b | gate_up_fused | FP8 pt + act quant | nvjet_32x32 | 555.30 | 997.70 | 211.6 | 88.4 | 1.80 | 2.00 | 89.9% | 1.94 | 1.89 | - | - | - |
| llama31_8b | gate_up_fused | FP8 rw + act quant | cutlass_sm120 | 536.70 | 997.70 | 219.2 | 91.5 | 1.86 | 2.00 | 93.1% | 1.85 | 1.81 | - | - | - |
| llama31_8b | lm_head | FP8 pt (_scaled_mm) | xmma_sm89 | 2614.82 | 6010.62 | 201.0 | 83.9 | 2.30 | 2.00 | 115.0% | 2.32 | 2.29 | 420.18 | -7.36 | 427.54 |
| llama31_8b | lm_head | FP8 rw (CUTLASS) | cutlass_sm120 | 2346.85 | 6010.62 | 224.2 | 93.6 | 2.56 | 2.00 | 128.2% | 2.57 | 2.55 | 150.08 | -3.14 | 153.21 |
| llama31_8b | lm_head | INT4 tinygemm | tinygemm | 1199.92 | 6010.62 | 232.8 | 97.2 | 5.01 | 3.76 | 133.1% | 4.99 | 4.96 | 33.50 | 11.66 | 21.84 |
| llama31_8b | lm_head | INT4 Marlin | marlin | 1171.54 | 6010.62 | 231.4 | 96.6 | 5.13 | 3.88 | 132.4% | 5.08 | 5.02 | 39.39 | 1.36 | 38.02 |
| llama31_8b | lm_head | FP8 pt + act quant | xmma_sm89 | 2623.06 | 6010.62 | 200.4 | 83.7 | 2.29 | 2.00 | 114.6% | 2.30 | 2.31 | - | - | - |
| llama31_8b | lm_head | FP8 rw + act quant | cutlass_sm120 | 2365.24 | 6010.62 | 222.4 | 92.9 | 2.54 | 2.00 | 127.2% | 2.55 | 2.54 | - | - | - |
| qwen25_7b | v_proj | FP8 pt (_scaled_mm) | nvjet_32x32 | 11.35 | 17.65 | 162.1 | 67.7 | 1.55 | 2.00 | 77.7% | 1.58 | 1.04 | - | - | - |
| qwen25_7b | v_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 31.66 | 17.65 | 58.2 | 24.3 | 0.56 | 2.00 | 27.9% | 0.59 | 0.58 | 23.97 | 23.44 | 0.54 |
| qwen25_7b | v_proj | INT4 tinygemm | tinygemm | 8.37 | 17.65 | 117.5 | 49.1 | 2.11 | 3.74 | 56.4% | 1.96 | 1.79 | 4.26 | 4.18 | 0.08 |
| qwen25_7b | v_proj | INT4 Marlin | marlin | 14.35 | 17.65 | 66.5 | 27.8 | 1.23 | 3.85 | 31.9% | 1.18 | 0.86 | 10.37 | 10.23 | 0.13 |
| qwen25_7b | v_proj | FP8 pt + act quant | nvjet_32x32 | 24.62 | 17.65 | 74.9 | 31.3 | 0.72 | 2.00 | 35.9% | 0.76 | 0.72 | - | - | - |
| qwen25_7b | v_proj | FP8 rw + act quant | cutlass_sm120 | 34.29 | 17.65 | 53.8 | 22.5 | 0.51 | 1.99 | 25.8% | 0.53 | 0.50 | - | - | - |
| qwen25_7b | k_proj | FP8 pt (_scaled_mm) | nvjet_32x32 | 11.34 | 17.66 | 162.2 | 67.7 | 1.56 | 2.00 | 77.9% | 1.61 | 1.04 | - | - | - |
| qwen25_7b | k_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 31.68 | 17.66 | 58.1 | 24.3 | 0.56 | 2.00 | 27.9% | 0.59 | 0.58 | 23.99 | 23.46 | 0.54 |
| qwen25_7b | k_proj | INT4 tinygemm | tinygemm | 8.36 | 17.66 | 117.6 | 49.1 | 2.11 | 3.74 | 56.5% | 1.99 | 1.78 | 4.25 | 4.17 | 0.08 |
| qwen25_7b | k_proj | INT4 Marlin | marlin | 14.58 | 17.66 | 65.5 | 27.3 | 1.21 | 3.85 | 31.4% | 1.19 | 0.85 | 10.59 | 10.46 | 0.13 |
| qwen25_7b | k_proj | FP8 pt + act quant | nvjet_32x32 | 24.64 | 17.66 | 74.8 | 31.2 | 0.72 | 2.00 | 35.9% | 0.76 | 0.70 | - | - | - |
| qwen25_7b | k_proj | FP8 rw + act quant | cutlass_sm120 | 34.33 | 17.66 | 53.8 | 22.4 | 0.51 | 1.99 | 25.8% | 0.52 | 0.49 | - | - | - |
| qwen25_7b | qkv_fused | FP8 pt (_scaled_mm) | nvjet_64x64 | 76.43 | 198.61 | 216.2 | 90.3 | 2.60 | 2.00 | 130.0% | 2.64 | 2.30 | - | - | - |
| qwen25_7b | qkv_fused | FP8 rw (CUTLASS) | cutlass_sm120 | 79.82 | 198.61 | 207.3 | 86.6 | 2.49 | 2.00 | 124.6% | 2.46 | 2.22 | 10.73 | 5.91 | 4.82 |
| qwen25_7b | qkv_fused | INT4 tinygemm | tinygemm | 40.65 | 198.61 | 216.2 | 90.3 | 4.89 | 3.76 | 130.0% | 4.69 | 4.32 | 3.95 | 3.26 | 0.69 |
| qwen25_7b | qkv_fused | INT4 Marlin | marlin | 39.44 | 198.61 | 216.3 | 90.3 | 5.04 | 3.87 | 130.0% | 4.91 | 3.82 | 3.82 | 2.62 | 1.20 |
| qwen25_7b | qkv_fused | FP8 pt + act quant | nvjet_64x64 | 92.16 | 198.61 | 179.4 | 74.9 | 2.16 | 2.00 | 107.8% | 2.24 | 2.03 | - | - | - |
| qwen25_7b | qkv_fused | FP8 rw + act quant | cutlass_sm120 | 83.06 | 198.61 | 199.2 | 83.2 | 2.39 | 2.00 | 119.8% | 2.34 | 2.01 | - | - | - |
| qwen25_7b | o_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 65.66 | 156.18 | 195.8 | 81.8 | 2.38 | 2.00 | 119.0% | 2.39 | 2.10 | - | - | - |
| qwen25_7b | o_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 63.96 | 156.18 | 201.2 | 84.0 | 2.44 | 2.00 | 122.3% | 2.39 | 2.10 | 10.22 | 6.48 | 3.75 |
| qwen25_7b | o_proj | INT4 tinygemm | tinygemm | 32.65 | 156.18 | 209.5 | 87.5 | 4.78 | 3.76 | 127.3% | 4.53 | 4.15 | 4.09 | 3.56 | 0.53 |
| qwen25_7b | o_proj | INT4 Marlin | marlin | 31.64 | 156.18 | 209.8 | 87.6 | 4.94 | 3.87 | 127.5% | 4.71 | 3.58 | 3.92 | 2.99 | 0.93 |
| qwen25_7b | o_proj | FP8 pt + act quant | nvjet_64x64 | 76.93 | 156.18 | 167.2 | 69.8 | 2.03 | 2.00 | 101.6% | 2.04 | 1.84 | - | - | - |
| qwen25_7b | o_proj | FP8 rw + act quant | cutlass_sm120 | 67.01 | 156.18 | 192.1 | 80.2 | 2.33 | 2.00 | 116.7% | 2.25 | 1.90 | - | - | - |
| qwen25_7b | q_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 65.57 | 158.15 | 196.1 | 81.9 | 2.41 | 2.00 | 120.6% | 2.41 | 2.09 | - | - | - |
| qwen25_7b | q_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 64.44 | 158.15 | 199.7 | 83.4 | 2.45 | 2.00 | 122.9% | 2.39 | 2.09 | 10.71 | 6.96 | 3.75 |
| qwen25_7b | q_proj | INT4 tinygemm | tinygemm | 32.51 | 158.15 | 210.3 | 87.8 | 4.86 | 3.76 | 129.4% | 4.59 | 4.13 | 3.96 | 3.42 | 0.53 |
| qwen25_7b | q_proj | INT4 Marlin | marlin | 31.57 | 158.15 | 210.3 | 87.8 | 5.01 | 3.87 | 129.4% | 4.75 | 3.56 | 3.85 | 2.92 | 0.93 |
| qwen25_7b | q_proj | FP8 pt + act quant | nvjet_64x64 | 77.15 | 158.15 | 166.7 | 69.6 | 2.05 | 2.00 | 102.6% | 2.04 | 1.83 | - | - | - |
| qwen25_7b | q_proj | FP8 rw + act quant | cutlass_sm120 | 66.95 | 158.15 | 192.3 | 80.3 | 2.36 | 2.00 | 118.3% | 2.27 | 1.88 | - | - | - |
| qwen25_7b | down_proj | FP8 pt (_scaled_mm) | nvjet_64x64 | 314.56 | 587.24 | 215.9 | 90.2 | 1.87 | 2.00 | 93.3% | 1.87 | 1.81 | - | - | - |
| qwen25_7b | down_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 315.19 | 587.24 | 215.5 | 90.0 | 1.86 | 2.00 | 93.2% | 1.87 | 1.81 | 31.52 | 11.74 | 19.78 |
| qwen25_7b | down_proj | INT4 tinygemm | tinygemm | 161.72 | 587.24 | 223.3 | 93.2 | 3.63 | 3.76 | 96.5% | 3.60 | 3.53 | 10.92 | 8.10 | 2.82 |
| qwen25_7b | down_proj | INT4 Marlin | marlin | 153.39 | 587.24 | 228.5 | 95.4 | 3.83 | 3.88 | 98.8% | 3.81 | 3.56 | 7.02 | 2.11 | 4.92 |
| qwen25_7b | down_proj | FP8 pt + act quant | nvjet_64x64 | 354.95 | 587.24 | 191.4 | 79.9 | 1.65 | 2.00 | 82.7% | 1.65 | 1.64 | - | - | - |
| qwen25_7b | down_proj | FP8 rw + act quant | cutlass_sm120 | 322.98 | 587.24 | 210.4 | 87.9 | 1.82 | 2.00 | 91.0% | 1.82 | 1.77 | - | - | - |
| qwen25_7b | gate_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 315.40 | 798.83 | 215.4 | 89.9 | 2.53 | 2.00 | 126.7% | 2.52 | 2.36 | 31.73 | -23.53 | 55.26 |
| qwen25_7b | gate_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 329.95 | 798.83 | 206.1 | 86.1 | 2.42 | 2.00 | 121.2% | 2.41 | 2.35 | 45.96 | 26.15 | 19.81 |
| qwen25_7b | gate_proj | INT4 tinygemm | tinygemm | 157.96 | 798.83 | 228.6 | 95.5 | 5.06 | 3.76 | 134.5% | 5.01 | 4.88 | 7.16 | 4.34 | 2.82 |
| qwen25_7b | gate_proj | INT4 Marlin | marlin | 154.25 | 798.83 | 227.2 | 94.9 | 5.18 | 3.88 | 133.6% | 5.14 | 4.81 | 7.89 | 2.97 | 4.92 |
| qwen25_7b | gate_proj | FP8 pt + act quant | xmma_sm89 | 339.60 | 798.83 | 200.1 | 83.5 | 2.35 | 2.00 | 117.7% | 2.41 | 2.26 | - | - | - |
| qwen25_7b | gate_proj | FP8 rw + act quant | cutlass_sm120 | 329.68 | 798.83 | 206.3 | 86.1 | 2.42 | 2.00 | 121.3% | 2.41 | 2.32 | - | - | - |
| qwen25_7b | up_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 316.70 | 800.09 | 214.5 | 89.6 | 2.53 | 2.00 | 126.4% | 2.52 | 2.36 | 33.03 | -22.23 | 55.26 |
| qwen25_7b | up_proj | FP8 rw (CUTLASS) | cutlass_sm120 | 329.96 | 800.09 | 206.1 | 86.1 | 2.42 | 2.00 | 121.4% | 2.41 | 2.34 | 45.97 | 26.17 | 19.81 |
| qwen25_7b | up_proj | INT4 tinygemm | tinygemm | 157.87 | 800.09 | 228.8 | 95.5 | 5.07 | 3.76 | 134.7% | 4.96 | 4.88 | 7.07 | 4.25 | 2.82 |
| qwen25_7b | up_proj | INT4 Marlin | marlin | 154.07 | 800.09 | 227.5 | 95.0 | 5.19 | 3.88 | 134.0% | 5.14 | 4.77 | 7.70 | 2.79 | 4.92 |
| qwen25_7b | up_proj | FP8 pt + act quant | xmma_sm89 | 328.57 | 800.09 | 206.8 | 86.3 | 2.44 | 2.00 | 121.8% | 2.49 | 2.27 | - | - | - |
| qwen25_7b | up_proj | FP8 rw + act quant | cutlass_sm120 | 329.46 | 800.09 | 206.4 | 86.2 | 2.43 | 2.00 | 121.6% | 2.41 | 2.32 | - | - | - |
| qwen25_7b | gate_up_fused | FP8 pt (_scaled_mm) | nvjet_128x72 | 595.08 | 1588.10 | 228.3 | 95.3 | 2.67 | 2.00 | 133.5% | 2.67 | 2.57 | - | - | - |
| qwen25_7b | gate_up_fused | FP8 rw (CUTLASS) | cutlass_sm120 | 603.03 | 1588.10 | 225.6 | 94.2 | 2.63 | 2.00 | 131.9% | 2.62 | 2.53 | 35.07 | -4.54 | 39.61 |
| qwen25_7b | gate_up_fused | INT4 tinygemm | tinygemm | 315.08 | 1588.10 | 229.2 | 95.7 | 5.04 | 3.76 | 134.0% | 5.06 | 4.97 | 13.51 | 7.87 | 5.65 |
| qwen25_7b | gate_up_fused | INT4 Marlin | marlin | 305.38 | 1588.10 | 229.6 | 95.8 | 5.20 | 3.88 | 134.2% | 5.17 | 4.97 | 12.67 | 2.84 | 9.83 |
| qwen25_7b | gate_up_fused | FP8 pt + act quant | nvjet_128x72 | 608.80 | 1588.10 | 223.2 | 93.2 | 2.61 | 2.00 | 130.5% | 2.63 | 2.54 | - | - | - |
| qwen25_7b | gate_up_fused | FP8 rw + act quant | cutlass_sm120 | 626.70 | 1588.10 | 217.0 | 90.6 | 2.53 | 2.00 | 126.9% | 2.60 | 2.52 | - | - | - |
| qwen25_7b | lm_head | FP8 pt (_scaled_mm) | xmma_sm89 | 2618.90 | 6305.90 | 208.2 | 86.9 | 2.41 | 2.00 | 120.4% | 2.41 | 2.39 | 341.97 | -101.60 | 443.57 |
| qwen25_7b | lm_head | FP8 rw (CUTLASS) | cutlass_sm120 | 2452.53 | 6305.90 | 222.6 | 92.9 | 2.57 | 2.00 | 128.7% | 2.56 | 2.55 | 173.07 | 14.09 | 158.98 |
| qwen25_7b | lm_head | INT4 tinygemm | tinygemm | 1238.64 | 6305.90 | 234.0 | 97.7 | 5.09 | 3.76 | 135.3% | 5.10 | 5.02 | 28.40 | 5.75 | 22.66 |
| qwen25_7b | lm_head | INT4 Marlin | marlin | 1216.79 | 6305.90 | 231.2 | 96.5 | 5.18 | 3.88 | 133.7% | 5.18 | 5.12 | 42.12 | 2.67 | 39.45 |
| qwen25_7b | lm_head | FP8 pt + act quant | xmma_sm89 | 2636.84 | 6305.90 | 206.8 | 86.4 | 2.39 | 2.00 | 119.6% | 2.39 | 2.38 | - | - | - |
| qwen25_7b | lm_head | FP8 rw + act quant | cutlass_sm120 | 2453.47 | 6305.90 | 222.5 | 92.9 | 2.57 | 2.00 | 128.7% | 2.61 | 2.59 | - | - | - |

%B0 is also the efficiency against an ideal BF16 kernel at B0 (cuBLAS BF16 itself runs at about 70% of B0 on 13 of 20 shapes, so efficiency against cuBLAS can exceed 100%). excess = t - B / B0; per-byte = B (1/R_asym - 1/B0) with the asymptotic rate of the kernel dispatched for that shape (below; '-' when that kernel has fewer than 3 ops >= 64 MiB); per-op = excess - per-byte.

### Asymptotic streaming rate per implementation (graph, M=1, ops >= 64 MiB)

| impl | R_asym GB/s | CI95 | %B0 | intercept us | points | kernels |
|---|---|---|---|---|---|---|
| INT4 Marlin (marlin) | 231.7 | 230.5 to 233.6 | 96.8 | 3.87 | 9 | marlin |
| INT4 tinygemm (tinygemm) | 235.1 | 228.3 to 239.7 | 98.2 | 20.69 | 9 | tinygemm |
| FP8 pt (_scaled_mm) (nvjet_192x112) | 198.5 | 172.7 to unbounded | 82.9 | 16.57 | 3 | nvjet_192x112 |
| FP8 pt (_scaled_mm) (xmma_sm89) | 200.4 | 199.0 to 4830.3 | 83.7 | -14.70 | 6 | xmma_sm89 |
| FP8 rw (CUTLASS) (cutlass_sm120) | 223.9 | 220.7 to 227.5 | 93.5 | 0.52 | 15 | cutlass_sm120 |

### Per-kernel fits t = tau + B / R (M=1; informational, misfits are expected for tiled kernels)

| impl | kernel | mode | points | tau us | tau CI95 | R_inf GB/s | R_inf %B0 | B_half MiB | rms rel resid |
|---|---|---|---|---|---|---|---|---|---|
| INT4 Marlin | marlin | eager | 44 | 14.55 | 11.98 to 18.15 | 263.3 | 110.0 | 3.653 | 20.7% misfit |
| INT4 Marlin | marlin | graph | 44 | 12.60 | 10.12 to 16.15 | 264.4 | 110.4 | 3.176 | 22.0% misfit |
| INT4 Marlin | marlin | isolated | 44 | 29.35 | 26.16 to 33.29 | 259.2 | 108.2 | 7.256 | 15.0% misfit |
| INT4 tinygemm | tinygemm | eager | 46 | 6.73 | 6.09 to 7.72 | 236.7 | 98.8 | 1.519 | 10.3% misfit |
| INT4 tinygemm | tinygemm | graph | 46 | 4.48 | 3.91 to 5.77 | 234.6 | 98.0 | 1.002 | 12.8% misfit |
| INT4 tinygemm | tinygemm | isolated | 46 | 12.21 | 11.45 to 13.10 | 236.2 | 98.6 | 2.749 | 6.9% misfit |
| FP8 pt (_scaled_mm) | nvjet_16x16 | eager | 5 | 11.90 | 11.73 to 11.94 | 23709.7 | 9900.0 | 269.161 | 0.7% |
| FP8 pt (_scaled_mm) | nvjet_16x16 | graph | 5 | 6.06 | 5.78 to 6.29 | 398.1 | 166.2 | 2.301 | 2.6% |
| FP8 pt (_scaled_mm) | nvjet_16x16 | isolated | 5 | 19.85 | 18.67 to 20.62 | 397.8 | 166.1 | 7.530 | 2.1% |
| FP8 pt (_scaled_mm) | nvjet_64x64 | eager | 12 | 5.42 | 1.69 to 9.55 | 224.6 | 93.8 | 1.162 | 3.2% |
| FP8 pt (_scaled_mm) | nvjet_64x64 | graph | 12 | 5.72 | 2.47 to 9.41 | 224.9 | 93.9 | 1.227 | 2.8% |
| FP8 pt (_scaled_mm) | nvjet_64x64 | isolated | 12 | 19.22 | 16.68 to 22.17 | 224.5 | 93.7 | 4.116 | 2.0% |
| FP8 pt (_scaled_mm) | xmma_sm89 | eager | 8 | -17.34 | -22.01 to -11.46 | 203.9 | 85.2 | -3.373 | 0.9% |
| FP8 pt (_scaled_mm) | xmma_sm89 | graph | 8 | -22.54 | -29.97 to -14.84 | 201.8 | 84.3 | -4.337 | 1.2% |
| FP8 pt (_scaled_mm) | xmma_sm89 | isolated | 8 | 19.43 | 9.69 to 30.12 | 205.4 | 85.8 | 3.806 | 1.7% |
| FP8 rw (CUTLASS) | cutlass_sm120 | eager | 46 | 30.43 | 26.39 to 35.54 | 249.0 | 104.0 | 7.227 | 24.9% misfit |
| FP8 rw (CUTLASS) | cutlass_sm120 | graph | 46 | 28.70 | 24.81 to 33.52 | 249.8 | 104.3 | 6.837 | 25.4% misfit |
| FP8 rw (CUTLASS) | cutlass_sm120 | isolated | 46 | 43.41 | 38.73 to 49.72 | 244.6 | 102.1 | 10.125 | 22.1% misfit |

### Activation quantization cost (full op minus GEMM only), M=1

| impl | mode | median us | min us | max us | shapes |
|---|---|---|---|---|---|
| FP8 pt + act quant | isolated | 11.66 | -16.06 | 33.74 | 20 |
| FP8 pt + act quant | eager | 13.48 | 4.16 | 40.99 | 20 |
| FP8 pt + act quant | graph | 13.68 | 8.24 | 59.78 | 20 |
| FP8 rw + act quant | isolated | 8.08 | -41.39 | 16.58 | 20 |
| FP8 rw + act quant | eager | 4.09 | -46.57 | 19.32 | 20 |
| FP8 rw + act quant | graph | 2.64 | -0.50 | 23.68 | 20 |

### Per-token GEMV cost (graph, M=1; all layers + lm_head)

| model | layers | impl | ms/token | GiB/token | ideal ms at B0 | % of ideal | speedup vs BF16 | theory |
|---|---|---|---|---|---|---|---|---|
| llama31_8b | unfused | cuBLAS BF16 | 70.48 | 13.98 | 62.69 | 89.0 | 1.00 | 1.00 |
| llama31_8b | unfused | FP8 pt (_scaled_mm) | 34.82 | 6.99 | 31.35 | 90.1 | 2.02 | 2.00 |
| llama31_8b | unfused | FP8 rw (CUTLASS) | 36.01 | 7.00 | 31.38 | 87.1 | 1.96 | 2.00 |
| llama31_8b | unfused | INT4 tinygemm | 17.85 | 3.72 | 16.67 | 93.4 | 3.95 | 3.76 |
| llama31_8b | unfused | INT4 Marlin | 17.59 | 3.61 | 16.18 | 92.0 | 4.01 | 3.87 |
| llama31_8b | unfused | FP8 pt + act quant | 39.13 | 6.99 | 31.36 | 80.1 | 1.80 | 2.00 |
| llama31_8b | unfused | FP8 rw + act quant | 36.76 | 7.00 | 31.38 | 85.4 | 1.92 | 2.00 |
| llama31_8b | fused | cuBLAS BF16 | 70.28 | 13.98 | 62.69 | 89.2 | 1.00 | 1.00 |
| llama31_8b | fused | FP8 pt (_scaled_mm) | 33.34 | 6.99 | 31.35 | 94.0 | 2.11 | 2.00 |
| llama31_8b | fused | FP8 rw (CUTLASS) | 34.22 | 7.00 | 31.38 | 91.7 | 2.05 | 2.00 |
| llama31_8b | fused | INT4 tinygemm | 17.62 | 3.72 | 16.67 | 94.6 | 3.99 | 3.76 |
| llama31_8b | fused | INT4 Marlin | 16.98 | 3.61 | 16.18 | 95.3 | 4.14 | 3.88 |
| llama31_8b | fused | FP8 pt + act quant | 37.14 | 6.99 | 31.36 | 84.4 | 1.89 | 2.00 |
| llama31_8b | fused | FP8 rw + act quant | 34.93 | 7.00 | 31.38 | 89.8 | 2.01 | 2.00 |
| qwen25_7b | unfused | cuBLAS BF16 | 77.31 | 13.17 | 59.07 | 76.4 | 1.00 | 1.00 |
| qwen25_7b | unfused | FP8 pt (_scaled_mm) | 33.44 | 6.59 | 29.54 | 88.3 | 2.31 | 2.00 |
| qwen25_7b | unfused | FP8 rw (CUTLASS) | 35.12 | 6.59 | 29.57 | 84.2 | 2.20 | 2.00 |
| qwen25_7b | unfused | INT4 tinygemm | 16.90 | 3.50 | 15.71 | 92.9 | 4.57 | 3.76 |
| qwen25_7b | unfused | INT4 Marlin | 16.72 | 3.40 | 15.24 | 91.2 | 4.62 | 3.87 |
| qwen25_7b | unfused | FP8 pt + act quant | 36.98 | 6.59 | 29.54 | 79.9 | 2.09 | 2.00 |
| qwen25_7b | unfused | FP8 rw + act quant | 35.62 | 6.60 | 29.57 | 83.0 | 2.17 | 2.00 |
| qwen25_7b | fused | cuBLAS BF16 | 77.15 | 13.17 | 59.06 | 76.6 | 1.00 | 1.00 |
| qwen25_7b | fused | FP8 pt (_scaled_mm) | 32.07 | 6.59 | 29.54 | 92.1 | 2.41 | 2.00 |
| qwen25_7b | fused | FP8 rw (CUTLASS) | 32.19 | 6.59 | 29.56 | 91.8 | 2.40 | 2.00 |
| qwen25_7b | fused | INT4 tinygemm | 16.64 | 3.50 | 15.70 | 94.4 | 4.64 | 3.76 |
| qwen25_7b | fused | INT4 Marlin | 16.05 | 3.40 | 15.24 | 95.0 | 4.81 | 3.88 |
| qwen25_7b | fused | FP8 pt + act quant | 34.36 | 6.59 | 29.54 | 86.0 | 2.25 | 2.00 |
| qwen25_7b | fused | FP8 rw + act quant | 33.25 | 6.59 | 29.57 | 88.9 | 2.32 | 2.00 |

### M sweep (graph): t(M) / t(1)

| impl | model | layer | M=2 | M=4 | M=8 | M=16 | kernels M=1 / 16 |
|---|---|---|---|---|---|---|---|
| FP8 pt (_scaled_mm) | llama31_8b | down_proj | 1.00 | 1.00 | 1.00 | 1.01 | nvjet_64x64 / nvjet_64x64 |
| FP8 pt (_scaled_mm) | llama31_8b | gate_proj | 1.03 | 1.03 | 1.04 | 1.05 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | llama31_8b | gate_up_fused | 1.01 | 1.01 | 1.12 | 1.13 | nvjet_32x32 / nvjet_128x160 |
| FP8 pt (_scaled_mm) | llama31_8b | k_proj | 0.99 | 0.99 | 1.00 | 0.98 | nvjet_64x32 / nvjet_64x32 |
| FP8 pt (_scaled_mm) | llama31_8b | lm_head | 0.90 | 0.90 | 0.90 | 0.91 | xmma_sm89 / nvjet_192x160 |
| FP8 pt (_scaled_mm) | llama31_8b | o_proj | 1.00 | 0.92 | 0.93 | 0.94 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt (_scaled_mm) | llama31_8b | q_proj | 1.01 | 0.94 | 0.94 | 0.94 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt (_scaled_mm) | llama31_8b | qkv_fused | 0.99 | 0.98 | 1.05 | 1.07 | nvjet_64x64 / nvjet_64x88 |
| FP8 pt (_scaled_mm) | llama31_8b | up_proj | 1.01 | 1.03 | 1.02 | 1.04 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | llama31_8b | v_proj | 1.00 | 1.00 | 1.00 | 0.99 | nvjet_64x32 / nvjet_64x32 |
| FP8 pt (_scaled_mm) | qwen25_7b | down_proj | 1.00 | 1.01 | 1.00 | 1.01 | nvjet_64x64 / nvjet_64x64 |
| FP8 pt (_scaled_mm) | qwen25_7b | gate_proj | 0.97 | 1.01 | 0.97 | 1.02 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | qwen25_7b | gate_up_fused | 1.00 | 1.00 | 1.01 | 1.04 | nvjet_128x72 / nvjet_128x96 |
| FP8 pt (_scaled_mm) | qwen25_7b | k_proj | 0.93 | 0.94 | 0.94 | 1.01 | nvjet_32x32 / nvjet_32x16 |
| FP8 pt (_scaled_mm) | qwen25_7b | lm_head | 0.93 | 0.93 | 0.93 | 0.94 | xmma_sm89 / nvjet_192x160 |
| FP8 pt (_scaled_mm) | qwen25_7b | o_proj | 1.01 | 0.90 | 0.90 | 0.91 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt (_scaled_mm) | qwen25_7b | q_proj | 1.02 | 0.90 | 0.90 | 0.91 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt (_scaled_mm) | qwen25_7b | qkv_fused | 1.01 | 1.00 | 1.00 | 1.01 | nvjet_64x64 / nvjet_96x64 |
| FP8 pt (_scaled_mm) | qwen25_7b | up_proj | 0.97 | 1.00 | 0.98 | 0.98 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | qwen25_7b | v_proj | 0.94 | 0.94 | 0.94 | 1.01 | nvjet_32x32 / nvjet_32x16 |
| FP8 rw (CUTLASS) | llama31_8b | down_proj | 1.01 | 1.01 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | gate_proj | 1.00 | 1.00 | 0.99 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | gate_up_fused | 1.01 | 1.01 | 1.00 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | k_proj | 1.01 | 1.01 | 1.01 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | lm_head | 1.00 | 1.01 | 1.01 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | o_proj | 1.00 | 1.01 | 0.99 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | q_proj | 1.01 | 1.00 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | qkv_fused | 1.02 | 1.01 | 1.01 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | up_proj | 1.01 | 1.00 | 1.00 | 0.99 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | llama31_8b | v_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | down_proj | 1.00 | 0.99 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | gate_proj | 0.95 | 0.98 | 0.97 | 0.97 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | gate_up_fused | 1.02 | 1.01 | 1.02 | 1.04 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | k_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | lm_head | 0.98 | 0.99 | 0.99 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | o_proj | 1.02 | 1.01 | 0.99 | 0.99 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | q_proj | 1.01 | 1.00 | 0.99 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | qkv_fused | 1.02 | 1.01 | 1.01 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | up_proj | 0.94 | 0.99 | 0.95 | 0.97 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw (CUTLASS) | qwen25_7b | v_proj | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| INT4 tinygemm | llama31_8b | down_proj | 1.00 | 1.02 | 1.06 | 2.27 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | gate_proj | 1.01 | 1.01 | 1.02 | 1.96 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | gate_up_fused | 1.00 | 1.00 | 1.01 | 1.78 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | k_proj | 0.99 | 0.99 | 1.07 | 1.84 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | lm_head | 1.00 | 1.00 | 1.01 | 1.78 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | o_proj | 1.00 | 1.00 | 1.03 | 2.04 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | q_proj | 1.00 | 1.00 | 1.02 | 1.97 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | qkv_fused | 1.00 | 1.00 | 1.03 | 2.03 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | up_proj | 1.00 | 1.01 | 1.02 | 1.96 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | v_proj | 0.99 | 0.98 | 1.05 | 1.83 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | down_proj | 1.01 | 1.04 | 1.08 | 2.47 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | gate_proj | 1.00 | 1.00 | 1.02 | 2.03 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | gate_up_fused | 1.00 | 1.00 | 1.01 | 2.01 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | k_proj | 1.01 | 0.94 | 1.05 | 1.63 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | lm_head | 1.00 | 1.01 | 1.01 | 2.00 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | o_proj | 1.00 | 0.99 | 1.04 | 2.00 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | q_proj | 1.00 | 0.99 | 1.03 | 2.00 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | qkv_fused | 1.00 | 1.01 | 1.04 | 2.01 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | up_proj | 1.00 | 1.00 | 1.02 | 2.03 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | v_proj | 0.98 | 0.95 | 1.05 | 1.62 | tinygemm / tinygemm |
| INT4 Marlin | llama31_8b | down_proj | 1.00 | 1.00 | 1.01 | 1.02 | marlin / marlin |
| INT4 Marlin | llama31_8b | gate_proj | 1.00 | 1.01 | 1.01 | 1.03 | marlin / marlin |
| INT4 Marlin | llama31_8b | gate_up_fused | 1.00 | 1.01 | 1.01 | 1.03 | marlin / marlin |
| INT4 Marlin | llama31_8b | k_proj | 1.00 | 1.00 | 1.02 | 1.09 | marlin / marlin |
| INT4 Marlin | llama31_8b | lm_head | 1.00 | 1.01 | 1.02 | 1.03 | marlin / marlin |
| INT4 Marlin | llama31_8b | o_proj | 1.00 | 1.01 | 1.00 | 1.02 | marlin / marlin |
| INT4 Marlin | llama31_8b | q_proj | 1.00 | 1.01 | 1.00 | 1.03 | marlin / marlin |
| INT4 Marlin | llama31_8b | qkv_fused | 0.95 | 1.00 | 1.01 | 1.02 | marlin / marlin |
| INT4 Marlin | llama31_8b | up_proj | 1.00 | 1.00 | 1.01 | 1.03 | marlin / marlin |
| INT4 Marlin | llama31_8b | v_proj | 1.00 | 1.00 | 1.01 | 1.09 | marlin / marlin |
| INT4 Marlin | qwen25_7b | down_proj | 1.00 | 1.00 | 1.01 | 1.03 | marlin / marlin |
| INT4 Marlin | qwen25_7b | gate_proj | 1.00 | 1.00 | 1.02 | 1.04 | marlin / marlin |
| INT4 Marlin | qwen25_7b | gate_up_fused | 1.00 | 1.01 | 0.99 | 1.03 | marlin / marlin |
| INT4 Marlin | qwen25_7b | k_proj | 1.00 | 0.99 | 1.01 | 1.12 | marlin / marlin |
| INT4 Marlin | qwen25_7b | lm_head | 1.00 | 1.01 | 1.01 | 1.03 | marlin / marlin |
| INT4 Marlin | qwen25_7b | o_proj | 0.99 | 0.99 | 1.00 | 1.01 | marlin / marlin |
| INT4 Marlin | qwen25_7b | q_proj | 0.99 | 0.99 | 1.00 | 1.02 | marlin / marlin |
| INT4 Marlin | qwen25_7b | qkv_fused | 0.99 | 1.00 | 0.97 | 1.02 | marlin / marlin |
| INT4 Marlin | qwen25_7b | up_proj | 1.00 | 1.00 | 1.02 | 1.04 | marlin / marlin |
| INT4 Marlin | qwen25_7b | v_proj | 1.01 | 1.02 | 1.02 | 1.13 | marlin / marlin |
| FP8 pt + act quant | llama31_8b | down_proj | 1.00 | 1.01 | 1.01 | 1.01 | nvjet_64x64 / nvjet_64x64 |
| FP8 pt + act quant | llama31_8b | gate_proj | 0.93 | 0.94 | 0.95 | 0.94 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | llama31_8b | gate_up_fused | 1.00 | 0.95 | 1.15 | 1.10 | nvjet_32x32 / nvjet_128x160 |
| FP8 pt + act quant | llama31_8b | k_proj | 1.04 | 1.04 | 1.05 | 1.05 | nvjet_64x32 / nvjet_64x32 |
| FP8 pt + act quant | llama31_8b | lm_head | 0.89 | 0.90 | 0.91 | 0.91 | xmma_sm89 / nvjet_192x160 |
| FP8 pt + act quant | llama31_8b | o_proj | 1.00 | 0.94 | 0.94 | 0.96 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt + act quant | llama31_8b | q_proj | 1.00 | 0.94 | 0.95 | 0.95 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt + act quant | llama31_8b | qkv_fused | 0.99 | 0.99 | 1.19 | 1.11 | nvjet_64x64 / nvjet_64x88 |
| FP8 pt + act quant | llama31_8b | up_proj | 1.02 | 1.02 | 1.03 | 1.03 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | llama31_8b | v_proj | 1.00 | 1.01 | 1.01 | 1.00 | nvjet_64x32 / nvjet_64x32 |
| FP8 pt + act quant | qwen25_7b | down_proj | 1.00 | 1.00 | 1.01 | 1.01 | nvjet_64x64 / nvjet_64x64 |
| FP8 pt + act quant | qwen25_7b | gate_proj | 0.94 | 0.98 | 0.94 | 0.99 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | qwen25_7b | gate_up_fused | 1.00 | 0.99 | 0.99 | 1.03 | nvjet_128x72 / nvjet_128x96 |
| FP8 pt + act quant | qwen25_7b | k_proj | 0.96 | 0.98 | 0.97 | 0.99 | nvjet_32x32 / nvjet_32x16 |
| FP8 pt + act quant | qwen25_7b | lm_head | 0.92 | 0.92 | 0.93 | 0.94 | xmma_sm89 / nvjet_192x160 |
| FP8 pt + act quant | qwen25_7b | o_proj | 0.99 | 0.93 | 0.94 | 0.95 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt + act quant | qwen25_7b | q_proj | 0.98 | 0.93 | 0.94 | 0.94 | nvjet_64x64 / nvjet_192x16 |
| FP8 pt + act quant | qwen25_7b | qkv_fused | 0.97 | 0.97 | 0.97 | 0.97 | nvjet_64x64 / nvjet_96x64 |
| FP8 pt + act quant | qwen25_7b | up_proj | 1.00 | 0.98 | 0.98 | 1.02 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | qwen25_7b | v_proj | 0.97 | 0.97 | 0.96 | 0.99 | nvjet_32x32 / nvjet_32x16 |
| FP8 rw + act quant | llama31_8b | down_proj | 1.00 | 1.01 | 1.00 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | gate_proj | 1.00 | 0.98 | 0.99 | 0.99 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | gate_up_fused | 1.00 | 0.99 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | k_proj | 0.99 | 0.99 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | lm_head | 1.00 | 1.00 | 1.01 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | o_proj | 1.00 | 1.00 | 1.00 | 1.02 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | q_proj | 0.99 | 1.00 | 1.01 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | qkv_fused | 1.00 | 1.00 | 1.00 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | up_proj | 1.01 | 0.99 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | llama31_8b | v_proj | 0.99 | 0.99 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | down_proj | 1.00 | 1.00 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | gate_proj | 0.97 | 0.96 | 0.98 | 0.97 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | gate_up_fused | 0.98 | 0.97 | 0.98 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | k_proj | 1.00 | 1.00 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | lm_head | 0.98 | 0.99 | 0.98 | 1.00 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | o_proj | 0.99 | 0.98 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | q_proj | 0.98 | 0.98 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | qkv_fused | 0.99 | 0.99 | 0.99 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | up_proj | 0.97 | 0.94 | 0.98 | 0.98 | cutlass_sm120 / cutlass_sm120 |
| FP8 rw + act quant | qwen25_7b | v_proj | 1.00 | 1.00 | 1.00 | 1.01 | cutlass_sm120 / cutlass_sm120 |

## Canary

- reference 2026-10-23T04:56:12+00:00: 238.92 GB/s, CV 1.09%
- reference 2026-10-23T05:56:18+00:00: 234.61 GB/s, CV 0.51%, drift -1.80%
- per-cell canary over clean cells: median 234.58 GB/s, CV 0.71%, range 223.82 to 238.74 (n=2292)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 2171.0, 'median': 2184.0, 'max': 2190}
- max temperature: 65 C
- clock-event reasons seen: none

## Cross-session comparison

1146 matching clean cells; |diff| median 0.32%, max 29.46%; 36 differ by more than 3% (listed in compare.csv).

## Plots

- [e4_bandwidth](e4_bandwidth.png)
- [e4_pct_b0](e4_pct_b0.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
