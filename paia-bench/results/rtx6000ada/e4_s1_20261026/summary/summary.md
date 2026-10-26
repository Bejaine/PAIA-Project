# Summary: e4_s1_20261026

Cells: 641 total; clean 624, unclean_exhausted 17

Schema v2: 5 allocations per cell; spread of per-allocation medians (max - min) / median: median 0.17%, p90 1.24%, max 30.68%.

Clean cells flagged noisy (CV > 3%): 75. With the tail-insensitive robust CV (IQR/1.349/median): 10 above 3%, median 0.11%.

## E4 library FP8 / INT4 vs cuBLAS BF16

B0 = 891.69 GB/s. BF16 reference: E2 session(s) e2v2_s1_20261026 (mean of medians). Theoretical speedup = stored bytes ratio (weights + metadata + x + y). Identity check, max |efficiency - BW ratio| over all rows: 3.33e-16.

### Per real shape, M=1, graph mode

| model | layer | impl | kernel | graph us | BF16 us | GB/s | %B0 | S measured | S theory | efficiency | eager S | isolated S | excess us | per-op us | per-byte us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama31_8b | k_proj | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | 0.58 | - | - | - | - |
| llama31_8b | k_proj | FP8 rw (CUTLASS) | cutlass | 7.90 | 11.84 | 532.0 | 59.7 | 1.50 | 2.00 | 75.0% | - | 0.67 | 3.19 | 2.97 | 0.22 |
| llama31_8b | k_proj | INT4 tinygemm | - | - | - | - | - | - | 3.75 | - | 1.33 | 1.28 | - | - | - |
| llama31_8b | k_proj | INT4 Marlin | - | - | - | - | - | - | 3.87 | - | - | 0.39 | - | - | - |
| llama31_8b | k_proj | FP8 pt + act quant | xmma_sm89 | 19.73 | 11.84 | 213.1 | 23.9 | 0.60 | 2.00 | 30.0% | - | 0.30 | - | - | - |
| llama31_8b | k_proj | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | 0.35 | 0.45 | - | - | - |
| llama31_8b | v_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 10.83 | 11.85 | 387.9 | 43.5 | 1.09 | 2.00 | 54.7% | - | - | 6.12 | 5.89 | 0.23 |
| llama31_8b | v_proj | INT4 tinygemm | - | - | - | - | - | - | 3.75 | - | - | 1.28 | - | - | - |
| llama31_8b | v_proj | INT4 Marlin | - | - | - | - | - | - | 3.87 | - | 0.33 | 0.39 | - | - | - |
| llama31_8b | v_proj | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | 0.24 | - | - | - | - |
| llama31_8b | v_proj | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | 0.35 | 0.45 | - | - | - |
| llama31_8b | o_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 20.99 | 40.40 | 799.8 | 89.7 | 1.92 | 2.00 | 96.2% | 1.88 | - | 2.16 | 1.25 | 0.91 |
| llama31_8b | o_proj | FP8 rw (CUTLASS) | - | - | - | - | - | - | 2.00 | - | 1.83 | 1.18 | - | - | - |
| llama31_8b | o_proj | INT4 tinygemm | - | - | - | - | - | - | 3.76 | - | 3.21 | 2.67 | - | - | - |
| llama31_8b | o_proj | INT4 Marlin | marlin | 12.86 | 40.40 | 674.1 | 75.6 | 3.14 | 3.87 | 81.1% | - | 1.04 | 3.14 | 3.01 | 0.13 |
| llama31_8b | o_proj | FP8 rw + act quant | cutlass | 23.51 | 40.40 | 715.0 | 80.2 | 1.72 | 2.00 | 86.0% | - | - | - | - | - |
| llama31_8b | q_proj | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | 1.88 | - | - | - | - |
| llama31_8b | q_proj | INT4 tinygemm | tinygemm | 11.74 | 40.34 | 760.5 | 85.3 | 3.44 | 3.76 | 91.4% | - | 2.67 | 1.73 | 1.74 | -0.01 |
| llama31_8b | q_proj | INT4 Marlin | marlin | 12.87 | 40.34 | 673.4 | 75.5 | 3.13 | 3.87 | 80.9% | 1.35 | - | 3.15 | 3.02 | 0.13 |
| llama31_8b | q_proj | FP8 pt + act quant | xmma_sm89 | 29.55 | 40.34 | 568.2 | 63.7 | 1.36 | 2.00 | 68.3% | 0.84 | - | - | - | - |
| llama31_8b | q_proj | FP8 rw + act quant | cutlass | 23.52 | 40.34 | 714.8 | 80.2 | 1.72 | 2.00 | 85.9% | - | - | - | - | - |
| llama31_8b | qkv_fused | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | 1.86 | 1.23 | - | - | - |
| llama31_8b | qkv_fused | FP8 rw (CUTLASS) | - | - | - | - | - | - | 2.00 | - | 1.86 | 1.32 | - | - | - |
| llama31_8b | qkv_fused | INT4 tinygemm | tinygemm | 16.84 | 58.66 | 795.2 | 89.2 | 3.48 | 3.76 | 92.6% | - | - | 1.82 | 1.84 | -0.02 |
| llama31_8b | qkv_fused | INT4 Marlin | - | - | - | - | - | - | 3.87 | - | 1.99 | 1.41 | - | - | - |
| llama31_8b | qkv_fused | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | - | 0.77 | - | - | - |
| llama31_8b | qkv_fused | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | - | 1.03 | - | - | - |
| llama31_8b | up_proj | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | 1.87 | 1.49 | - | - | - |
| llama31_8b | up_proj | FP8 rw (CUTLASS) | - | - | - | - | - | - | 2.00 | - | 1.72 | - | - | - | - |
| llama31_8b | up_proj | INT4 tinygemm | tinygemm | 37.68 | 134.06 | 828.9 | 93.0 | 3.56 | 3.76 | 94.6% | - | - | 2.65 | 2.69 | -0.04 |
| llama31_8b | up_proj | INT4 Marlin | marlin | 37.08 | 134.06 | 817.6 | 91.7 | 3.62 | 3.88 | 93.3% | - | 2.12 | 3.08 | 2.63 | 0.45 |
| llama31_8b | up_proj | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | 1.66 | - | - | - | - |
| llama31_8b | up_proj | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | 1.67 | - | - | - | - |
| llama31_8b | gate_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 69.84 | 134.08 | 841.3 | 94.3 | 1.92 | 2.00 | 96.0% | 1.88 | 1.49 | 3.95 | 0.75 | 3.19 |
| llama31_8b | gate_proj | FP8 rw (CUTLASS) | cutlass | 77.19 | 134.08 | 761.9 | 85.4 | 1.74 | 2.00 | 87.0% | - | - | 11.23 | 8.21 | 3.02 |
| llama31_8b | gate_proj | INT4 tinygemm | - | - | - | - | - | - | 3.76 | - | 3.50 | - | - | - | - |
| llama31_8b | gate_proj | INT4 Marlin | marlin | 37.11 | 134.08 | 816.9 | 91.6 | 3.61 | 3.88 | 93.2% | - | - | 3.11 | 2.66 | 0.45 |
| llama31_8b | gate_proj | FP8 pt + act quant | xmma_sm89 | 79.64 | 134.08 | 737.7 | 82.7 | 1.68 | 2.00 | 84.2% | - | 1.11 | - | - | - |
| llama31_8b | gate_proj | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | 1.67 | 1.26 | - | - | - |
| llama31_8b | down_proj | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | 1.89 | 1.48 | - | - | - |
| llama31_8b | down_proj | FP8 rw (CUTLASS) | cutlass | 73.35 | 134.94 | 801.0 | 89.8 | 1.84 | 2.00 | 92.0% | - | - | 7.46 | 4.44 | 3.02 |
| llama31_8b | down_proj | INT4 tinygemm | - | - | - | - | - | - | 3.76 | - | 3.51 | - | - | - | - |
| llama31_8b | down_proj | INT4 Marlin | marlin | 37.35 | 134.94 | 811.6 | 91.0 | 3.61 | 3.88 | 93.2% | - | - | 3.35 | 2.90 | 0.45 |
| llama31_8b | down_proj | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | - | 1.11 | - | - | - |
| llama31_8b | down_proj | FP8 rw + act quant | cutlass | 77.43 | 134.94 | 759.1 | 85.1 | 1.74 | 2.00 | 87.2% | - | 1.31 | - | - | - |
| llama31_8b | gate_up_fused | FP8 pt (_scaled_mm) | xmma_sm89 | 145.21 | 265.89 | 809.2 | 90.7 | 1.83 | 2.00 | 91.6% | - | 1.60 | 13.44 | 7.05 | 6.39 |
| llama31_8b | gate_up_fused | FP8 rw (CUTLASS) | cutlass | 148.93 | 265.89 | 789.8 | 88.6 | 1.79 | 2.00 | 89.4% | 1.77 | - | 17.02 | 10.99 | 6.04 |
| llama31_8b | gate_up_fused | INT4 tinygemm | tinygemm | 71.96 | 265.89 | 868.0 | 97.3 | 3.70 | 3.76 | 98.2% | - | - | 1.91 | 1.99 | -0.08 |
| llama31_8b | gate_up_fused | INT4 Marlin | - | - | - | - | - | - | 3.88 | - | 3.71 | 2.68 | - | - | - |
| llama31_8b | gate_up_fused | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | 1.71 | 1.34 | - | - | - |
| llama31_8b | gate_up_fused | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | 1.75 | 1.49 | - | - | - |
| llama31_8b | lm_head | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | 1.93 | - | - | - | - |
| llama31_8b | lm_head | FP8 rw (CUTLASS) | cutlass | 620.23 | 1179.80 | 848.3 | 95.1 | 1.90 | 2.00 | 95.2% | - | - | 30.21 | 3.21 | 27.00 |
| llama31_8b | lm_head | INT4 tinygemm | tinygemm | 316.28 | 1179.80 | 883.2 | 99.1 | 3.73 | 3.76 | 99.2% | - | 3.69 | 3.00 | 3.34 | -0.34 |
| llama31_8b | lm_head | INT4 Marlin | marlin | 310.07 | 1179.80 | 874.5 | 98.1 | 3.80 | 3.88 | 98.2% | - | 3.48 | 5.99 | 1.97 | 4.02 |
| llama31_8b | lm_head | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | - | 1.77 | - | - | - |
| llama31_8b | lm_head | FP8 rw + act quant | cutlass | 622.53 | 1179.80 | 845.1 | 94.8 | 1.90 | 2.00 | 94.9% | 1.90 | - | - | - | - |
| qwen25_7b | v_proj | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | - | 0.51 | - | - | - |
| qwen25_7b | v_proj | FP8 rw (CUTLASS) | cutlass | 6.18 | 5.92 | 298.1 | 33.4 | 0.96 | 2.00 | 48.0% | 0.49 | 0.61 | 4.11 | 4.02 | 0.09 |
| qwen25_7b | v_proj | INT4 tinygemm | tinygemm | 4.59 | 5.92 | 214.0 | 24.0 | 1.29 | 3.74 | 34.5% | 1.07 | 1.18 | 3.49 | 3.49 | -0.00 |
| qwen25_7b | v_proj | INT4 Marlin | marlin | 10.82 | 5.92 | 88.2 | 9.9 | 0.55 | 3.85 | 14.2% | - | 0.36 | 9.75 | 9.74 | 0.01 |
| qwen25_7b | v_proj | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | - | 0.28 | - | - | - |
| qwen25_7b | v_proj | FP8 rw + act quant | cutlass | 7.99 | 5.92 | 230.9 | 25.9 | 0.74 | 1.99 | 37.2% | - | - | - | - | - |
| qwen25_7b | k_proj | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | 0.47 | - | - | - | - |
| qwen25_7b | k_proj | FP8 rw (CUTLASS) | cutlass | 6.18 | 5.94 | 297.9 | 33.4 | 0.96 | 2.00 | 48.1% | 0.49 | - | 4.12 | 4.02 | 0.09 |
| qwen25_7b | k_proj | INT4 tinygemm | tinygemm | 4.59 | 5.94 | 214.1 | 24.0 | 1.29 | 3.74 | 34.6% | 1.07 | 1.18 | 3.49 | 3.49 | -0.00 |
| qwen25_7b | k_proj | INT4 Marlin | - | - | - | - | - | - | 3.85 | - | 0.27 | 0.37 | - | - | - |
| qwen25_7b | k_proj | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | 0.20 | 0.28 | - | - | - |
| qwen25_7b | k_proj | FP8 rw + act quant | cutlass | 8.00 | 5.94 | 230.8 | 25.9 | 0.74 | 1.99 | 37.3% | - | - | - | - | - |
| qwen25_7b | o_proj | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | - | 1.01 | - | - | - |
| qwen25_7b | o_proj | FP8 rw (CUTLASS) | - | - | - | - | - | - | 2.00 | - | 1.59 | 1.11 | - | - | - |
| qwen25_7b | o_proj | INT4 tinygemm | tinygemm | 9.64 | 32.18 | 709.1 | 79.5 | 3.34 | 3.76 | 88.8% | 3.10 | 2.35 | 1.98 | 1.98 | -0.01 |
| qwen25_7b | o_proj | INT4 Marlin | - | - | - | - | - | - | 3.87 | - | 1.07 | 0.89 | - | - | - |
| qwen25_7b | o_proj | FP8 pt + act quant | xmma_sm89 | 24.89 | 32.18 | 516.6 | 57.9 | 1.29 | 2.00 | 64.7% | - | 0.57 | - | - | - |
| qwen25_7b | qkv_fused | FP8 pt (_scaled_mm) | xmma_sm89 | 20.71 | 40.26 | 797.9 | 89.5 | 1.94 | 2.00 | 97.2% | - | - | 2.18 | 1.28 | 0.90 |
| qwen25_7b | qkv_fused | FP8 rw (CUTLASS) | - | - | - | - | - | - | 2.00 | - | 1.86 | 1.20 | - | - | - |
| qwen25_7b | qkv_fused | INT4 Marlin | - | - | - | - | - | - | 3.87 | - | 1.37 | - | - | - | - |
| qwen25_7b | qkv_fused | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | - | 0.65 | - | - | - |
| qwen25_7b | qkv_fused | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | 1.16 | - | - | - | - |
| qwen25_7b | q_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 16.59 | 32.19 | 774.9 | 86.9 | 1.94 | 2.00 | 97.0% | 1.71 | 1.01 | 2.17 | 1.47 | 0.70 |
| qwen25_7b | q_proj | FP8 rw (CUTLASS) | cutlass | 17.11 | 32.19 | 752.2 | 84.4 | 1.88 | 2.00 | 94.2% | 1.59 | - | 2.68 | 2.02 | 0.66 |
| qwen25_7b | q_proj | INT4 tinygemm | tinygemm | 9.64 | 32.19 | 709.1 | 79.5 | 3.34 | 3.76 | 88.8% | - | 2.35 | 1.97 | 1.98 | -0.01 |
| qwen25_7b | q_proj | INT4 Marlin | marlin | 10.61 | 32.19 | 625.6 | 70.2 | 3.03 | 3.87 | 78.3% | - | 0.88 | 3.17 | 3.07 | 0.10 |
| qwen25_7b | q_proj | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | 0.66 | - | - | - | - |
| qwen25_7b | q_proj | FP8 rw + act quant | cutlass | 18.98 | 32.19 | 678.1 | 76.1 | 1.70 | 2.00 | 84.9% | - | - | - | - | - |
| qwen25_7b | down_proj | FP8 pt (_scaled_mm) | xmma_sm89 | 80.93 | 156.19 | 839.3 | 94.1 | 1.93 | 2.00 | 96.5% | 1.90 | - | 4.76 | 1.06 | 3.69 |
| qwen25_7b | down_proj | FP8 rw (CUTLASS) | cutlass | 82.89 | 156.19 | 819.6 | 91.9 | 1.88 | 2.00 | 94.2% | 1.85 | 1.58 | 6.70 | 3.21 | 3.49 |
| qwen25_7b | down_proj | INT4 tinygemm | tinygemm | 43.07 | 156.19 | 838.6 | 94.0 | 3.63 | 3.76 | 96.4% | 3.54 | 3.30 | 2.56 | 2.61 | -0.04 |
| qwen25_7b | down_proj | INT4 Marlin | marlin | 42.88 | 156.19 | 817.5 | 91.7 | 3.64 | 3.88 | 94.0% | 3.61 | 2.21 | 3.57 | 3.05 | 0.52 |
| qwen25_7b | down_proj | FP8 pt + act quant | xmma_sm89 | 105.77 | 156.19 | 642.4 | 72.0 | 1.48 | 2.00 | 73.9% | - | 1.18 | - | - | - |
| qwen25_7b | down_proj | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | - | 1.38 | - | - | - |
| qwen25_7b | up_proj | FP8 rw (CUTLASS) | cutlass | 91.94 | 154.49 | 739.7 | 83.0 | 1.68 | 2.00 | 84.1% | 1.67 | 1.45 | 15.67 | 12.18 | 3.49 |
| qwen25_7b | up_proj | INT4 tinygemm | tinygemm | 42.88 | 154.49 | 842.3 | 94.5 | 3.60 | 3.76 | 95.8% | - | 3.31 | 2.38 | 2.42 | -0.04 |
| qwen25_7b | up_proj | INT4 Marlin | marlin | 42.59 | 154.49 | 823.1 | 92.3 | 3.63 | 3.88 | 93.6% | 3.59 | 2.22 | 3.27 | 2.75 | 0.52 |
| qwen25_7b | up_proj | FP8 pt + act quant | xmma_sm89 | 88.02 | 154.49 | 771.9 | 86.6 | 1.76 | 2.00 | 87.8% | 1.74 | - | - | - | - |
| qwen25_7b | up_proj | FP8 rw + act quant | cutlass | 93.69 | 154.49 | 726.0 | 81.4 | 1.65 | 2.00 | 82.6% | - | - | - | - | - |
| qwen25_7b | gate_proj | FP8 rw (CUTLASS) | - | - | - | - | - | - | 2.00 | - | 1.68 | 1.45 | - | - | - |
| qwen25_7b | gate_proj | INT4 tinygemm | - | - | - | - | - | - | 3.76 | - | 3.55 | - | - | - | - |
| qwen25_7b | gate_proj | INT4 Marlin | - | - | - | - | - | - | 3.88 | - | - | 2.24 | - | - | - |
| qwen25_7b | gate_proj | FP8 pt + act quant | xmma_sm89 | 87.97 | 154.50 | 772.3 | 86.6 | 1.76 | 2.00 | 87.8% | - | 1.19 | - | - | - |
| qwen25_7b | gate_proj | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | 1.64 | 1.28 | - | - | - |
| qwen25_7b | gate_up_fused | FP8 rw (CUTLASS) | - | - | - | - | - | - | 2.00 | - | 1.79 | 1.66 | - | - | - |
| qwen25_7b | gate_up_fused | INT4 tinygemm | - | - | - | - | - | - | 3.76 | - | 3.66 | 3.51 | - | - | - |
| qwen25_7b | gate_up_fused | INT4 Marlin | - | - | - | - | - | - | 3.88 | - | 3.71 | - | - | - | - |
| qwen25_7b | gate_up_fused | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | - | 1.46 | - | - | - |
| qwen25_7b | gate_up_fused | FP8 rw + act quant | cutlass | 171.82 | 306.40 | 791.7 | 88.8 | 1.78 | 2.00 | 89.3% | 1.77 | 1.53 | - | - | - |
| qwen25_7b | lm_head | FP8 pt (_scaled_mm) | - | - | - | - | - | - | 2.00 | - | 1.97 | 1.91 | - | - | - |
| qwen25_7b | lm_head | FP8 rw (CUTLASS) | cutlass | 642.87 | 1223.99 | 849.2 | 95.2 | 1.90 | 2.00 | 95.3% | - | 1.86 | 30.65 | 2.63 | 28.02 |
| qwen25_7b | lm_head | INT4 tinygemm | - | - | - | - | - | - | 3.76 | - | - | 3.69 | - | - | - |
| qwen25_7b | lm_head | INT4 Marlin | - | - | - | - | - | - | 3.88 | - | 3.81 | - | - | - | - |
| qwen25_7b | lm_head | FP8 pt + act quant | - | - | - | - | - | - | 2.00 | - | 1.94 | 1.81 | - | - | - |
| qwen25_7b | lm_head | FP8 rw + act quant | - | - | - | - | - | - | 2.00 | - | - | 1.82 | - | - | - |

%B0 is also the efficiency against an ideal BF16 kernel at B0 (cuBLAS BF16 itself runs at about 70% of B0 on 13 of 20 shapes, so efficiency against cuBLAS can exceed 100%). excess = t - B / B0; per-byte = B (1/R_asym - 1/B0) with the asymptotic rate of the kernel dispatched for that shape (below; '-' when that kernel has fewer than 3 ops >= 64 MiB); per-op = excess - per-byte.

### Asymptotic streaming rate per implementation (graph, M=1, ops >= 64 MiB)

| impl | R_asym GB/s | CI95 | %B0 | intercept us | points | kernels |
|---|---|---|---|---|---|---|
| INT4 Marlin (marlin) | 880.0 | 851.3 to 2362.1 | 98.7 | 1.95 | 4 | marlin |
| INT4 tinygemm (tinygemm) | 892.7 | 852.0 to unbounded | 100.1 | 4.35 | 6 | tinygemm |
| FP8 pt (_scaled_mm) (xmma_sm89) | 850.5 | 813.9 to 1005.5 | 95.4 | -3.72 | 7 | xmma_sm89 |
| FP8 rw (CUTLASS) (cutlass) | 852.7 | 718.6 to 867.4 | 95.6 | 8.16 | 10 | cutlass |

### Per-kernel fits t = tau + B / R (M=1; informational, misfits are expected for tiled kernels)

| impl | kernel | mode | points | tau us | tau CI95 | R_inf GB/s | R_inf %B0 | B_half MiB | rms rel resid |
|---|---|---|---|---|---|---|---|---|---|
| INT4 Marlin | marlin | eager | 22 | 27.50 | 22.73 to 32.62 | 1432.6 | 160.7 | 37.576 | 23.0% misfit |
| INT4 Marlin | marlin | graph | 21 | 8.04 | 5.65 to 11.38 | 1052.6 | 118.0 | 8.069 | 23.4% misfit |
| INT4 Marlin | marlin | isolated | 27 | 49.48 | 44.92 to 54.01 | 1078.0 | 120.9 | 50.867 | 13.4% misfit |
| INT4 tinygemm | tinygemm | eager | 26 | 7.30 | 6.10 to 8.20 | 1049.1 | 117.7 | 7.308 | 15.0% misfit |
| INT4 tinygemm | tinygemm | graph | 25 | 3.30 | 2.84 to 4.42 | 919.4 | 103.1 | 2.890 | 19.0% misfit |
| INT4 tinygemm | tinygemm | isolated | 27 | 14.27 | 12.51 to 15.72 | 999.3 | 112.1 | 13.601 | 11.7% misfit |
| FP8 pt (_scaled_mm) | xmma_sm89 | eager | 30 | 15.47 | 12.38 to 17.87 | 1128.8 | 126.6 | 16.653 | 20.9% misfit |
| FP8 pt (_scaled_mm) | xmma_sm89 | graph | 22 | 7.27 | 6.47 to 7.85 | 970.1 | 108.8 | 6.726 | 10.3% misfit |
| FP8 pt (_scaled_mm) | xmma_sm89 | isolated | 21 | 38.03 | 35.44 to 40.05 | 981.6 | 110.1 | 35.603 | 9.1% misfit |
| FP8 rw (CUTLASS) | cutlass | eager | 31 | 14.43 | 10.63 to 17.01 | 1078.2 | 120.9 | 14.839 | 21.6% misfit |
| FP8 rw (CUTLASS) | cutlass | graph | 27 | 4.79 | 3.90 to 5.85 | 862.4 | 96.7 | 3.937 | 18.6% misfit |
| FP8 rw (CUTLASS) | cutlass | isolated | 28 | 31.23 | 28.95 to 33.07 | 893.6 | 100.2 | 26.616 | 9.5% misfit |

### Activation quantization cost (full op minus GEMM only), M=1

| impl | mode | median us | min us | max us | shapes |
|---|---|---|---|---|---|
| FP8 pt + act quant | isolated | 33.57 | 32.83 | 35.55 | 7 |
| FP8 pt + act quant | eager | 27.23 | 8.86 | 30.55 | 5 |
| FP8 pt + act quant | graph | 17.32 | 9.81 | 24.84 | 2 |
| FP8 rw + act quant | isolated | 15.55 | 15.17 | 16.70 | 6 |
| FP8 rw + act quant | eager | 2.15 | 1.84 | 13.28 | 5 |
| FP8 rw + act quant | graph | 1.84 | 1.74 | 4.07 | 6 |

### Per-token GEMV cost (graph, M=1; all layers + lm_head)

| model | layers | impl | ms/token | GiB/token | ideal ms at B0 | % of ideal | speedup vs BF16 | theory |
|---|---|---|---|---|---|---|---|---|
| llama31_8b | unfused | cuBLAS BF16 | 17.42 | 13.98 | 16.84 | 96.7 | 1.00 | 1.00 |
| llama31_8b | fused | cuBLAS BF16 | 17.18 | 13.98 | 16.84 | 98.0 | 1.00 | 1.00 |
| qwen25_7b | unfused | cuBLAS BF16 | 16.38 | 13.17 | 15.86 | 96.8 | 1.00 | 1.00 |
| qwen25_7b | fused | cuBLAS BF16 | 16.20 | 13.17 | 15.86 | 97.9 | 1.00 | 1.00 |

### M sweep (graph): t(M) / t(1)

| impl | model | layer | M=2 | M=4 | M=8 | M=16 | kernels M=1 / 16 |
|---|---|---|---|---|---|---|---|
| FP8 pt (_scaled_mm) | llama31_8b | gate_proj | - | - | 1.00 | 1.00 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | llama31_8b | gate_up_fused | - | 1.00 | - | 1.00 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | llama31_8b | o_proj | 1.00 | 1.00 | 1.00 | - | xmma_sm89 / None |
| FP8 pt (_scaled_mm) | llama31_8b | v_proj | - | - | - | - | xmma_sm89 / None |
| FP8 pt (_scaled_mm) | qwen25_7b | down_proj | 1.00 | - | 1.03 | 1.03 | xmma_sm89 / xmma_sm89 |
| FP8 pt (_scaled_mm) | qwen25_7b | q_proj | - | - | 1.01 | - | xmma_sm89 / None |
| FP8 pt (_scaled_mm) | qwen25_7b | qkv_fused | 1.00 | 0.99 | 1.00 | 1.01 | xmma_sm89 / xmma_sm89 |
| FP8 rw (CUTLASS) | llama31_8b | down_proj | 1.00 | - | 1.00 | - | cutlass / None |
| FP8 rw (CUTLASS) | llama31_8b | gate_proj | 1.01 | - | - | 1.01 | cutlass / cutlass |
| FP8 rw (CUTLASS) | llama31_8b | gate_up_fused | - | - | 1.01 | - | cutlass / None |
| FP8 rw (CUTLASS) | llama31_8b | k_proj | 1.00 | 1.00 | 1.01 | 1.02 | cutlass / cutlass |
| FP8 rw (CUTLASS) | llama31_8b | lm_head | - | - | - | 1.02 | cutlass / cutlass |
| FP8 rw (CUTLASS) | qwen25_7b | down_proj | - | 1.01 | 1.01 | 1.02 | cutlass / cutlass |
| FP8 rw (CUTLASS) | qwen25_7b | k_proj | - | - | - | - | cutlass / None |
| FP8 rw (CUTLASS) | qwen25_7b | lm_head | - | - | 1.01 | - | cutlass / None |
| FP8 rw (CUTLASS) | qwen25_7b | q_proj | 1.00 | 1.01 | - | - | cutlass / None |
| FP8 rw (CUTLASS) | qwen25_7b | up_proj | 1.00 | 1.01 | 1.01 | - | cutlass / None |
| FP8 rw (CUTLASS) | qwen25_7b | v_proj | - | 1.00 | 1.01 | - | cutlass / None |
| INT4 tinygemm | llama31_8b | gate_up_fused | 1.01 | - | 1.56 | - | tinygemm / None |
| INT4 tinygemm | llama31_8b | lm_head | 1.01 | - | 1.61 | 2.71 | tinygemm / tinygemm |
| INT4 tinygemm | llama31_8b | q_proj | - | 1.02 | 1.18 | - | tinygemm / None |
| INT4 tinygemm | llama31_8b | qkv_fused | - | 1.06 | - | - | tinygemm / None |
| INT4 tinygemm | llama31_8b | up_proj | 1.00 | 1.06 | - | - | tinygemm / None |
| INT4 tinygemm | qwen25_7b | down_proj | - | 1.14 | - | - | tinygemm / None |
| INT4 tinygemm | qwen25_7b | k_proj | 1.06 | 1.12 | - | 1.65 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | o_proj | - | - | 1.18 | - | tinygemm / None |
| INT4 tinygemm | qwen25_7b | q_proj | - | - | - | - | tinygemm / None |
| INT4 tinygemm | qwen25_7b | up_proj | - | 1.07 | - | 2.86 | tinygemm / tinygemm |
| INT4 tinygemm | qwen25_7b | v_proj | - | - | - | 1.65 | tinygemm / tinygemm |
| INT4 Marlin | llama31_8b | down_proj | 1.00 | - | - | - | marlin / None |
| INT4 Marlin | llama31_8b | gate_proj | - | 1.00 | 1.00 | - | marlin / None |
| INT4 Marlin | llama31_8b | lm_head | 1.00 | 1.01 | 1.02 | 1.04 | marlin / marlin |
| INT4 Marlin | llama31_8b | o_proj | 1.00 | - | - | - | marlin / None |
| INT4 Marlin | llama31_8b | q_proj | 1.00 | 1.00 | 1.01 | - | marlin / None |
| INT4 Marlin | llama31_8b | up_proj | 1.00 | 1.00 | 1.00 | - | marlin / None |
| INT4 Marlin | qwen25_7b | down_proj | 1.00 | - | 1.01 | - | marlin / None |
| INT4 Marlin | qwen25_7b | q_proj | 1.00 | - | - | 1.05 | marlin / marlin |
| INT4 Marlin | qwen25_7b | up_proj | - | - | - | 1.02 | marlin / marlin |
| INT4 Marlin | qwen25_7b | v_proj | 1.00 | - | - | 1.09 | marlin / marlin |
| FP8 pt + act quant | llama31_8b | gate_proj | 1.00 | 1.00 | 1.01 | 1.01 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | llama31_8b | k_proj | 1.01 | 1.01 | - | 1.02 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | llama31_8b | q_proj | 1.02 | 1.03 | - | 1.03 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | qwen25_7b | down_proj | 1.00 | 1.03 | 1.03 | 1.03 | xmma_sm89 / xmma_sm89 |
| FP8 pt + act quant | qwen25_7b | gate_proj | 1.00 | 1.00 | 1.01 | - | xmma_sm89 / None |
| FP8 pt + act quant | qwen25_7b | o_proj | - | - | 1.02 | - | xmma_sm89 / None |
| FP8 pt + act quant | qwen25_7b | up_proj | 1.00 | 1.00 | - | - | xmma_sm89 / None |
| FP8 rw + act quant | llama31_8b | down_proj | - | 1.00 | 1.00 | 1.00 | cutlass / cutlass |
| FP8 rw + act quant | llama31_8b | lm_head | 1.00 | - | - | - | cutlass / None |
| FP8 rw + act quant | llama31_8b | o_proj | - | 1.00 | 1.00 | - | cutlass / None |
| FP8 rw + act quant | llama31_8b | q_proj | - | 1.00 | - | 1.01 | cutlass / cutlass |
| FP8 rw + act quant | qwen25_7b | gate_up_fused | 1.00 | 1.01 | 1.01 | - | cutlass / None |
| FP8 rw + act quant | qwen25_7b | k_proj | - | - | 1.02 | - | cutlass / None |
| FP8 rw + act quant | qwen25_7b | q_proj | - | - | - | - | cutlass / None |
| FP8 rw + act quant | qwen25_7b | up_proj | - | 1.01 | - | - | cutlass / None |
| FP8 rw + act quant | qwen25_7b | v_proj | - | 1.01 | 1.02 | 1.03 | cutlass / cutlass |

Kernel names: 635 cells filled from `dispatch_table_e2_e4.json` (profiler capture lost on this stack, findings.md); 5 recorded cells matched it exactly, 1 partial recordings were a subset of it and were completed.

## Canary

- reference 2026-10-26T04:20:13+05:30: 824.83 GB/s, CV 0.02%
- reference 2026-10-26T05:20:23+05:30: 824.11 GB/s, CV 0.02%, drift -0.09%
- reference 2026-10-26T06:20:31+05:30: 823.84 GB/s, CV 0.13%, drift -0.03%
- per-cell canary over clean cells: median 824.13 GB/s, CV 0.18%, range 822.86 to 827.79 (n=1248)

## Telemetry (clean cells)

- SM clock median per cell: {'min': 840, 'median': 2610.0, 'max': 2685.0}
- max temperature: 70 C
- clock-event reasons seen: ['gpu_idle', 'sw_power_cap']

## Plots

- [e4_bandwidth](e4_bandwidth.png)
- [e4_pct_b0](e4_pct_b0.png)

Files: `fits.json`, `cells.csv` (clean cells), `unclean.csv` (contaminated / failed / exhausted).
