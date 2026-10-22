# src/: P1 kernels (BF16 S1, FP8 B3, INT4 K1/K2)

Owner: P1. Companion to `paia-bench/` (P2's harness). Every claim carries a status:
**[verified]** done and checked in the authoring sandbox or on a GPU, **[dev]** measured on a GPU without the idle gate
(a hint, never a result), **[not verified]** needs a run, **[assumption]** reasoned, unchecked.

## Files

| Path | What |
|---|---|
| `paia_gemv/gemv_bf16.cuh`, `gemv_bf16.cu` | S1 BF16 GEMV (96 variants) and its C ABI |
| `paia_gemv/gemv_fp8.cuh`, `gemv_int4.cuh`, `gemv_quant.cu` | FP8 W8A16 (36 variants), INT4 W4A16 fused K2 (48 variants), naive K1, unfused control, and their C ABI |
| `paia_gemv/quant_host.h` | host quantizers and reference decoders (C++), format definitions |
| `paia_gemv/quant.py`, `loader.py` | torch quantizers (GPU), ctypes loader |
| `paia_gemv/build.sh` | builds `src/build/<arch>/libpaia_gemv.so` (static cudart, no torch dependency) |
| `tests/gemv_bf16_check.cu` | standalone CUDA check and dev benchmark, no Python (`--quant`, `--bench`, `--bench-quant`) |
| `tests/emulate_mapping.py`, `emulate_formats.py`, `quant_host_selftest.cpp` | CPU checks (no GPU) |
| `tests/test_gemv_bf16.py` | torch-based GPU test (for the container; the standalone check is the one to use on turing) |
| `tools/nvrtc_check.py` | compile every variant without a GPU or nvcc: NVRTC then ptxas, registers and spills |
| `tools/dev_sweep_bf16.py` | variant sweep with the harness's own timing (`--kind bf16\|fp8\|int4`), dev only |
| `tools/compare_to_cublas.py` | per-shape table from harness cell JSONs |
| `../paia-bench/bench/ops/ours_bf16.py`, `ours_quant.py` | harness `Op` adapters (phases s1, s3, s4 via the integration patches) |

## Formats (fixed here; `quant_host.h` and `quant.py` implement exactly this)

- **FP8 W8A16.** e4m3fn weights (bias 7, no inf, 0x7F/0xFF NaN, max 448), one fp32 scale per output row, `scale = amax/448`,
  round-to-nearest-even. Activations bf16, fp32 accumulate. Bytes per weight 1 + 4/K. Ideal speedup over BF16 about 2.0x.
- **INT4 W4A16 g128.** Unsigned nibbles, zero point 8, group 128 along K, fp16 scale = `fp16(amax/7)`, `n = clamp(rint(w/scale), -7, 7) + 8`.
  Byte j of a row = weight 2j (low nibble), 2j+1 (high nibble). Scales `[N][K/128]`. 4.125 bits per weight, ideal speedup about 3.88x.
  The asymmetric variants some libraries use (scale plus zero point, 4.25 bits, ideal 3.76x) are **not implemented**; the zero point in
  tinygemm is an additive real-domain offset, not an integer subtract, so it needs its own kernel path and reference.

## Kernel designs

All kernels: one warp owns R rows; a lane issues R*U independent 16 B weight loads before consuming any; fp32 accumulate; warp-shuffle
reduction; one bf16 store per row; no split-K, no shared memory, no atomics. Knobs: R rows/warp, U chunks in flight, W warps/block, H hoist
(`__launch_bounds__(W*32,1)`), S weight loads via `ld.global.nc.L1::no_allocate` (bf16 only; the quantized kernels always use it).

- **S1 BF16.** 256 columns per warp-chunk. Needs K % 8 == 0.
- **FP8.** 512 columns per chunk (16 weights per lane). Decode is pure integer bit work and exact for every finite code: the 7 magnitude
  bits go to bits 13..7 of a half (exponent field = e instead of e+8, so the half equals value/256 exactly, subnormals included), the sign to
  bit 15; the factor 256 is folded into the per-row scale at the end. No per-weight multiply. Needs K % 16 == 0.
- **INT4 fused (K2).** 1024 columns per chunk (32 weights per lane, one fp16 scale per lane per chunk). **No repack is needed**: with natural
  packing, `w & 0x0F0F0F0F` gives the four even-position weights of a 32-bit word and `(w>>4) & 0x0F0F0F0F` the four odd ones; each byte becomes
  the float 8388608+n with one PRMT against 0x4B000000, and subtracting 8388616 gives n-8 exactly. Even weights pair with the low bf16 half of each
  x word, odd weights with the high half. About 3.4 thread-instructions per weight; at B0 that is roughly 12% of issue capacity on GB10 and 14% on Ada
  **[assumption, check with ncu]**. Needs K % 128 == 0.
- **INT4 naive (K1).** One row per warp, one 4 B load per lane per iteration, scale reloaded from global every iteration, shift/mask/I2F per
  weight, no unrolling. The deliberate per-byte-loss control.
- **INT4 unfused.** Kernel A writes a bf16 copy of the weights (2 B per weight), then S1 reads it back: 4.5 B per weight of traffic against BF16's
  2. A sanity control that is slower than BF16 by construction, never a headline.

## Findings that shaped the design

1. **ptxas register policy [verified, ptxas 12.9 -v].** With default launch bounds ptxas holds registers near the full-occupancy limit (about 40) and
   does not keep all R*U loads in flight (R=2,U=8 BF16: 40 registers where 16 resident 16 B loads alone need 64). `__launch_bounds__(W*32,1)` lets
   it allocate what the batch needs (123 registers). H is therefore a sweep dimension. A few INT4 variants with H=0 and R>=4 spill 4 to 24 bytes
   (R=8,U=2,W=4; R=4,U=2,W=4 and 8; R=4,U=4,W=4); the H=1 variants do not spill. The `need*` column of `nvrtc_check.py` is the BF16 formula and is only a
   rough guide for FP8 and INT4.
2. **Dev benchmark on an RTX 6000 Ada [dev, BF16 only so far].** Dev read probe B0 = 891 GB/s (93% of 960 spec). S1 reaches 96 to 99% of B0 on shapes of
   24 MiB or more and is 1.00 to 1.13x over `cublasGemmEx` (n=1); the fixed cost per op (time minus bytes/B0) is about 0.8 to 1.3 us for S1 against 1.8 to
   3.6 us for cuBLAS. On Ada cuBLAS's loss is per op; on GB10 P2 measured it per byte (gemvx v6). `cublasGemmEx` is not the `F.linear` path.

## Status

| Item | Status |
|---|---|
| All variants of all kernels compile for sm_89, sm_120, sm_121 (NVRTC then ptxas 12.9) | verified |
| Index and loop logic (coverage of every element once, remainders, partial chunks) for BF16, FP8, INT4 | verified (CPU emulation) |
| e4m3 decode exact for all finite codes; nibble decode exact; kernel dataflow equals the dequantized reference | verified (bit-exact numpy port of the device code) |
| Host quantizers: round trips, nearest code, ties to even, fp16 conversion over all 65536 patterns | verified (CPU self-test) |
| BF16: all 96 variants correct on RTX 6000 Ada, worst relative error 3.1e-3 | verified (GPU) |
| FP8 and INT4: correctness on a GPU (`gemv_check --quant`) | **not verified** |
| Any performance of FP8 and INT4 | **not verified** |
| sm_121 build and any run on the Spark | **not verified** |
| Harness phases s1, s3, s4 under P2's gate | **not verified** |
| That ptxas issues the loads back to back (SASS not inspected) | **not verified** |

## Build, test

Standalone (no Python, no torch; only the CUDA toolkit):

    NVCC=$HOME/cuda-12.9/bin/nvcc
    $NVCC -O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 \
      -DWITH_CUBLAS -I src/paia_gemv src/tests/gemv_bf16_check.cu src/paia_gemv/gemv_bf16.cu src/paia_gemv/gemv_quant.cu \
      -lcublas -Xlinker -rpath,$HOME/cuda-12.9/lib64 -o src/build/gemv_check        # one line
    src/build/gemv_check              # BF16 correctness
    src/build/gemv_check --quant      # FP8 + INT4 correctness (all variants, naive, unfused)
    src/build/gemv_check --bench-quant

For the Spark use `-gencode arch=compute_121,code=sm_121` inside `paia-bench:25.11` (CUDA 13.0). The Spark is aarch64 and the RTX server x86_64:
build once per CPU architecture. CPU-only checks: `python3 src/tests/emulate_mapping.py`, `emulate_formats.py`, and
`g++ -std=c++17 -I src/paia_gemv src/tests/quant_host_selftest.cpp && ./a.out`.

## Pre-registered predictions (written before any run of the code they concern)

Setup for GB10: M = 1, graph mode, best variant per shape, B0 = 239.5 GB/s, tau = 1.7 us, P2's gated harness. "Ideal" = bytes ratio including metadata
and x/y traffic.

BF16 S1 (committed earlier): large shapes with cuBLAS on v6 reach 93 to 99% of B0 (1.30 to 1.45x over cuBLAS); shapes where cuBLAS is on v7: 0.98 to 1.10x;
Qwen k/v no more than 3% better than cuBLAS; lm_head 85 to 97% with low confidence.

FP8 W8A16 (new): speedup over our best BF16 on shapes of 24 MiB or more is 1.85 to 1.97x (ideal 1.99x, so 93 to 99% of ideal); Llama k/v (8 MiB) 1.6 to 1.85x;
Qwen k/v (3.5 MiB) 1.4 to 1.7x (its tau ceiling limits it). Achieved bandwidth on the actual bytes is 88 to 99% of B0 on large shapes.

INT4 fused K2 (new): speedup over our best BF16 on shapes of 24 MiB or more is 3.2 to 3.8x (ideal 3.88x, so 82 to 98% of ideal); Llama k/v 2.4 to 3.1x;
Qwen k/v 1.8 to 2.4x. Dequant issue load at most about 15% of issue slots on GB10 and 15 to 25% on Ada / RTX PRO 5000 (ncu), so latency exposure, not
issue throughput, is the loss when K2 falls short of ideal.

INT4 naive K1: 0.8 to 1.6x over our BF16 on large shapes (it keeps about one 128 B load in flight per warp, far below the 3 to 6 KB per SM needed),
i.e. at most about 40% of K2's speed. INT4 unfused: 0.35 to 0.5x of BF16 (slower than BF16 by construction; on GB10 writes are about 19% slower than reads).

Variant structure: no single knob changes a shape of 24 MiB or more by more than 5%; R and U matter mainly at 8 MiB and below; the best INT4 variants
use R >= 2 (x conversion and x loads are shared by the rows of a warp).

## Open items

Asymmetric INT4 (scale plus zero point) and M > 1 (the deck promises M = 1..16; the kernels here are M = 1); Triton cross-check of S1; NVFP4; SASS check
(`cuobjdump -sass`, count LDG.E.128 before the first FFMA in one loop body); sm_121 build on the Spark; P2's pointer-chase latency probe to replace the
3 to 6 KB per SM bytes-in-flight range.
