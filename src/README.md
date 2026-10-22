# src/: P1 kernels (S1 BF16 GEMV first)

Owner: P1. Companion to `paia-bench/` (P2's harness). Status of every claim below is marked
**[verified here]** (done in the authoring sandbox, no GPU), **[not verified]** (needs a GPU run) or **[assumption]**.

## What S1 is

`y[N] = W[N,K] @ x[K]`, bf16 in/out, fp32 accumulate, M = 1. One warp owns R rows; each lane issues R*U independent 16 B
loads per row-chunk before consuming any, so a warp keeps R*U*512 B in flight. No split-K, no shared memory, no atomics.
Motivation: P2's E3 shows cuBLAS `gemvx/v6` (16-lane rows) streams at 68-72% of B0 on GB10 while v7 (32-lane rows) reaches
87-98%, with identical bytes and identical 25% occupancy (164 regs); the gap is bytes in flight per warp.

Knobs (all compiled in, 96 variants): R rows/warp {1,2,4}; U chunks in flight {1,2,4,8}; W warps/block {4,8};
S weight loads via `ld.global.nc.L1::no_allocate` (keeps x in L1); H `__launch_bounds__(W*32,1)` (see next section).

## Finding that shaped the design [verified here, ptxas 12.9 -v, sm_121/sm_120/sm_89]

With default launch bounds, ptxas holds registers near the full-occupancy limit (about 40) and does **not** keep all R*U
loads in flight: R=2,U=8 uses 40 registers where 16 resident 16 B loads alone need 64. With `__launch_bounds__(W*32, 1)`
(H=1) the same variant uses 123 registers, matching the predicted need (4*U*(R+1) = 96 plus addressing). So U alone does
not set loads-in-flight; the register budget does. H is therefore a sweep dimension. `nvrtc_check.py` flags (`*`) every
variant whose registers are below the need, and in those rows the in-flight column is an upper bound, not a measurement.
No variant spills on any of the three architectures (288 builds). Whether H=1 or H=0 is faster is **[not verified]**.

Little's law, **[assumption]**: bytes in flight per SM needed = B0 x L / 48 SMs; for L = 0.6 to 1.2 us that is about 3 to 6 KB
per SM on GB10. Every variant exceeds that on paper, so differences should come from scheduling, LSU queueing and
small-shape tail effects, not from raw in-flight capacity. P2's pointer-chase probe (S5) will replace this range with L.

## Verified here / not verified

| Item | Status |
|---|---|
| Device code compiles for sm_89, sm_120, sm_121 (NVRTC 12.9 -> ptxas 12.9), all 96 variants, `L1::no_allocate` accepted | verified |
| No register spills in any variant on any of the three archs | verified |
| Index logic: every (row, col) read exactly once, every row stored once, incl. K%256 != 0, N not a multiple of R*W, remainder batches (`tests/emulate_mapping.py`, 1056 combinations) | verified (CPU emulation of the control flow) |
| `integration.patch` applies cleanly to P2's tree; patched `run.py` parses | verified |
| Python files byte-compile; `compare_to_cublas.py` runs on synthetic cells | verified |
| nvcc build (CUDA 13.0 in paia-bench:25.11), the .so loads, launches work, graph capture works | **not verified** |
| Numerical correctness on a device (`tests/test_gemv_bf16.py`) | **not verified** |
| That ptxas actually issues the loads back to back (registers say yes for H=1; SASS not inspected) | **not verified**: on turing run `cuobjdump -sass libpaia_gemv.so`, count `LDG.E.128` before the first `FFMA` in one loop body |
| Performance of any variant | **not verified** |

## Build, test, integrate (in this order)

1. Copy `src/` to the repo root and `paia-bench/bench/ops/ours_bf16.py` into P2's tree. Ask P2 to review and apply
   `integration.patch` (adds phase `s1`, the op factory branch, records the chosen variant per cell, and mounts `src/` into the
   container with `PYTHONPATH=/bench:/src`). Own phase name on purpose: `analyze_e2.py` reads phase `e2` only.
2. Build where nvcc exists, once per CPU architecture (GB10 is aarch64, the RTX server x86_64; the .so uses static cudart and
   no torch headers, so it does not depend on the torch version):
   `scripts/drun.sh bash /src/paia_gemv/build.sh` then read `src/build/<arch>/ptxas_report.txt`.
   The RTX server has no Docker and probably no nvcc: build the x86_64 .so on turing (`NVCC=~/cuda-12.9/bin/nvcc`) and copy it.
3. `scripts/drun.sh python /src/tests/test_gemv_bf16.py` (add `--full` for every variant on every shape).
4. Sweep: `scripts/drun.sh python /src/tools/dev_sweep_bf16.py --platform-tag dgxspark_gb10 --all-real --write-tuned`.
   It refuses to run while `/tmp/paia_bench.lock` exists (P2's gated session). Dev numbers are never results.
5. P2 runs `bench.run --phase s1` under the gate (two sessions), then
   `python src/tools/compare_to_cublas.py <e2 sessions> --ours <s1 sessions> --b0 <B0>`.
6. ncu on the winning variant with the same metric set as P2's E3 (long-scoreboard and LSU stall per issue, occupancy,
   sysmem fill bytes / weight bytes).

## Pre-registered predictions for S1 (written before any GPU run)

Setup: GB10, M = 1, graph mode, best variant per shape, B0 = 239.5 GB/s, tau = 1.7 us (v2 probe). Ceiling = B/(B + B_half) x B0.
Percentages are of B0. "cuBLAS" values are P2's E2 v2 numbers.

| Shape class | tau ceiling | cuBLAS | Predicted ours | Confidence |
|---|---|---|---|---|
| Large, cuBLAS on v6 (Llama q/o, qkv; Qwen q/o, qkv, gate/up, gate_up) | 98 to 100% | 68 to 72% | **93 to 99%**, i.e. **1.30 to 1.45x** over cuBLAS | high |
| Large, cuBLAS on v7 (Llama gate/up/down, gate_up; Qwen down) | 99 to 100% | 87 to 98% | 95 to 99%, i.e. **0.98 to 1.10x** | medium |
| Llama k/v (8 MiB) | 95% | v6 ~70% | 88 to 95% | medium |
| Qwen k/v (3.5 MiB) | 90% | v7 87% | 80 to 88%; **no more than 3% better than cuBLAS** | medium |
| lm_head (about 1 GiB) | 99.9% | 72 to 73% | 85 to 97%; P(>= 90%) about 50% | low: cause of cuBLAS's drop unknown, GiB footprints plateau about 4% lower on this machine |

Variant structure: no single dimension (S, H, W) moves a shape >= 24 MiB by more than 5%; R and U matter mainly at <= 8 MiB.
Eager mode: our per-op eager time is within 1.5 us of cuBLAS's (we pay one ctypes call; cuBLAS pays the allocator).
ncu on the best large-shape variant: long-scoreboard stall per issue <= 17 cycles (v7's value) and issue-active 4 to 10%.
RTX PRO 5000 (preliminary tau 1.36 us, B0 1239 GB/s, mismatched software stack): large shapes 88 to 97% of B0, Qwen k/v
55 to 68% (its ceiling is 68.5%); low confidence until P3 has a full, same-stack E1/E2.
