# P1 branch guide: where my work lives and how to read it

For: P2 and P3 (reviewing or building on the kernels) and P1 (finding my own way back). Repo: <https://github.com/Bejaine/PAIA-Project>.
Last updated 2026-10-22. Status words: **verified**, **not verified**. Nothing here is a measurement result; results of record come only from P2's gated harness.

## 1. The branches

| Branch | What it is | Head | Links |
|---|---|---|---|
| `main` | the team's branch (P2's harness and results). I never push to it | `05476b2` | [tree](https://github.com/Bejaine/PAIA-Project/tree/main) |
| `p1-s1` | **the pull-request branch**: S1 BF16 GEMV, the harness op, the first integration patch. Two commits on top of `main` | `1859d1e` | [tree](https://github.com/Bejaine/PAIA-Project/tree/p1-s1/src) · [diff vs main](https://github.com/Bejaine/PAIA-Project/compare/main...p1-s1) · [open PR](https://github.com/Bejaine/PAIA-Project/pull/new/p1-s1) |
| `p1-wip-quant` | `p1-s1` plus **one commit** that adds FP8, INT4, tests, the README rewrite and the second patch. **Work in progress, FP8/INT4 not GPU-verified.** It is a straight line on top of `p1-s1`, so it fast-forwards into it once verified | `8878453` | [tree](https://github.com/Bejaine/PAIA-Project/tree/p1-wip-quant/src) · [diff vs p1-s1](https://github.com/Bejaine/PAIA-Project/compare/p1-s1...p1-wip-quant) |
| `results/rtx6000ada` | not mine (another teammate's results branch) | `3cf7803` | [tree](https://github.com/Bejaine/PAIA-Project/tree/results/rtx6000ada) |

Rule: P1 never merges into `main`. P2 (or whoever the team agrees on) merges the pull request. Build outputs (`src/build/`) are git-ignored and never committed.

## 2. The three commits

| Commit | Branch | Content | Touches P2's files? |
|---|---|---|---|
| [`add0de4`](https://github.com/Bejaine/PAIA-Project/commit/add0de4) | `p1-s1` | S1 BF16 GEMV (96 variants), C ABI, ctypes loader, `Op` adapter `ours_bf16.py`, standalone check and bench, CPU checks, tools, README (S1 only) | no |
| [`1859d1e`](https://github.com/Bejaine/PAIA-Project/commit/1859d1e) | `p1-s1` | the integration patch: phase `s1`, one branch in `make_op`, records the chosen variant per cell, mounts `src/` into the container | **yes**: `paia-bench/bench/run.py`, `scripts/drun.sh`, `scripts/hrun.sh` (30 added lines, 4 changed). Please review this commit on its own |
| [`8878453`](https://github.com/Bejaine/PAIA-Project/commit/8878453) | `p1-wip-quant` | FP8 W8A16 and INT4 W4A16 kernels, `ours_quant.py`, quantizers, CPU emulation tests, README v2, `src/patches/integration_quant.patch` (phases `s3`, `s4`) | no (the patch is a file, not applied) |

## 3. Reading order (about 30 minutes)

1. **`src/README.md`**: what exists, formats, designs, a status table, the pre-registered predictions. On `p1-s1` it is the S1-only version; on `p1-wip-quant` it is the full one: [read it](https://github.com/Bejaine/PAIA-Project/blob/p1-wip-quant/src/README.md).
2. **The format definitions**: [`quant_host.h`](https://github.com/Bejaine/PAIA-Project/blob/p1-wip-quant/src/paia_gemv/quant_host.h) (FP8 e4m3 per-row scale; INT4 g128 fp16 scale, zero point 8, natural packing).
3. **One kernel to understand the pattern**: [`gemv_bf16.cuh`](https://github.com/Bejaine/PAIA-Project/blob/p1-s1/src/paia_gemv/gemv_bf16.cuh) (a warp owns R rows; R*U independent 16 B loads before any is consumed). Then FP8 and INT4 on `p1-wip-quant`: [`gemv_fp8.cuh`](https://github.com/Bejaine/PAIA-Project/blob/p1-wip-quant/src/paia_gemv/gemv_fp8.cuh), [`gemv_int4.cuh`](https://github.com/Bejaine/PAIA-Project/blob/p1-wip-quant/src/paia_gemv/gemv_int4.cuh).
4. **How it plugs into your harness**: [`ours_bf16.py`](https://github.com/Bejaine/PAIA-Project/blob/p1-s1/paia-bench/bench/ops/ours_bf16.py), [`ours_quant.py`](https://github.com/Bejaine/PAIA-Project/blob/p1-wip-quant/paia-bench/bench/ops/ours_quant.py), then the diff of [`1859d1e`](https://github.com/Bejaine/PAIA-Project/commit/1859d1e).
5. **Tests**: [`gemv_bf16_check.cu`](https://github.com/Bejaine/PAIA-Project/blob/p1-wip-quant/src/tests/gemv_bf16_check.cu) (standalone, no Python), [`emulate_formats.py`](https://github.com/Bejaine/PAIA-Project/blob/p1-wip-quant/src/tests/emulate_formats.py) (bit-exact CPU port of the decode and dataflow).

## 4. File map (`src/`)

| Path | Branch | What |
|---|---|---|
| `paia_gemv/gemv_bf16.cuh`, `gemv_bf16.cu` | both | S1 kernel and C ABI |
| `paia_gemv/gemv_fp8.cuh`, `gemv_int4.cuh`, `gemv_quant.cu` | wip | FP8, INT4 (fused K2, naive K1, unfused control) and their C ABI |
| `paia_gemv/quant_host.h`, `quant.py` | wip | host and torch quantizers and reference decoders |
| `paia_gemv/loader.py`, `build.sh` | both (wip is newer) | ctypes loader, `.so` build (static cudart, no torch dependency) |
| `tests/gemv_bf16_check.cu` | both (wip is newer) | standalone check and dev bench (`--quant`, `--bench`, `--bench-quant`); refuses to bench while `/tmp/paia_bench.lock` exists |
| `tests/emulate_mapping.py`, `emulate_formats.py`, `quant_host_selftest.cpp` | both / wip / wip | CPU checks, no GPU |
| `tools/nvrtc_check.py`, `dev_sweep_bf16.py`, `compare_to_cublas.py` | both | compile every variant without a GPU; variant sweep (dev only); per-shape table from harness cells |
| `patches/integration_quant.patch` | wip | phases `s3` and `s4` for `run.py`, to apply after commit `1859d1e`: `git apply src/patches/integration_quant.patch` |
| `tuned/` | both | frozen per-shape variant choices (empty until a gated sweep) |

## 5. What is verified, what is not (details in the README)

- **Verified:** every variant of every kernel compiles for sm_89, sm_120 and sm_121 (NVRTC plus ptxas 12.9) and for sm_80/89/120 with nvcc 12.9; the decode, packing and loop logic match the dequantized reference in a CPU port; the BF16 kernels are correct on an RTX 6000 Ada (all 96 variants, worst relative error 3.1e-3).
- **Not verified:** FP8 and INT4 on any GPU; anything on sm_121 (the Spark); the harness phases `s1`, `s3`, `s4` under the gate; that ptxas issues the loads back to back (SASS not read). The Ada correctness run for FP8 and INT4 is queued on turing.
- **Dev numbers** exist for BF16 on the Ada card only (ungated, a hint, not a result); they are in the README and are not to be quoted.

## 6. What I need from P2

1. Review commit `1859d1e` (it edits your `run.py` and run scripts) and tell me if the `src/` mount in `drun.sh` is acceptable.
2. A slot on the Spark for an sm_121 build and a correctness check, then the gated `s1` run (and `s3`, `s4` once they pass).
3. Who presses merge on the pull request.

## 7. How this branch will move

1. FP8 and INT4 pass on a GPU, then `git checkout p1-s1 && git merge --ff-only p1-wip-quant && git push` updates the pull request (no history rewrite).
2. After the Spark runs, frozen variants go into `src/tuned/` and a "result vs prediction" section is added to the README; the predictions themselves are never edited after the fact.
3. This guide gets a line in the table below whenever a branch moves.

| Date | Change |
|---|---|
| 2026-10-22 | `p1-s1` pushed (`add0de4`, `1859d1e`); `p1-wip-quant` pushed (`8878453`); this guide added |
