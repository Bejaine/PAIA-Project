# PAIA Plan (from Evaluation 1 to Evaluation 2)

The forward plan. Update it when priorities change; record what actually happened in `docs/PROGRESS.md`.
Evaluation 2 window: November 9 to 21, 2026.

## Where we stand

- GB10: E0, E1 and E2 are done to the standard of record (schema v2, two sessions each). E3 is done for the key question
  (why cuBLAS v6 is slow). E4 and E5 have not started; every library path they need is verified to work.
- RTX PRO 5000: environment done; E1 preliminary (80 of 360 cells) on a different software stack, with an unstable SM clock.
- Scientific direction: on GB10 the quantization gap looks **per byte (R)**, not per op (tau): B_half is only 0.39 MiB,
  and the largest measured loss is cuBLAS streaming at about 70% of B0 on 13 of 20 shapes. E4 decides whether the library
  quantized kernels show the same pattern.

## Step 0: Evaluation 1 deck (on hold)

Set aside by P2 on 2026-10-23; the focus is the GB10 measurements and kernels. The draft stays in the repo root (untracked).

## Step 1 (now): E4 library FP8 / INT4 on GB10 (code done 2026-10-23; sessions next), RTX parity (P3, on hold)

- **Ops** (in `paia-bench/bench/ops/`, same interface as `cublas_bf16.py`, correctness check first):
  - `scaled_mm_fp8.py`: `torch._scaled_mm`, e4m3 weights `[N, K]` passed as `W.t()`, per-tensor fp32 scales, per-tensor
    activation quantization, `out_dtype=bfloat16` (M = 1 accepted on GB10). Activation-quant kernel timed separately and
    as part of a "full op" variant.
  - Row-wise FP8: vLLM `ops.cutlass_scaled_mm` on GB10 (torch's row-wise path is broken on sm_121).
  - `tinygemm_int4.py`: `_convert_weight_to_int4pack(uint8 [N, K/2], innerKTiles=8)` + `_weight_int4pack_mm(x, packed,
    128, scales_and_zeros [K/128, N, 2] bf16)`; convention `w = (q - 8) * scale + zero`.
  - `marlin_int4.py`: vLLM `apply_gptq_marlin_linear` with `marlin_quantize(W [K, N], scalar_types.uint4b8, 128)`.
- **Bytes include metadata** (scales, zeros); theoretical speedup computed from stored byte counts.
- **Cells:** 20 real shapes x {M = 1 in 3 modes, M = 2..16 graph}; add a `phase e4` cell builder; 2 sessions, schema v2.
- **Analysis (`analyze_e4.py`):** per shape: BF16 time (from E2 v2), quantized time, measured speedup, theoretical, efficiency,
  BW_q / BW_16; check the identity (efficiency == BW_q / BW_16); per-kernel Hockney fits (tau and R per library kernel).
- **E3 extension:** run `ncu_gemv_variants.sh`-style profiles for each library kernel on 2 to 4 shapes (bytes match, stalls).
- **RTX (P3):** finish a full E1, then E2, ideally on a stack matching GB10. Docker is not available on that server, so the
  options are: (a) ask the admin for Docker, (b) pin both platforms to the same PyPI torch build (then GB10 also needs a
  matching run), or (c) document the stack difference. Clock control (`nvidia-smi -lgc` via admin) or stratify by clock.
  Commit RTX predictions for E2, E4 and E5 before those phases run.

## Step 2 (Oct 29 to Nov 4): P1 kernel v1 + harness integration, E5

- P1: BF16 GEMV (ablation S1: 32-lane rows, 128-bit loads; S2: register and occupancy tuning). Target >= 90% of B0 on all
  20 shapes.
- P2: wrap P1's kernels as harness ops (`bench/ops/p1_*.py`), same cells, same correctness checks; Nsight per kernel.
- E5 (P2): download Qwen2.5-7B-Instruct (about 15 GB; Llama-3.1-8B is cached), vLLM `enforce_eager` True vs False,
  128 greedy tokens, nsys `--trace=cuda,nvtx --cuda-graph-trace=node`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`; per-token
  breakdown; compare with bytes/token / B0 (Llama floor 15.9 tok/s at 239 GB/s, Qwen 16.9 tok/s).

## Step 3 (Nov 5 to 11): fused quantization and ablations

- P1: S3 fused FP8 / INT4 dequant in registers; S4 one kernel per op (fused QKV and gate-up, no split-K).
- P2: ablation matrix on GB10 (2 sessions), trade-offs (group size 64/128/256, rows per warp vs occupancy, split-K vs single
  pass for small N, FP8 vs INT4 error); P3 the same on RTX.

## Step 4 (Nov 12 to 21): whole token and Evaluation 2

- S5: P1 kernels inside vLLM (or a minimal decode loop) with CUDA graphs; tokens/s vs library baselines.
- Cross-platform analysis: tau, R_inf, B_half per kernel on both GPUs; hypothesis verdict.
- Phase 6 report: summaries, cross-session comparisons, final deck.

## Standing rules (apply to every step)

See `handoff.md` (local, not committed) or `docs/P2_BRIEF.md`: ask before each new step; accuracy over convenience;
shared-machine protocol; no em dashes; PRs only, user merges; schema bump on any methodology change.
