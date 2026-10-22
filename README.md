# PAIA Project: Fused Low-Precision GEMV Kernels for LLM Decode

Models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, batch-1 decode, on NVIDIA DGX Spark (GB10) and RTX PRO 5000 Blackwell.

## Start here

- `docs/RESULTS.md`: numbers of record and where the evidence lives
- `docs/PROGRESS.md`: chronological log of what was done and why
- `docs/PLAN.md`: plan to Evaluation 2
- `docs/P2_BRIEF.md`: the measurement methodology contract

## Layout

- `paia-bench/`: measurement and profiling harness (library baselines, bandwidth probes, analysis). Owner: P2.
  Detailed verified facts: `paia-bench/findings.md`.
- `rtx-5000/`, `paia-results/`: RTX PRO 5000 Blackwell runbook and results. Owner: P3.
- `src/`: our kernels (BF16, FP8, INT4 GEMV), their tests and tools. Owner: P1. See `src/README.md` and `src/BRANCH_GUIDE.md`.
