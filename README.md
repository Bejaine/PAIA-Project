# PAIA Project: Fused Low-Precision GEMV Kernels for LLM Decode

Models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, batch-1 decode.

## Layout

- `paia-bench/`: measurement and profiling harness (library baselines, bandwidth probes, analysis). Owner: P2.
- `src/`: our kernels (BF16, FP8, INT4 GEMV), their tests and tools. Owner: P1. See `src/README.md` and `src/BRANCH_GUIDE.md`.
