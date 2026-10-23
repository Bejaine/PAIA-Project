# P2 Brief: PAIA Baseline Benchmarks (original instructions, preserved)

This is the brief P2 gave at the start of the project (2026-10-16). It is the methodology contract for `paia-bench`.
Where the work deliberately deviates from it, `docs/PROGRESS.md` and `paia-bench/findings.md` say so and why.

---

## 0. Ground rules

1. **No project kernels.** P1 writes all custom GEMV kernels. P2 tooling may only be trivial measurement tooling:
   bandwidth probes, timing harnesses, monitors, analysis scripts.
2. **Verify, don't assume.** Facts in this brief may be outdated or wrong for the machine (API signatures, sm_121 support,
   counter names, NVML fields). Check each, adapt, record in `findings.md`. Never invent a number.
3. **The machine is shared.** Never kill or renice other users' processes. Cap memory (default 12 GB for microbenchmarks).
   Ask before any `sudo` command or system-level install.
4. **Code must be platform-agnostic.** P3 runs the same repo on a discrete RTX 5000-class GPU. Read every hardware property
   at runtime. Hardcode nothing about GB10.
5. **Work in phases.** At each gate, print a short summary (tables, plot paths) and wait for P2's go-ahead.
6. **Writing style:** no em dashes in any file or message.

## 1. Project context

**Title:** Fused Low-Precision GEMV Kernels for LLM Decode.
**Models:** Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. Batch-1 decode is the main regime.

**Premise.** At batch 1, decode is memory-bound: each token reads every weight once. A bandwidth-only model predicts
quantization speedups equal to the byte reduction (FP8 about 2x; 4-bit about 3.6x to 3.9x with scale metadata).
Measured speedups fall well short.

**Model of the gap (per op):**

```
t_op(B) = tau + B / R_inf          B_half = tau * R_inf          efficiency = B / (B + B_half)
S_actual = (bytes_16 / bytes_q) * (BW_q / BW_16)
```

- `R_inf`: asymptotic streaming bandwidth; a per-byte loss lowers R_inf.
- `tau`: fixed per-op cost (launch, inter-kernel gap, ramp-up, tail); a per-op loss raises tau.
- Quantization shrinks B but not tau, so small ops slide down the efficiency curve.

**Hypothesis (untested at the start).** Since B_half = tau * R_inf, a low-bandwidth platform like GB10 (273 GB/s spec)
should be less tau-sensitive than a fast discrete GPU, unless GB10's tau is larger. Measuring tau is the most important
early experiment.

## 2. Experiments

| ID | Name | Implementation | Question |
|---|---|---|---|
| E0 | Fingerprint | scripts + NVML + `nvidia-smi -q` | What exactly is this machine and stack? |
| E1 (B0) | Achievable bandwidth | torch `sum`, `copy_`, `fill_`; optional Triton read probe | Real ceiling R_inf and floor tau |
| E2 (B1) | cuBLAS BF16 GEMV | `F.linear(x, W)`, x `[M,K]`, W `[N,K]` | Production yardstick on real shapes; tau and R_inf of a real GEMV |
| E3 | ncu smoke test | Nsight Compute on E2 shapes | Can we count DRAM bytes? Do they match expected bytes? |
| E4 | Library FP8 and INT4 | `torch._scaled_mm` (FP8 e4m3); tinygemm int4 (`_convert_weight_to_int4pack` + `_weight_int4pack_mm`); Marlin via vLLM | Measured vs metadata-corrected theoretical speedups |
| E5 (B5) | Whole-token BF16 decode | vLLM `enforce_eager=True` vs `False` (else HF transformers); nsys | Kernels per token, matvec share, inter-kernel idle, tokens/s |

## 3. Shared-machine protocol

- **No pause/resume** (SIGSTOP) of measurements: contaminated windows, changed L2/clock/thermal state, held memory.
- **Cells:** one configuration (op, impl, shape, M, timing mode, hot/cold), target <= 15 s, written atomically to
  `results/<platform_tag>/<session_id>/cells/<cell_id>.json` (temp file + rename; `cell_id` = stable hash of config).
  The runner is resumable: skips `status == "clean"`; a contaminated cell is retried up to `--max-retries` (5), then
  `unclean_exhausted`.
- **Idle gate before each cell:** no foreign GPU compute process (NVML; foreign = not in our process tree); device GPU
  utilization < 5% over 2 s; other users' processes < 2 cores total (psutil); canary passes. Otherwise back off
  (30 s doubling to 10 min), log, free large buffers.
- **Live monitor during each cell (100 ms):** NVML util, foreign processes, SM/mem clock, temperature, power, throttle
  reasons; CPU use by other users, `/proc/loadavg`. Any foreign activity marks the cell contaminated. N/A NVML fields are
  recorded as `null`.
- **Bandwidth canary (critical on GB10, CPU and GPU share LPDDR5x):** cold read of rotating buffers (512 MB, 10 reps).
  Reference at session start = median of 20 runs with CV < 2%. Run before and after every cell; below 0.95 x reference
  marks the cell contaminated. Re-take the reference every 60 min when idle; log drift.
- Run in `tmux`. Courtesy lockfile `/tmp/paia_bench.lock` (user, contact, expected end); removed on exit.
- Every phase's suite runs in **at least two separate sessions**; analysis flags medians differing by more than 3%.

## 4. Measurement methodology

- **Timing:** CUDA events only. Warmup >= 20 calls and >= 200 ms. >= 30 samples. Report median, p5, p95, mean, std, CV;
  `noisy: true` if CV > 3%. Keep raw samples.
- **Timing modes:** `isolated` (sync, start, 1 op, end, sync), `eager` (events around a Python loop of R ops),
  `graph` (R ops captured in a CUDA Graph, events around one replay). One loop or replay >= about 5 ms.
- **L2 rotation:** `n_copies = max(2, ceil(4 * L2_bytes / op_bytes))`, capped by `--mem-cap-gb` (12); if coverage < 4x L2,
  record it and set `l2_safe: false`. Hot variants (E1 only) reuse one buffer on purpose.
- **Bytes per call:** weights, metadata (scales, zeros), activations in, output out, each stored, plus total.
  Achieved GB/s = total / median. `pct_of_b0` = % of this platform's measured cold-read R_inf (never the spec).
- **Weights:** random data is fine for timing; real checkpoints only for E5.
- **Library dispatch:** record CUDA kernel names per E2/E4 shape with `torch.profiler`; record
  `torch.backends.cuda.preferred_blas_library()`.

## 5. Shapes (frozen in `paia-bench/configs/shapes.yaml`)

Weight matrices are `[N_out, K_in]`, verified against HF `config.json`.

- **Llama-3.1-8B** (hidden 4096, intermediate 14336, 32 heads, 8 KV heads, head_dim 128, vocab 128256): q/o 4096x4096;
  k/v 1024x4096; gate/up 14336x4096; down 4096x14336; qkv_fused 6144x4096; gate_up_fused 28672x4096; lm_head 128256x4096.
- **Qwen2.5-7B** (hidden 3584, intermediate 18944, 28 heads, 4 KV heads, head_dim 128, vocab 152064): q/o 3584x3584;
  k/v 512x3584; gate/up 18944x3584; down 3584x18944; qkv_fused 4608x3584; gate_up_fused 37888x3584;
  lm_head 152064x3584. QKV bias ignored for GEMV timing.
- **Synthetic sweep:** K = 4096 and K = 14336; N so BF16 weight bytes span 0.25 MiB to 1 GiB in about 14 geometric steps.
- **Batch sweep:** M in {1, 2, 4, 8, 16} on real shapes, graph mode only; M = 1 in all three modes.

## 6. Repository layout and schema

`paia-bench/` with `README.md`, `findings.md`, `predictions.md`, `configs/`, `bench/{env,monitor,timing,run,analyze}.py`,
`bench/ops/` (one module per impl), `scripts/`, `results/<platform_tag>/<session_id>/{env.json, cells/, logs/, ncu/,
nsys/, summary/}`. Cell JSON fields: see `bench/run.py:build_record` (schema_version now 2). `env.json`: GPU properties,
driver, CUDA, torch, triton, vLLM, cuBLAS, `nvidia-smi -q`, CPU, `pip freeze`, container, clock-lock attempt, profilers,
perf-counter permission.

## 7. Phases and gates

- **Phase 0:** environment and fingerprint. Gate 0: environment summary and open questions.
- **Phase 1:** harness, monitor, canary, then E1. Analysis: GB/s vs size (hot, cold), cold-read plateau R_inf_B0, fit
  `t = tau + B / R` on cold graph-mode reads (least squares + Theil-Sen), B_half, R^2, residuals, L2 falloff.
  Gate 1: plots, fit, canary stability, 5 repeated cells within 2%.
- **Phase 2:** E2. Per-shape table (median_us, GB/s, pct_of_B0); fits per K and mode; `tau_eager - tau_graph`;
  `tau_isolated` vs `tau_graph`; efficiency curve; predicted efficiency at BF16 and at 4-bit size.
  Gate 2: table and plots, flag non-GEMV dispatch.
- **Phase 3:** E3 ncu on one large and one small shape with `--clock-control none`; metrics: duration, DRAM bytes
  read/write, L2 bytes, DRAM/SM throughput %, warps active, grid size, registers. Compare DRAM bytes vs expected.
  Gate 3: bytes-match ratio or the exact blocker.
- **Phase 4:** E4. FP8: weights `float8_e4m3fn [N,K]` passed as `W.t()`, per-tensor float32 scales, activations quantized
  per tensor, `out_dtype=bfloat16`; pad M if M=1 is rejected; try row-wise. INT4 tinygemm g128 (adapt to the installed
  signature). Marlin via vLLM (time-box 1 hour). Theoretical speedup from stored byte counts (FP8 about 2.0x;
  INT4 g128 scale+zero 4.25 bits about 3.76x; scale-only 4.125 bits about 3.88x). Table: BF16 time, quantized time,
  measured speedup, theoretical, efficiency (measured / theoretical), BW_q / BW_16; the last two must agree.
  Gate 4: the table.
- **Phase 5:** E5. Qwen2.5-7B first, Llama-3.1-8B if accessible. One prompt, greedy, 128 new tokens, batch 1,
  decode tokens/s excluding prefill. vLLM eager vs graph (or HF eager + static cache / compile). nsys `--trace=cuda,nvtx`:
  per-token kernel launches, kernel time, idle between kernels, matmul share, top-10 kernels. Compare with
  bytes-per-token / R_inf_B0. Gate 5: per-token breakdown per config.
- **Phase 6:** report. `analyze.py` writes `summary/` (summary.md, cells.csv of clean cells, unclean list, cross-session
  comparison). Commit code and summaries, not large raw nsys/ncu files.

## 8. Do not

- Use spec bandwidth as the denominator (use measured B0).
- Loop a single small matrix (that measures L2, not DRAM).
- Use ncu durations as performance numbers.
- Average clean and contaminated cells, or silently drop failures.
- Change shapes, timing parameters or the schema mid-session without bumping `schema_version` and noting it.
- Spend more than about an hour fighting one library build.
