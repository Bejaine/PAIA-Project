# PAIA Progress Log

Chronological record of what was done, what was decided and why. Newest entries at the bottom. Add an entry for every
work session. Numbers of record are in `docs/RESULTS.md`; the forward plan is in `docs/PLAN.md`; the original P2 brief is
`docs/P2_BRIEF.md`.

Team roles: **P1** custom GEMV kernels; **P2** measurement, profiling and baselines on GB10 (this log is mostly P2's);
**P3** RTX PRO 5000 Blackwell runs (manit2004).

---

## 2026-10-16: Phase 0, environment (PR #1)

- **Machine:** DGX Spark, NVIDIA GB10 (cc 12.1, 48 SMs, 24 MiB L2, 128 GB unified LPDDR5x), driver 580.159 (CUDA 13.0),
  aarch64, Ubuntu 24.04, shared by about 8 users.
- **Stack decision:** NGC containers, chosen for accuracy over convenience. NGC 25.11 is the newest release that runs
  natively on driver 580 (25.12+ needs a newer driver, and system-wide upgrades are not allowed). Built one image,
  `paia-bench:25.11` = `nvcr.io/nvidia/vllm:25.11-py3` + nsys/ncu from `pytorch:25.11-py3` + pandas/matplotlib/nvtx.
  torch 2.10.0a0+b558c98.nv25.11, cuBLAS 13.1.0, Triton 3.5.0, vLLM 0.11.0.
- **Brief claims found wrong on GB10:** no `dram__*` ncu metrics (use `lts__d_sectors_fill_sysmem` x 32, bytes matched at
  1.002); row-wise FP8 `torch._scaled_mm` silently returns garbage on sm_121 (use vLLM `cutlass_scaled_mm`); NVML memory
  clock / memory info / power limit unsupported. Per-tensor FP8 works at M = 1 (no padding).
- Container needs `--pid=host` (monitor) and a mounted `/etc/passwd` (torch.profiler). All in `scripts/drun.sh`.
- Shapes frozen in `configs/shapes.yaml`; synthetic sweep deviates from the brief (N multiples of 32 below 128 rows so
  0.25 to 1 MiB is covered).
- Repo: `git@github.com:Bejaine/PAIA-Project.git`, initial commit on main, everything else via PRs.

## 2026-10-16: Phase 1, harness (PR #2)

- `bench/monitor.py` (idle gate, monitor in a separate process pinned to slow cores, canary, lockfile), `bench/timing.py`
  (isolated / eager / graph, L2 rotation by reuse distance, R sized to >= 5 ms), `bench/run.py` (resumable atomic cells,
  correctness check before timing, fixed shuffled order, tmux required, `debug_*` sessions), `bench/ops/bw_probe.py`.
- Kill-and-resume test passes (SIGKILL mid-cell, no clean cell redone).
- Debug run on a busy GPU showed why gating matters: canary CV 0.30 under another user's job.

## 2026-10-16: predictions committed before any valid measurement (PR #3)

`paia-bench/predictions.md`, written from theory at P2's request.

## 2026-10-16 to 10-18: E1 on GB10, schema v1 (PRs #4, #5)

- `bench/analyze.py`: plateaus, Hockney fits (1/t-weighted LS + Theil-Sen + bootstrap CI), hot/L2 falloff, canary,
  session comparison, plots; synthetic test passes.
- s1 waited hours for another user's job, then ran 360/360 clean. Gate waited 30 s after every cell for NVML's trailing
  utilization of our own cell; fixed with a settle step (s2 took 27 min vs 3.2 h).
- B0 241.7 / 242.1 GB/s, probe tau 1.85 us. Found: torch `sum` misfits a single-tau model (Triton probe defines tau);
  monitor does not bias medians; rotation truly cold (ncu, with `--cache-control none`).

## 2026-10-18 to 10-19: E2 on GB10, schema v1 (PR #6)

- `bench/ops/cublas_bf16.py`, `bench/analyze_e2.py` (per-shape table, %B0 and same-size E1 comparison, per-kernel fits,
  launch split, efficiency, M sweep).
- **Main finding:** cuBLAS `gemvx` v6 (11 shapes) streams at about 70% of B0; v7 at 96 to 98%. M = 2 is faster than M = 1.
- **Methodology problem:** sessions disagreed by 2.1% median (max 15%). Cause: variance between allocations (6% span over
  10 fresh allocations of an 8 MiB GEMV). Lesson also recorded: running anything of ours concurrently with a
  measurement (even analysis) contaminates it; the gate caught it (canary and foreign-process checks).

## 2026-10-19 to 10-21: schema v2 and reruns (PR #7)

- **Schema v2:** each cell measured over 5 independent allocations; value = median across allocations; spread recorded.
  The runner refuses to mix schema versions in a session.
- Reruns (all clean, 1,156 cells): E1 B0 239.0 / 239.5 GB/s, tau 1.70 / 1.66 us, B_half 0.39 MiB; E2 cross-session
  agreement 0.27% median (from 2.08%). v2 is the result of record.
- P3 merged RTX PRO 5000 environment + preliminary E1 (80 cells) and a CPU-pinning fix for hybrid x86 CPUs.

## 2026-10-21: focused E3 (pushed to PR #7)

`scripts/ncu_gemv_variants.sh`: v6 vs v7 move identical bytes at identical register-limited occupancy (164 regs, 25%),
but v6's 16-lane rows keep half the bytes in flight per warp, so it stalls on memory about 1.7x longer.

## 2026-10-21 to 10-22: Evaluation 1 deck (in progress)

- The course template `PAA-Evaluation-1-Presentation-Template.pptx` (repo root, untracked) has malformed XML
  (ElementTree-mangled namespaces) that LibreOffice cannot open; a repaired copy is used for rendering only.
- Draft deck built with pptxgenjs in the template's style (9 slides: 8 prescribed + 1 custom "model of the gap"),
  template Roboto fonts embedded, speaker notes on every slide: `PAIA-Evaluation-1-DRAFT.pptx` (repo root, untracked).
  Source and tooling in `~/.paia_deck_work/` (see `handoff.md` for how to rebuild).
- **Waiting on P2:** names and roll numbers, the `github.com/paa-iiit/...` repo link, P1's confirmation of the slide-5
  approach, timeline / GPU-slot confirmation, whether to commit the deck.
- Logs written: `docs/PROGRESS.md`, `docs/RESULTS.md`, `docs/PLAN.md`, `docs/P2_BRIEF.md`.

## 2026-10-23: direction change; E4 harness (library FP8 / INT4)

- **P2's direction:** set the Evaluation 1 deck, the course repo and P3's RTX work aside for now. Focus: baselines and
  profiling on the DGX Spark, the gap and its root cause, then kernels that close it. P2 may now modify P1's kernels
  (`src/paia_gemv/`) for GB10 or write new ones following P1's design (this lifts the brief's "no project kernels" rule).
- P1's work is on `main` (PRs #9, #10): S1 BF16 (phase `s1` wired in), FP8 W8A16 and INT4 W4A16 kernels with their `Op`
  adapters; nothing has run on the Spark yet. `src/patches/integration_quant.patch` (phases `s3`, `s4`) no longer applies
  to `run.py` and will be merged by hand.
- **E4 code:** `bench/ops/lib_quant.py` (torch `_scaled_mm` per-tensor FP8, vLLM CUTLASS row-wise FP8, both also with
  vLLM dynamic activation quantization, tinygemm INT4, Marlin INT4), phase `e4` (1,146 cells), `bench/analyze_e4.py`
  (per-shape speedups and %B0, identity check, per-kernel asymptotic rates and per-op / per-byte split, activation-quant
  cost, per-token GEMV cost, M sweep), `analyze.py --bf16-sessions`.
- All 172 correctness checks pass on GB10; dispatch recorded (FP8 per-tensor uses several `nvjet` tiles and an
  `sm89_xmma` GEMM). A gate-bypassed debug run (`debug_e4a`, 220 graph cells) exercised the analysis; hints only (see
  `findings.md`): quantized large ops run near B0, the loss concentrates on small ops (Qwen k/v).
- Next: two gated E4 sessions (about 2 h each), then ncu on the library kernels.
