# P3 runbook: first days on the RTX 5000 server

Companion to `P3_RTX5000_benchmark_guide.md`. That guide explains the why; this file is the exact order of
commands for this repo as it stands (commit `044ba02` plus the P3 prep changes listed at the end).

## What the repo can run today

| Phase | Status in `paia-bench/` | Command |
|---|---|---|
| E0 fingerprint | ready | `bench.env` |
| selfcheck (library paths, no timing) | ready | `bench.selfcheck` |
| E1 (B0) bandwidth | ready | `bench.run --phase e1` |
| E2, E3, E4, E5, analyze | **not written yet** (P2) | wait for the commit P2 gives you |

So the first access window is for: preflight, environment, E0, selfcheck, predictions, E1 session s1.

## Day 0 (now, before access)

1. Request Hugging Face access to `meta-llama/Llama-3.1-8B-Instruct` and create a read token.
2. Ask P2 for the commit to use, and send them the "PR for P2" note at the bottom of this file.
3. Send the admin message below as soon as you know who runs the server.
4. Fill in `paia-bench/predictions.md` for your platform after preflight tells you the variant (the arithmetic
   table below does most of the work). Commit it **before** E1 runs.

## Day 1: first SSH login

```bash
git clone https://github.com/Bejaine/PAIA-Project.git && cd PAIA-Project/paia-bench
git checkout <commit from P2>
scripts/preflight.sh            # read-only, ~1 min; writes results/preflight/<host>_<time>.txt
```

The summary at the end of the preflight report tells you:

- **variant and platform tag**: `rtx5000ada`, `rtxpro5000bw` or `quadrortx5000`;
- **which path to take** (next section);
- whether perf counters are open (E3), whether tmux and enough disk exist, and who else is on the GPU.

Send the report file to P2.

If the host has several GPUs, choose yours (agree with the admin) and export it for every command:

```bash
export PAIA_GPU=<nvidia-smi index or UUID>      # both drun.sh and hrun.sh honor this
```

## Choose the software path

| Preflight says | Path |
|---|---|
| driver >= 580, Docker usable without sudo, cc 8.9 or 12.0 | **A: Docker**, same image as P2 (preferred: identical stack) |
| driver < 580, or no Docker access, or cc 7.5 (Turing) | **B: venv on the host** |

**Path A (Docker)**

```bash
docker build -t paia-bench:25.11 -f docker/Dockerfile docker    # pulls ~20-30 GB from nvcr.io
R=scripts/drun.sh
```

**Path B (venv)**

```bash
scripts/setup_venv.sh            # add --vllm to try vLLM; time-box vLLM problems to 1 hour
R=scripts/hrun.sh
```

`setup_venv.sh` picks the torch CUDA flavor from the driver, freezes the result to
`.venv-host/requirements.lock.txt`, and refuses a torch build without code for your GPU. Use the same venv for
both sessions. ncu and nsys must exist on the host (preflight lists them); if they are missing, ask the admin,
or install the Nsight `.run` installers into `$HOME` without root.

## Environment check and selfcheck (no timing)

```bash
TAG=rtx5000ada          # from preflight
$R python -m bench.env       --session env_$(date +%Y%m%d) --platform-tag $TAG --probe-ncu
$R python -m bench.selfcheck --session env_$(date +%Y%m%d) --platform-tag $TAG
```

Check in `results/$TAG/env_*/`:

- `env.json` `nvml.device_uuid` is **your** GPU (the UUID from `nvidia-smi -L`), `device_count_visible_to_nvml` is as expected;
- `nvml.ecc_mode_current_pending` recorded (do not let ECC change between sessions);
- `perf_counters.ncu_probe_ok` is true, or note why not (E3 blocker);
- `selfcheck.json`: anything with `correct: false` must never be benchmarked on this platform. Copy surprises into `findings.md` under a new `## Platform: <tag>` heading, the same way P2 did for GB10.

Expected selfcheck failures by variant (these are findings, not bugs):

- **Turing 7.5**: BF16 paths slow or failing, no FP8 (`_scaled_mm`), no tinygemm INT4, likely no Marlin.
- **Ada 8.9**: FP8 per-tensor should work. Watch row-wise FP8 for the same silent-garbage failure GB10 had.
- **Blackwell 12.0**: needs CUDA 12.8+ torch; check that `_scaled_mm` row-wise does not dispatch an sm_100-only kernel.

## E1, session s1

```bash
tmux new -s paia                 # real sessions refuse to run outside tmux
S=s1_$(date +%Y%m%d)
$R python -m bench.run --phase e1 --session $S --platform-tag $TAG --contact "<your name/email>"
# detach: Ctrl-b d. Lost connection or Ctrl-C: rerun the same command, it resumes.
$R python -m bench.run --phase e1 --session $S --platform-tag $TAG --list     # status of every cell
```

E1 is 15 sizes (256 KiB to 4 GiB) x 4 probes x hot/cold x 3 modes = 360 cells, roughly 1 to 2 hours on an idle machine.
If the log keeps saying `waiting ...: foreign_cpu ...` or `foreign_gpu_process`, the machine is not idle; that
is the gate working. Do not loosen thresholds: tell P2 and pick a quieter window (preflight shows who is busy).

Gate before E2: cold-read plateau visible, canary reference CV < 2% (`canary_ref.json`), repeated cells agree
within 2%. Until `bench.analyze` exists, eyeball `cells/*.json` (`median_us`, `achieved_gbps`, `status`).

## Message to the admin (send before access)

> Hi, I'm running GPU memory-bandwidth benchmarks for a course project (PAIA) on the RTX 5000 server. A few
> questions before I start:
> 1. Which GPU index should I use, and can I have it exclusively during agreed windows (I'd suggest nights or
>    weekends)? My runs pause themselves automatically whenever anyone else uses the GPU.
> 2. Am I allowed to lock clocks with `nvidia-smi -lgc/-lmc`? If not, that's fine; I log clocks instead.
> 3. Are GPU performance counters available to non-root users (`RmProfilingAdminOnly=0`)? Nsight Compute needs it.
> 4. Is ECC on or off? Please don't change it for the next few weeks, because it affects bandwidth.
> 5. Can I use Docker with the NVIDIA runtime (NGC images), and is there about 60 GB of disk I can use for an
>    image and two model checkpoints?
> 6. Are `tmux`, Nsight Systems and Nsight Compute installed?
>
> While a run is active there is a note in `/tmp/paia_bench.lock` with my contact details and the expected end time.

## Prediction arithmetic (fill your own guesses into `predictions.md`)

Bytes per decode token, BF16, all linear layers plus lm_head, embedding excluded (from `configs/shapes.yaml`):
Llama-3.1-8B **15.01 GB**, Qwen2.5-7B **14.14 GB**.

| Variant | Spec GB/s | Floor tok/s Llama / Qwen (at 90% of spec) | B_half at tau 3 / 5 us (MiB) | k_proj Llama eff. BF16 / INT4, tau 3 to 5 us |
|---|---|---|---|---|
| RTX 5000 Ada | ~576 | 34.5 / 36.7 | 1.5 / 2.5 | 0.84-0.76 / 0.59-0.46 |
| RTX PRO 5000 Blackwell | ~1344 | 80.6 / 85.5 | 3.5 / 5.8 | 0.70-0.58 / 0.38-0.27 |
| Quadro RTX 5000 | ~448 | 26.9 / 28.5 | 1.2 / 1.9 | 0.87-0.81 / 0.65-0.53 |
| GB10 (P2, reference) | 273 | | 0.7 / 1.2 | 0.92-0.87 / 0.75-0.64 |

Model: `eff = B / (B + B_half)`, `B_half = tau * R_inf`, `R_inf` taken as 90% of spec. Llama k_proj is 8 MiB at
BF16 and 2.1 MiB at INT4 (3.76x); gate_proj is 112 MiB, so it stays above 95% efficiency on every variant.
These are inputs for your guesses, not the guesses: write down your own tau and R_inf and your reasoning.

## Prep changes made in this repo (to go to P2 as a PR)

- **NVML picked the wrong GPU on multi-GPU hosts.** The monitor, idle gate and `env.json` used NVML device 0,
  but NVML ignores `CUDA_VISIBLE_DEVICES`. Now the benchmark process passes its CUDA device UUID and NVML is
  opened by UUID (`bench/common.py: gpu_uuid, nvml_handle`). With several GPUs and no UUID it raises instead of
  guessing. GB10 behavior is unchanged (one device). Tested on a GTX 1650: lookup by UUID, single-device
  fallback, bogus UUID rejected, monitor and gate run; `tests/test_resume.py` still passes.
- `env.json` now also records NVML device index, name, UUID, PCI bus ID, ECC mode, memory bus width and MIG mode.
- The ncu permission probe in `bench/env.py` requested `lts__d_sectors_fill_sysmem.sum` (a GB10 metric). If
  that metric is absent elsewhere, the probe would report "no counter access" falsely. It now asks only for
  `gpu__time_duration.sum`.
- `scripts/drun.sh` honors `PAIA_GPU` (`--gpus device=...`).
- New: `scripts/preflight.sh`, `scripts/setup_venv.sh`, `scripts/hrun.sh` (host venv fallback with the same
  provenance variables as `drun.sh`; `.venv-host/` is git-ignored).

Things to raise with P2 (not changed, since they are methodology):

- On a big shared x86 server, "other users < 2 CPU cores" may rarely be true. Preflight shows the usual load;
  if it is always above 2 cores, agree a platform-specific threshold with P2 before s1, and record it.
- On AMD CPUs with preferred cores, `cpuinfo_max_freq` can differ per core, so the "fast" set may be just 1 or 2
  cores. Check `cpu_affinity` in the first cell JSON.
- Turing needs an FP16 baseline for E2 and has no FP8 or INT4 library paths; E2/E4 code must branch on cc.
