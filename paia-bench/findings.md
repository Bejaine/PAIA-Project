# Findings (living document)

Verified facts, surprises and failures. Every entry names its evidence. Platform: `dgxspark_gb10` unless stated.
Raw evidence for Phase 0: `results/dgxspark_gb10/phase0_20261016/{env.json,selfcheck.json}`.

## Phase 0: environment (2026-10-16)

### Machine

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GB10, compute capability 12.1, 48 SMs, L2 25,165,824 B (24 MiB), 128,518,619,136 B memory | `env.json` gpu |
| Driver | 580.159.03, CUDA driver API 13.0 | `env.json` nvml, cuda |
| CPU | aarch64, 20 cores: 10x Cortex-X925 (3.9 GHz max) + 10x Cortex-A725 (2.8 GHz max) | `env.json` cpu |
| Memory | unified LPDDR5x shared by CPU and GPU, 119 GiB visible to the OS | `free -g` |
| OS | Ubuntu 24.04.4 LTS, kernel 6.17.0-1021-nvidia | `env.json` uname |
| Disk | 1.5 TB free on `/` | `df -h` |

### Software stack decision

- **Image:** `paia-bench:25.11`, built from `docker/Dockerfile`. It is NGC `vllm:25.11-py3` plus nsys and ncu copied from NGC `pytorch:25.11-py3`, plus pandas, matplotlib and nvtx pinned to the PyTorch image's versions. All phases run in this one image.
- **Stack:** torch `2.10.0a0+b558c986e8.nv25.11`, CUDA 13.0 (cudart 13000), cuBLAS 13.1.0, Triton 3.5.0, vLLM `0.11.0+582e4e37.nv25.11`, transformers 4.57.1, ncu 2025.3.1, nsys 2025.5.1.
- **Same build in both NGC images:** NGC `pytorch:25.11-py3` and `vllm:25.11-py3` ship the identical torch build and cuBLAS (checked in both).
- **Why 25.11:** the host driver is 580 (CUDA 13.0). NGC 25.12 and later are built on CUDA 13.1 to 13.4 and need a newer driver. A forum report shows `vllm:26.04-py3` failing on a 580 driver with "compatibility mode is UNAVAILABLE". System-wide upgrades are not allowed on this shared machine, so 25.11 is the newest stack that runs natively.
- **sm_121 support:**
  - torch is compiled for `sm_80 ... sm_120, compute_120`, so there is no sm_121 SASS in torch itself.
  - cuBLAS ships native `nvjet_sm121_*` kernels, and Triton compiles natively for `arch=121`.
- **No clock locking.** `nvidia-smi -lgc/-lmc` needs root and changes clocks for every user. The SM clock is recorded per cell instead. NVML reports max SM clock 3003 MHz and application clock 2418 MHz.
- **For P3:** use the same image (the Dockerfile builds on amd64 too; both NGC tags are multi-arch). That requires driver 580 or newer on the RTX machine.

### Container settings that matter for correctness

- **`--pid=host` is required.** Without it, psutil cannot see other users' processes and NVML's PIDs (host PIDs) cannot be resolved. Verified: from inside the container, NVML PID 3934636 resolves to user `viswas` and psutil sees all 627 host processes.
- **Mount `/etc/passwd` read-only.** Without it, `torch.profiler` raises `KeyError: getpwuid(): uid not found` when running as a non-root uid.
- **Shared `/tmp`**, so the courtesy lockfile is visible to everyone.
- All three settings are in `scripts/drun.sh`.

### NVML fields on GB10

- **Work:** utilization, compute/graphics process lists, per-process utilization, SM and graphics clock, max SM clock, application clock, temperature, power usage, clock event (throttle) reasons, pstate, persistence and compute mode.
- **Not supported:** memory clock, max memory clock, power limit, memory info. The monitor records `null` for these.
- **Consequence:** no NVML signal tracks the memory bus. The bandwidth canary is the only guard against CPU-side bandwidth theft.

### Performance counters (ncu)

- **Non-root access works.** `RmProfilingAdminOnly: 0`, and an ncu probe as uid 1005 succeeded.
- **`dram__*` metrics do not exist on GB10.** `dram__bytes_read`, `dram__bytes_write`, `dram__throughput` and `dram__bytes` are all absent from `ncu --query-metrics`.
- **Replacement:** memory traffic goes through the L2 sysmem aperture, so use
  - `lts__d_sectors_fill_sysmem.sum` x 32 B for bytes read from memory;
  - `lts__t_sectors_aperture_sysmem_op_write.sum` x 32 B for writes (to be confirmed in E3);
  - `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` in place of `dram__throughput`.
- **Byte check on a cuBLAS GEMV, 14336x4096 BF16, M=1:**
  - `lts__d_sectors_fill_sysmem` = 3,679,531 sectors = 117,744,992 B;
  - expected = 117,440,512 (W) + 8,192 (x) + 28,672 (y) = 117,477,376 B;
  - **ratio 1.002**;
  - `lts__d_sectors_fill_device` = 0.
- **`launch__grid_size`, `launch__block_size` and `launch__registers_per_thread` work** even though `--query-metrics` does not list them. That GEMV ran with grid 3584, block 128, 164 registers per thread.
- The metric map lives in `configs/platform_notes.yaml`.

### Library paths (`selfcheck.json`, functional only, no timing)

All results below are on the stack above.

**Plain BF16**
- **BF16 matmul vs CPU:** correct (relative error 3.4e-3).
- **E1 torch ops:** `sum`, `copy_` and `fill_` are all graph-capturable.
  - `sum` launches **two kernels** (a Memset of the output, then `reduce_kernel`). Its fixed cost therefore includes an extra kernel. That is one reason to cross-check E1 with the single-kernel Triton probe.
  - `copy_` appears as `Memcpy DtoD`.
- **Triton read probe:** compiles for `GPUTarget(cuda, arch=121)`, is correct and is graph-capturable.
- **cuBLAS BF16 `F.linear`:**
  - at M=1, all 6 tested shapes dispatch to `internal::gemvx::kernel`;
  - at M=16 they dispatch to `cutlass_80_wmma_tensorop_bf16_s161616gemm` (an sm80 kernel), and 512x3584 at M=16 adds `splitKreduce_kernel`;
  - all are correct and graph-capturable;
  - `preferred_blas_library` = Cublas.

**FP8**
- **FP8 `torch._scaled_mm`, per-tensor scales:** correct at M = 1, 2, 4, 8, 16, 32, and graph-capturable.
  - Kernels: `nvjet_sm121_qqtst_mma_64x64x128...` (M <= 2, M = 32) and `nvjet_sm121_qqtst_mma_192x16x128...` (M = 4 to 16).
  - **M = 1 is accepted, so no padding is needed.**
- **FP8 `torch._scaled_mm`, row-wise scales: BROKEN on sm_121, silently.**
  - It dispatches `cutlass::...enable_3x_kernel_for_sm10_or_later...`, which prints about 1e5 lines of "Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting." from the device.
  - It returns garbage (relative error 1.2e34 at M=1, NaN at M=16) and **raises no Python exception**.
- **FP8 row-wise via vLLM `cutlass_scaled_mm`:** correct at M = 1 and 16, and graph-capturable. `cutlass_scaled_mm_supports_fp8(121)` returns True. This is the row-wise FP8 route on GB10.

**INT4**
- **INT4 tinygemm:**
  - schemas are `_convert_weight_to_int4pack(Tensor self, int innerKTiles)` and `_weight_int4pack_mm(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros)`;
  - the packed input is **uint8 `[N, K/2]`** (two nibbles per byte), packed output `[N/8, K/128, 32, 4]` int32; scales and zeros are `[K/G, N, 2]` bf16;
  - dequant convention `w = (q - 8) * scale + zero`;
  - kernel `tinygemm_m16n8k16_chunk_kernel`; correct and graph-capturable at M = 1 and 16.
- **INT4 Marlin (vLLM `gptq_marlin_gemm` through `apply_gptq_marlin_linear`, uint4b8, g128, scale-only):**
  - `check_marlin_supported(cc=121)` returns True;
  - correct and graph-capturable on 4 shapes at M = 1 and 16;
  - kernel `marlin::Marlin<__nv_bfloat16, ...>`.

**Rule adopted because of the row-wise FP8 failure:** every cell checks its op against a reference once before timing. A cell whose output is wrong is marked `failed` and never timed.

### E5 feasibility (smoke test only, numbers not meaningful)

- **vLLM loads Llama-3.1-8B-Instruct in BF16:** 14.99 GiB of weights, `gpu_memory_utilization=0.25` leaves 12.5 GiB for the KV cache.
- **Both modes decode, with identical greedy output:** `enforce_eager=True`, and `enforce_eager=False` (PIECEWISE plus FULL decode CUDA graphs).
- **Eager trace:** cuBLAS `gemvx` dominates kernel time, so the decode GEMVs use the same kernel E2 measures.
- **Graph mode:** uses torch.compile (Inductor Triton kernels for norms and activations) as well as CUDA graphs.
- **nsys needs `--cuda-graph-trace=node`.** Without it, graph-mode kernels collapse into `cudaGraphLaunch` entries.
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps the engine in-process so a single nsys trace sees it.
- Wall time was about 5.1 s for 32 tokens, under nsys and with the GPU busy with another user's job. **Not a result.**

### Model configs

- Both HF `config.json` files match the brief exactly: hidden, intermediate, heads, KV heads, vocab; head_dim = 128; `tie_word_embeddings: false`; bf16.
- The HF token has access to gated Llama.
- Llama-3.1-8B-Instruct is fully cached locally (15 GB). Qwen2.5-7B-Instruct still needs downloading (about 15 GB).
- Qwen2.5 has a QKV bias. It is ignored for GEMV timing, since the bias add is a separate elementwise op in most paths and is not a weight read.

### Shape set decision

- The brief asks for a synthetic sweep from 0.25 MiB with N a multiple of 128. At K = 4096 the smallest N = 128 already gives 1 MiB, and at K = 14336 it gives 3.5 MiB. That would lose the small-B region where tau dominates (B_half is likely around 1 MiB).
- **Decision:** N is a multiple of 128 when the target is at least 128 rows, and a multiple of 32 (minimum 32) below that.
  - K = 4096: 14 points, 0.25 MiB to 1 GiB.
  - K = 14336: 12 points after dedup, 0.875 MiB to 1 GiB.
- Frozen in `configs/shapes.yaml` (shapes_version 1).

### Shared-machine observations

- **GPU busy:** at Phase 0 time, user `viswas` has run `predict_pivots_combined_attention_rw.py` (PID 3934636) on the GPU for more than 8 hours. It holds about 9.4 GiB at 59 to 96% utilization and 73 to 76 C, and uses one full CPU core. During that job the idle gate correctly blocks every measurement. **No timing will be taken while any foreign GPU or bus activity is present.**
- **Possible foreign GPU users:** an `ollama` service user exists, and so does `dgx-dashboard-service-user`. The idle gate checks NVML compute processes for any foreign PID.
- **Login load:** 8 interactive sessions (users imc, vedant-tejas, viswas). Load average was about 1.0 to 1.5 at check time.

## Phase 1: harness (2026-10-16)

### Design decisions that affect accuracy

- **Monitor process.** The monitor is a separate process, not a thread, pinned to the slowest CPU cores (the A725 cluster on GB10). The benchmark process is pinned to the fastest (X925). Reason: a polling thread would contend for the benchmark's GIL and inflate the host launch cost that eager mode measures. Fast and slow cores are detected from `cpuinfo_max_freq`. On GB10 they are interleaved by ID (fast: 5 to 9 and 15 to 19).
- **Correctness before timing.** Every cell checks its op against a reference before timing. A wrong output means `failed`, and the cell is never timed. This is the lesson from the silent row-wise FP8 failure.
- **Fixed shuffled order.** Cells run in a fixed shuffled order (seed in `configs/shapes.yaml`, stable across resumes), so slow drift such as thermals or other users cannot line up with size or op.
- **Random buffer contents.** Probe buffers hold random data, not constants.
- **R sizing.** R is re-sized from a measured unit until one eager loop or graph replay lasts at least 5 ms. The isolated single-op estimate includes launch and sync overhead, which made graph-mode R about 6x too small on a 1 MiB read.
- **Cold-cell safety.** It is judged by reuse distance: bytes touched between two uses of the same copy must be at least 4x L2. When the memory cap forces fewer copies than `max(2, ...)`, a cell can use 1 copy and still be `l2_safe` (for example a 4 GiB copy). `n_copies_wanted` records the formula's value.
- **Sample count.** It is `clamp(1000 ms / unit time, 30, 300)`. These parameters were added to `configs/shapes.yaml` before any measurement session.
- **Clock-event reasons.** HW slowdown, SW/HW thermal and power brake contaminate a cell. `sw_power_cap` is recorded but does not contaminate on its own.
- **Foreign CPU** is measured with psutil every 500 ms in the monitor process. Any window above 2 cores contaminates. The gate uses the same threshold over its 2 s window.
- **Triton probe tuning.** The probe config (BLOCK, warps, waves) is tuned once per session on a cold 1 GiB buffer, under the idle gate and monitor, then persisted, so every cell and every resume uses the same probe.
- **tmux required.** Real sessions refuse to start outside tmux.
- **Debug sessions.** `debug_*` sessions skip waiting, record gate failures as contamination, and are git-ignored.

### Verification

- **`tests/test_resume.py`, 5 of 5 runs passed.**
  - The test SIGKILLs the runner in the middle of a cell and restarts it.
  - No clean cell was redone (same `t_start`, one execution each), and the interrupted cell was redone once.
  - A cell with 2 injected contaminations ends `clean` with `retries=2`. A permanently contaminated one ends `unclean_exhausted` after `max_retries + 1` attempts.
  - A third run executes 0 cells.
- **Debug smoke run** (`debug_harness_20261016`, 48 E1 cells, GPU busy with another user's job):
  - every cell ran and every one was marked contaminated, naming the foreign process;
  - all schema fields are present; the lockfile was removed on exit;
  - **not results.**
- **Qualitative only:** with the other job running, cold reads saw about 100 to 110 GB/s and the canary reference CV was 0.30, against the required 0.02. Measuring on a busy GB10 would be badly biased, which is why the gate waits.

## E1 (B0) results, GB10 (sessions s1_20261016 and s2_20261018)

Two full sessions, 360 cells each, all clean, two days apart. Code: s1 at `a8d24b0`, s2 at `247bdc4` (adds the gate settle step only). Summaries are in `results/dgxspark_gb10/<session>/summary/`.

### Headline numbers

| Quantity | s1 | s2 | Prediction |
|---|---|---|---|
| **R_inf_B0** (cold graph read, >= 10x L2) | **241.74 GB/s** | **242.10 GB/s** | 225 (200 to 245) |
| as % of the 273 GB/s spec | 88.6% | 88.7% | 82% |
| torch `sum` vs Triton read plateau | 241.7 / 240.6 | 242.1 / 240.9 | |
| copy plateau (graph mode, 2B counted) | 238.7 GB/s (0.99x read) | 239.9 GB/s | 0.85x |
| write (fill) plateau | 196.8 GB/s (0.81x read) | 196.8 GB/s | 0.90x |
| **probe tau**, Triton read, graph mode | **1.85 us** [1.14, 1.97] | **1.86 us** [1.53, 1.95] | 3.5 us |
| probe tau, eager mode | 4.74 us | 4.70 us | |
| probe tau, isolated mode | 11.48 us | 11.37 us | |
| B_half, Triton graph mode | 0.433 MiB | 0.431 MiB | 0.75 MiB |
| hot L2 peak (Triton read, graph mode) | 1505 GB/s at 16 MiB | 1525 GB/s at 16 MiB | about 1 TB/s at 4 to 16 MiB |
| L2 falloff (hot falls to cold) | drop between 16 and 32 MiB; equal to cold by 64 MiB | same | 16 to 24 MiB |
| canary reference | 240.2 GB/s, CV 0.22 to 0.83%, drift <= 0.16%/h | 237.9 GB/s, CV 0.76% | |

### Cross-session agreement (Gate 1)

- **Headline cells, six chosen: all within 0.7%, four within 0.2%.** Examples: 1 GiB cold Triton read -0.13%, 256 MiB cold `sum` +0.11%, 64 MiB cold fill +0.01%.
- **All 360 matched cells:** median |diff| 0.27%; 91% within 2%, 94% within 3%.
  - Cold cells of 2 MiB and up: 96% within 2%, max 5.1%.
  - Cells of 64 MiB and up: median 0.13%.
- **The 21 cells above 3% fall into two groups:**
  - hot cells at the L2 edge (buffers 8 to 32 MiB, up to 30%), where partial L2 residency depends on the replacement state;
  - cold ops under 1 MiB taking 1.5 to 10 us, where 0.5 us is already 25%.
- **Not timing errors.** Both groups are properties of those cells. Tables and fits should treat them as such, not as results of record.

### Things the data showed that the brief did not anticipate

- **torch `sum` is not a valid single-tau probe.** It launches a memset plus a reduction whose launch configuration changes with input size, so its fixed cost is not constant. `t = tau + B/R` fits it with 11 to 15% RMS residuals, including +47% at 0.25 MiB.
  - The single-kernel Triton read (fixed grid: BLOCK 4096, 8 warps, 2 waves x 48 SMs, re-tuned identically in both sessions) fits with 1.8 to 2.9% residuals in graph mode.
  - **The Triton probe defines the B0 tau.** `analyze.py` marks fits above 5% residual as `misfit`.
- **Bandwidth depends weakly on footprint.**
  - Fits over 64 MiB and below give R of about 252 GB/s; the GiB-scale plateau is 241 GB/s, about 4% lower.
  - Rotation is not the cause: ncu shows DRAM fills = 1.00 to 1.06x bytes for 1, 4 and 16 MiB cells at rotation factors 1, 4, 8 and 16 (with `--cache-control none`; ncu's default flushes caches and must not be used for this check).
  - Likely TLB reach at multi-GiB footprints (open question).
- **CV overstates noise.** About half of all clean cells have CV above 3%, but the tail-insensitive robust CV (IQR/1.349/median) has median 0.40%, with only 17 to 18 cells above 3%.
  - Large cells show p5 = median and short bursts of slow samples (tens of ms), meaning brief episodes of bus contention.
- **Our monitor does not bias medians.** A controlled test (64 MiB fill and read, 600 isolated samples, with no monitor, NVML polling, psutil scanning, or both) gave medians equal within 0.2% in all cases. NVML polling roughly doubles the share of slow tail samples on fill. Evidence: `scripts/diag/monitor_perturbation.py`, `s1_20261016/diag_monitor_perturbation.json`.
- **Writes are the slow direction on LPDDR5x:** fill reaches 0.81x read. Copy reaches 0.93 to 0.99x read once both streams are counted.
- **Graph-mode tau is tiny for fill and copy** (0.2 to 0.7 us and 1.3 us): back-to-back kernels overlap heads and tails inside a graph.
- **Gate behavior:**
  - In s1, NVML's trailing utilization average made the gate wait 30 s after almost every one of our own cells (321 times), roughly doubling run time without affecting validity.
  - s2 adds a settle step (wait up to 10 s for our own utilization to decay, then the unchanged strict 2 s check). s2 took 27 minutes against about 3.2 hours.
  - The gate also catches `gnome-shell` (user `gdm`) GPU activity from the desktop.
- **SM clock** sat at 2177 to 2190 MHz in every cell, below the 3003 MHz max; no clock-event reasons; max temperature 49 C.

### Cross-platform caution

P3's first RTX PRO 5000 Blackwell results (`paia-results/rtxpro5000bw/`, 80 of 360 E1 cells) were measured in a host venv: torch 2.14.1+cu130, cuBLAS 13.1.1, Triton 3.8.0, no vLLM, driver 595.84. Our GB10 stack is torch 2.10.0a0 (NGC 25.11), cuBLAS 13.1.0, Triton 3.5.0. Platform comparisons are confounded by software until both platforms use the same stack. Driver 595 can run the `paia-bench:25.11` image natively.

## Open questions

1. Is `lts__t_sectors_aperture_sysmem_op_write` the right write-bytes counter? To be confirmed in E3 with a `fill_` kernel of known size.
2. Does an idle, unthrottled GB10 hold a stable SM clock across a session? Cell telemetry will show it.
3. Canary sensitivity: how much does a CPU-only memory-bound job by another user reduce GPU streaming bandwidth? This could be quantified with our own CPU load, if P2 approves.
5. Why is streaming bandwidth about 4% higher at <= 100 MiB footprints (about 252 GB/s) than at GiB scale (241 GB/s)? TLB reach is the leading hypothesis; testable with ncu TLB or page-walk counters.
6. Is the slow-sample tail (bursts of tens of ms) caused by other users' sub-threshold CPU memory traffic, LPDDR housekeeping, or NVML? Medians are unaffected, so this is low priority.
4. Does an RTX 5000-class machine have driver 580 or newer, so P3 can use the same image?
