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

## E2 (B1) cuBLAS BF16 results, GB10 (sessions e2s1_20261019 and e2s2_20261019, schema v1)

218 cells per session, all clean in the end (6 and 11 attempts retried, all for `canary_after` just under its threshold). B0 = 242.10 GB/s from E1 s2. Summaries: `results/dgxspark_gb10/e2s*_20261019/summary/`.

### cuBLAS's M=1 dispatch is the dominant BF16 loss on GB10, and it is per byte, not per op

cuBLAS dispatches `internal::gemvx::kernel` for every real shape at M=1, in two configurations (template parameter 6 vs 7):

| Kernel | Shapes | %B0 (graph mode) | Per-kernel fit (graph mode) |
|---|---|---|---|
| `gemvx/v6` | Llama q/k/v/o/qkv; Qwen q/o/qkv/gate/up/gate_up; synthetic K=4096 with N 768 to 36480 | **65 to 73%** | tau 1.95 us, **R_inf 169.9 GB/s**, 2.6% residual |
| `gemvx/v7` | Llama gate/up/gate_up; both down_proj; Qwen k/v; synthetic K=14336 | **95 to 104%** | R about 235 GB/s for N <= 37504 |
| `gemvx/v7`, very large N | lm_head (both), synthetic N >= 69120 | 72 to 75% | second regime, so v7 fits as a misfit overall |
| `cutlass_sm80` + `splitKreduce` | synthetic K=14336, small N | | tau 5.27 us, R 245.8 GB/s, 2.1% residual |

- **v6 has the ideal probe's tau (about 2 us) but streams 30% slower.** The loss is per byte (R_inf), not per op (tau).
- **Consequence for the project:** the BF16 baseline is not at the ceiling on most attention shapes and on Qwen's MLP. Quantized-kernel speedups against cuBLAS BF16 will look larger on those shapes than against an ideal BF16 GEMV. A custom BF16 GEMV at about 240 GB/s would be about 1.4x faster than cuBLAS there.
- **M sweep:** from M=2, cuBLAS switches to `cutlass_80_wmma_tensorop_bf16` (an sm80 GEMM), with `splitKreduce` added for small N.
  - On every v6 shape and on the lm_heads, **t(M=2..16) is 0.69 to 0.82x t(M=1)**: the GEMM does 2 to 16x the work faster than the M=1 GEMV.
  - On the v7 shapes it is 1.00 to 1.12x.
  - Qwen k/v_proj at M >= 2 (split-K GEMM) is 1.16 to 1.28x.
- **Single-model Hockney fits across kernels misfit (8 to 16%)** because efficiency depends on which kernel is chosen, not on size. `analyze_e2.py` therefore fits per dispatched kernel.
  - The `e2_efficiency.png` in these summaries still uses the pooled (misfit) fit and should be read as provisional.
  - The per-kernel table is the valid reading.
- **Launch split (pooled):** exposed host launch cost (eager - graph) is about 2 us at K=4096 and about 3 us at K=14336. Isolated - graph is about 9 to 11 us.

### Methodology problem found: variance between allocations (affects how precise E2 numbers are)

- **Cross-session agreement is worse than in E1:** median |diff| 2.08% (E1: 0.27%), 48% of cells within 2%, max 15%. The sessions do not shift systematically: signed median +0.5%, identical SM clock (2190 MHz) and robust within-cell spread (0.45%).
- **Within one session, identical-shape pairs** (k_proj vs v_proj, gate_proj vs up_proj: same N, K and even the same weight data) differ by median 1.4 to 1.7%, up to 17%.
- **Direct test** (`scripts/diag/realloc_variance.py`, `e2s2_20261019/diag_realloc_variance.json`): the graph-mode median measured 10 times on one allocation spans 0.1 to 1.3%, but over 10 fresh allocations it spans 6.2% (1024x4096), 2.0% (4096x4096), 2.0% (512x3584) and 0.25% (14336x4096).
- **Interpretation:** a cell's median is precise for its own allocation, but small and mid-size ops vary by several percent between allocations. Physical page placement across LPDDR5x channels and banks is the likely cause (unverified). Large buffers average it out, which is why E1 (mostly large) reproduced well.
- **Proposed fix (not yet applied):** measure each cell over several independent allocations and report the median across allocations (plus the spread), as schema_version 2.

## Schema v2 reruns, GB10: E1 and E2 (results of record)

Each cell is measured on 5 independent allocations; its value is the median across allocations (code `ad228d9`, rebased as
`2c0aeb6`). Sessions `e1v2_s1`, `e2v2_s1`, `e1v2_s2` and `e2v2_s2` ran back to back on 2026-10-18 from 21:48 to 23:42 UTC, so
same-phase sessions are about 1 hour apart (not different days). All 1,156 cells were clean (10 attempts retried). **These v2
sessions supersede the v1 numbers above for all later use; the v1 sessions stay in the repo as the record.**

| Quantity | v2 s1 | v2 s2 | (v1 s1 / s2) |
|---|---|---|---|
| R_inf_B0 | 239.0 GB/s | 239.5 GB/s | 241.7 / 242.1 |
| probe tau, Triton read, graph / eager / isolated | 1.70 / 4.85 / 11.31 us | 1.66 / 4.66 / 11.43 us | 1.85 / 4.7 / 11.4 |
| B_half (graph) | 0.39 MiB | 0.38 MiB | 0.43 |
| copy / write plateau | 240.2 / 196.4 GB/s | 241.8 / 196.8 GB/s | 239 / 197 |
| E2 `gemvx/v6` fit (graph): tau, R_inf | | 0.38 us, 167.8 GB/s (1.9% residual) | 1.95 us, 169.9 |
| E2 v6 shapes, %B0 | 68 to 72% | 68 to 72% | 65 to 73% |
| E2 v7 shapes (MLP, down_proj, Qwen k/v), %B0 | 87 to 98% | 87 to 98% | 84 to 104% |
| E2 lm_head, %B0 | 72 to 73% | 72 to 73% | 75 |

**Reproducibility, v2 vs v1**

| | E2 v1 | **E2 v2** | E1 v1 | E1 v2 |
|---|---|---|---|---|
| cross-session median |diff| | 2.08% | **0.27%** | 0.27% | 0.38% |
| max |diff| | 15% | **6%** | 30% | 20 to 25% (hot L2-edge cells) |
| cells above 3% | 64 / 218 | **8 / 218** | 21 / 360 | 17 to 18 / 360 |
| per-allocation spread (median / p90) | n/a | 0.44% / 1.3 to 1.7% | n/a | 0.38% / 2.9 to 3.2% |

- **v2 removes most of E2's allocation variance.** The remaining E1 outliers are the hot cells at the L2 edge, where a single
  allocation's placement decides how much of the buffer stays in L2 (per-allocation spread up to 85% on those cells).
- **B0 is about 1% lower in v2** (239 vs 242 GB/s). The median across 5 allocations is more representative than one allocation;
  v2 B0 is the denominator for later phases.
- **The E2 conclusion is unchanged and sharper:** `gemvx/v6` has essentially no fixed cost beyond the probe's (graph tau 0.4 us,
  CI includes 0) and streams at 168 GB/s (70% of B0). The BF16 loss on GB10 is per byte, set by cuBLAS's kernel choice.

## E3 (focused): why cuBLAS `gemvx/v6` streams 30% slower than v7 (GB10, 2026-10-21)

`scripts/ncu_gemv_variants.sh` profiles one M=1 `F.linear` call per shape with Nsight Compute (`--clock-control none`, idle
machine). Raw CSVs: `results/dgxspark_gb10/e3_ncu_gemv_variants/`. ncu durations are not headline numbers.

| | v6: Llama q_proj | v6: Qwen gate_proj | v7: Llama gate_proj | v7: Llama down_proj |
|---|---|---|---|---|
| grid x block | 1024 x 64 (16 x 4) | 4736 x 64 | 3584 x 128 (32 x 4) | 1024 x 128 |
| registers per thread | 164 | 164 | 164 | 164 |
| occupancy theoretical / achieved | 25% / 21.4% | 25% / 24.0% | 25% / 25.0% | 25% / 24.8% |
| DRAM fills (sysmem x 32 B) / weight bytes | 1.002 | 1.00 | 1.002 | 1.00 |
| long-scoreboard stall cycles per issue | 29.4 | 31.8 | 17.4 | 20.2 |
| LSU throttle stall per issue | 23.4 | 18.7 | 14.6 | 21.4 |
| issue active | 4.9% | 4.9% | 6.7% | 6.0% |

- **Both variants move exactly the weight bytes** and are capped at 25% occupancy (12 of 48 warps per SM) by 164 registers
  per thread.
- **v6 uses 16-lane rows (block 16 x 4) instead of 32 (32 x 4)**, so each warp keeps about half the bytes in flight. With the
  same resident warps, that halves the memory-level parallelism; v6 warps wait on memory (long scoreboard) about 1.7x longer
  per issued instruction. **v6 is latency-bound for lack of bytes in flight, not moving extra data.**
- This is the mechanism P1's kernel targets: more bytes in flight per warp (32-lane rows, wide vector loads, several rows per
  warp) and a smaller register budget for higher occupancy.

## E4 harness: library FP8 / INT4 ops (GB10, 2026-10-23, functional checks only)

Code: `bench/ops/lib_quant.py`, phase `e4` in `bench/run.py`, `bench/analyze_e4.py`.

### Verified on GB10 (correctness before any timing)
- All six ops are correct (max rel. error 1.9e-3 to 3.4e-3, gate 2e-2) and graph-capturable on all 20 real shapes at M = 1,
  and on q_proj, k_proj and lm_head of both models at M = 16 (172 checks).
- **Dispatch at M = 1:**
  - `torch._scaled_mm` per-tensor FP8 does **not** use one kernel. cuBLASLt picks by shape: `nvjet_sm121` tiles
    (64x64, 64x32, 32x32, 128x64, 128x72, 192x112, ...), and for gate/up (N = 14336 to 18944) and both lm_heads an
    **`sm89_xmma_gemm_e4m3bf16`** kernel (an Ada GEMM, not a Blackwell one).
  - vLLM `cutlass_scaled_mm` row-wise FP8: one CUTLASS `MainloopSm120TmaWarpSpecialized` GEMM on every shape.
  - tinygemm: `tinygemm_m16n8k16_chunk_kernel` (innerKTiles 8). Marlin: `marlin::Marlin<bf16, ...>` (one template
    instance at M = 1, another at M = 16).
  - Dynamic activation quantization (vLLM `scaled_fp8_quant`): per-tensor = memset + `segmented_max_reduction_strided`
    + `scaled_fp8_quant_kernel_strided_dynamic` (3 operations); per-token = one `dynamic_per_token_scaled_fp8_quant_kernel`.
- **Marlin weights are built the way vLLM loads a GPTQ checkpoint:** GPTQ layout (`qweight [K/8, N]` int32) then
  `ops.gptq_marlin_repack` on the GPU. vLLM's `marlin_quantize` test helper packs on the CPU and is too slow for lm_head.
- **Shape support:** Marlin needs N % 64 == 0 (K % 128): synthetic N = 32 is left out of its cells. Every other op runs
  every shape. `run.py --phase e4 --list` shows 1,146 cells (4 GEMM kernels x 218 E2-style cells, minus 6 for Marlin,
  plus 2 full variants x 140 real-shape cells).
- Only copy 0 of each rotation holds real quantized data; copies 1.. hold random codes (FP8 codes with NaN patterns
  excluded). These kernels have no data-dependent control flow, so timing does not depend on the values.
- Bytes are the stored format: FP8 1 B/weight + fp32 scales; tinygemm 0.5 B/weight + bf16 scale and zero per 128
  (4.25 bits); Marlin 0.5 B/weight + bf16 scale per 128 (4.125 bits). For the `*_full` ops the fp8 copy of x is
  intermediate traffic and is not counted.

### Analysis decisions
- **Efficiency against cuBLAS BF16 exceeds 100% on most shapes**, because cuBLAS BF16 itself runs at about 70% of B0 on
  13 of 20 shapes (E2). The meaningful efficiency is %B0 (efficiency against an ideal kernel at B0); both are reported.
  The identity efficiency == BW_q / BW_16 holds to 4e-16 (it is algebra; the check guards the code).
- **One `t = tau + B / R` fit does not describe the tiled library kernels** (Marlin 22%, tinygemm 13%, CUTLASS 26% rms
  residual over the synthetic sweep; Marlin's fit even reports R above B0). The per-op / per-byte split therefore uses
  each dispatched kernel's asymptotic rate R_asym (Theil-Sen slope of t against B over ops >= 64 MiB):
  per-byte = B (1/R_asym - 1/B0), per-op = (t - B/B0) - per-byte. Kernels with fewer than 3 large ops get no split.

### Debug observations (session `debug_e4a`, gate bypassed, graph M = 1 only: hints, NOT results)
- The quantized library kernels stream large ops at 92 to 98% of B0 (INT4 tinygemm and Marlin about 95 to 98%), better than
  cuBLAS BF16's v6 shapes. FP8 per-tensor's `sm89_xmma` kernel looked slower (about 83% of B0 on gate/up/lm_head).
- The loss concentrates on **small ops**: Qwen k/v (512 x 3584) at 24% (CUTLASS FP8), 28% (Marlin), 52% (tinygemm) and
  65% (FP8 `_scaled_mm`) of B0, against 87% for cuBLAS BF16. CUTLASS FP8 and Marlin spend 24 us and 10 us there beyond
  B/B0 (which is 8 us and 4 us). Per-tensor dynamic activation quantization added about 14 us per op.
- Predictions: the 2026-10-16 E4 predictions (FP8 `_scaled_mm`, tinygemm, Marlin) were committed before any E4 number.
  No prediction was added for row-wise CUTLASS FP8 or the `*_full` variants, because debug numbers for them had been seen.

## Platform `rtxpro5000bw`: environment and preliminary E1 (P3)

Raw evidence: `paia-results/rtxpro5000bw/{env_20261016,s1_20261016}/`, preflight report on the host at
`paia-bench/results/preflight/server_20261016_200948.txt`. E1 numbers below come from `bench.analyze` run on a copy
of the 80 cells (summary not committed, since the session is incomplete).

### Machine and stack

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA RTX PRO 5000 Blackwell, cc 12.0, 110 SMs, L2 100,663,296 B (96 MiB), 50,754,289,664 B memory, 384-bit GDDR7 | `env.json` gpu, nvml |
| Spec bandwidth | 1344 GB/s (384 bit x 28 Gbps at the 14001 MHz max memory clock) | NVML max mem clock, bus width |
| Memory clock under load | 13365 MHz in every busy cell, i.e. 1283 GB/s raw at the running clock | cell telemetry |
| Driver | 595.84; ECC disabled (current and pending); persistence on; power limit 300 W; max SM clock 3090 MHz | `env.json` nvml |
| CPU | Intel Core Ultra 9 285K, 24 cores, 1 NUMA node; 125 GiB RAM | `env.json` cpu |
| Stack | host venv (no Docker on this server): torch 2.14.1+cu130, cuBLAS 13.1.1, Triton 3.8.0, Python 3.12.3, no vLLM | `env.json` packages, cublas |
| Profilers | host ncu 2026.2.1, nsys 2026.1.3; `RmProfilingAdminOnly: 0`, ncu probe as uid 1019 succeeded | `env.json` perf_counters, preflight |
| Sharing | single GPU, shared by course teams; idle at survey time | preflight |

- **The stack differs from GB10's** (NGC 25.11: torch 2.10.0a0, cuBLAS 13.1.0, Triton 3.5.0). The server has no Docker, so the
  `paia-bench:25.11` image is not an option unless the admin provides Docker. Cross-platform comparisons carry this confound.
- **Triton needed Python headers.** The first venv used Ubuntu's Python without `python3.12-dev`; Triton could not build its
  launcher (`Python.h: No such file`), so the Triton probe failed. Fixed before E1; the selfcheck in `env_20261016` is the fixed run.

### Library paths (`selfcheck.json`, functional only)

- **BF16 vs CPU, cuBLAS BF16 on 6 shapes at M = 1 and 16:** correct and graph-capturable.
  - M=1 dispatches `internal::gemvx::kernel` on every shape, as on GB10.
  - M=16 dispatches `cutlass_80_wmma_tensorop_bf16` (an sm80 GEMM), with `splitKreduce` for 1024x4096 and 512x3584, as on GB10.
- **FP8 `_scaled_mm`, per-tensor scales:** correct at M = 1 to 16 (rel. error 0.003), graph-capturable, `nvjet_sm120_qqtst_mma_64x32x64...`. M=1 needs no padding.
- **FP8 `_scaled_mm`, row-wise scales: correct on sm_120** (rel. error 0.0026 to 0.0028), unlike GB10 (sm_121), where it returns garbage.
  It dispatches the same `cutlass ... enable_3x_kernel_for_sm10_or_later` kernel that fails on GB10, **plus a `Memcpy DtoD`**,
  so row-wise FP8 costs an extra operation per call on this stack. Row-wise FP8 is therefore usable here, but not on GB10.
- **INT4 tinygemm:** correct (rel. error 0.004) and graph-capturable at M = 1 and 16; kernel `tinygemm_m16n8k16_chunk_kernel`.
- **vLLM CUTLASS FP8 and Marlin:** not tested (vLLM not installed).

### Preliminary E1 (session `s1_20261016`, 80 of 360 cells, all clean, code `ddf2d53`)

The run was stopped by hand after 80 cells. Cells are a fixed shuffled sample, so some op/mode series have only 1 to 4 sizes.
**Treat the fits as indicative.** The plateaus rest on several independent cells and are solid.

| Quantity | RTX PRO 5000 (s1, partial) | GB10 (s1 / s2) |
|---|---|---|
| **R_inf_B0**, cold graph Triton read, >= 10x L2 | **1238.8 GB/s** | 241.7 / 242.1 |
| as % of spec | **92.2%** of 1344 (96.6% of the 1283 GB/s at the running memory clock) | 88.6% |
| read plateaus, all probes and modes | 1232 to 1239 GB/s | |
| copy plateau (2B counted) | 1105.6 GB/s, **0.89x read** | 0.99x |
| write (fill) plateau | 1233.6 GB/s, **1.00x read** | 0.81x |
| probe tau, Triton read, graph mode | 1.36 us [1.36, 1.51] (4 sizes, smallest 8 MiB: poorly constrained) | 1.85 / 1.86 |
| **B_half**, Triton graph mode | **1.61 MiB** | 0.43 |
| hot L2 peak, Triton read graph | 4572 GB/s at 16 MiB; falls off by 128 MiB | 1505 at 16 MiB |
| hot copy peak | 5547 GB/s at 32 MiB | |
| canary reference | 1127.0 GB/s, CV 0.04%; per-cell canaries 1123 to 1127 (n=160) | 240.2, CV 0.22 to 0.83% |

- **Hypothesis check (early): B_half is about 3.7x larger than on GB10** (1.61 vs 0.43 MiB), driven by 5.1x the bandwidth with a
  similar or smaller tau. This is inside P2's prediction of "3 to 6x larger B_half", but the measured tau on both platforms
  (1.4 and 1.9 us) is below the predicted 3 to 5 us.
- **GDDR7 writes are as fast as reads** (fill 1.00x read), unlike GB10's LPDDR5x (0.81x). Copy is the slow pattern here (0.89x).
- **The canary is about 10x quieter than on GB10** (CV 0.04% vs 0.2 to 0.8%): dedicated GPU memory, no CPU traffic on the bus.
- **Eager-mode per-op floor is about 5.8 us** for the Triton probe (cold reads of 0.25 to 2 MiB all take 5.8 us), so eager
  mode is host-launch-bound below about 7 MiB. The fit across sizes misfits (16% residual) for this reason. GB10's eager tau was 4.7 us.
  The benchmark process was pinned to CPUs 2 and 3 only (see below), and Triton 3.8's launcher differs from 3.5's, so this is
  not yet a CPU comparison.
- **torch `sum` and `fill_` fits** here rest on 1 to 4 points; the GB10 conclusion that only the Triton probe defines tau stands.

### Measurement issues found on this platform

1. **SM clock is not stable under load.** In graph and eager cells with the GPU 84 to 100% busy and the memory clock at
   13365 MHz, the median SM clock ranged from 180 to 2595 MHz between cells (GB10: 2177 to 2190 MHz in every cell).
   - Large-buffer bandwidth is unaffected: busy cells of 512 MiB and up reach 1233 to 1239 GB/s at SM clock medians from
     765 to 2595 MHz.
   - **tau is likely affected,** since launch, ramp and tail run on the SM clock. Clock state therefore has to be controlled or
     stratified before small-op and tau numbers from this GPU are compared across platforms.
   - The `sw_power_cap` reason was seen; no thermal or hardware slowdown (max 41 C).
   - Isolated-mode cells finish in well under 1 s, so they got only 1 to 3 telemetry samples, all taken while the GPU was idle
     during setup (memory clock 405 MHz). **Their clock telemetry is uninformative, not low.**
2. **Same gate slowdown as GB10 s1.** After almost every cell the gate waited 30 s for NVML's trailing utilization from our
   own cell (`gpu_util max 100%`): 80 cells took 45 minutes. The settle step added in `247bdc4` fixes this; it does not
   change the measurement code (`timing.py`, `bw_probe.py` are identical between `ddf2d53` and HEAD).
3. **Benchmark pinned to 2 cores.** `cpu_sets()` selects cores whose `cpuinfo_max_freq` equals the maximum. This CPU reports
   three levels (2 cores at 7.2 GHz, 6 at 6.9, 16 at 6.5), so the benchmark ran on CPUs 2 and 3 only and the monitor on the 16
   slowest cores. Intended: all 8 performance cores. **Fixed:** `cpu_sets()` now takes every core above the lowest
   frequency level as fast (8 P-cores here; unchanged 10-core split on GB10).

### Next steps on this platform

- Ask the admin whether `nvidia-smi -lgc` can be used during our windows (it fixes item 1). Otherwise add a clock check:
  record the SM clock inside the timed loop (or make `sw_power_cap`/low-clock cells retry) and report tau per clock band.
- Because the CPU pinning changed (item 3), do not resume `s1_20261016`: its 80 cells stay as a pilot. Run a fresh full
  E1 session on current HEAD, then `s2` on another day.
- Predictions for this platform were not committed before these cells ran, so E1 predictions here are no longer
  pre-registered. Commit E2, E4 and E5 predictions for `rtxpro5000bw` before those phases run.

## Open questions

1. Is `lts__t_sectors_aperture_sysmem_op_write` the right write-bytes counter? To be confirmed in E3 with a `fill_` kernel of known size.
2. Does an idle, unthrottled GB10 hold a stable SM clock across a session? Cell telemetry will show it.
3. Canary sensitivity: how much does a CPU-only memory-bound job by another user reduce GPU streaming bandwidth? This could be quantified with our own CPU load, if P2 approves.
5. Why is streaming bandwidth about 4% higher at <= 100 MiB footprints (about 252 GB/s) than at GiB scale (241 GB/s)? TLB reach is the leading hypothesis; testable with ncu TLB or page-walk counters.
6. Is the slow-sample tail (bursts of tens of ms) caused by other users' sub-threshold CPU memory traffic, LPDDR housekeeping, or NVML? Medians are unaffected, so this is low priority.
4. Does an RTX 5000-class machine have driver 580 or newer, so P3 can use the same image?
