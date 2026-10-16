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

## Open questions

1. Is `lts__t_sectors_aperture_sysmem_op_write` the right write-bytes counter? To be confirmed in E3 with a `fill_` kernel of known size.
2. Does an idle, unthrottled GB10 hold a stable SM clock across a session? Cell telemetry will show it.
3. Canary sensitivity: how much does a CPU-only memory-bound job by another user reduce GPU streaming bandwidth? This could be quantified in Phase 1 with our own CPU load, if P2 approves.
4. Does an RTX 5000-class machine have driver 580 or newer, so P3 can use the same image?
