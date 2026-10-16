# P3 Guide: Running the PAIA Baseline Benchmarks on the RTX 5000 Server

This guide is meant to be complete on its own. Read Parts 1 to 3 to understand what we are doing and why. Part 4 is the step-by-step procedure. Appendix A is the full technical specification. If our team repo is not ready, you can hand Appendix A to a Claude Code session and it will build the harness.

---

## Part 1: What the project is

**Course project:** Fused Low-Precision GEMV Kernels for LLM Decode (Programming AI Accelerators).

**Team roles:**

- **P1** writes all custom GPU kernels.
- **P2** owns measurement and profiling, and is running the same benchmarks on a DGX Spark (GB10).
- **You (P3)** run them on the RTX 5000 server, which gives us a third hardware data point.

### The physics in three sentences

When a language model generates one token at batch size 1, the GPU reads every weight matrix once and does almost no arithmetic per byte, so time is roughly bytes ÷ memory bandwidth. Quantizing weights from 16-bit to 8-bit or 4-bit cuts bytes by 2x or about 4x, so the naive prediction is a 2x or 4x speedup. Measured speedups are much smaller (one 2026 study saw 1.38x for 4-bit on an NVIDIA L4), and our project explains why.

### The model we test

Each operation costs

```
t(B) = tau + B / R_inf
```

| Symbol | Meaning |
|---|---|
| B | Bytes the op reads |
| R_inf | Streaming bandwidth the kernel reaches on large ops |
| tau | Fixed per-op cost: launch, gap between kernels, pipeline ramp-up, tail |

From this:

- An op of size `B_half = tau * R_inf` runs at only 50% of R_inf.
- Quantization makes ops smaller but leaves tau unchanged, so small quantized ops lose efficiency.

That gives two kinds of loss. A **per-byte** loss means a lower R_inf. A **per-op** loss means tau. One useful identity:

```
actual speedup = (bytes_16bit / bytes_quantized) * (achieved_BW_quantized / achieved_BW_16bit)
```

So the gap between predicted and real speedup is exactly a ratio of achieved bandwidths.

### Why your GPU matters

Each GPU is one (tau, R_inf) point. The hypothesis is that faster-memory GPUs have a larger B_half, so they lose more to tau when ops shrink. Your RTX 5000 sits at a different bandwidth from both the GB10 (273 GB/s spec) and our RTX 6000 Ada (960 GB/s spec), so it tests whether the model predicts across hardware.

---

## Part 2: What we are running right now, and why no custom kernels yet

Before P1's kernels can be judged, we need a ruler. All experiments below use existing libraries or trivial probes.

| ID | Experiment | Why | What to expect on a discrete GPU |
|---|---|---|---|
| E0 | Fingerprint the machine and software | Results need provenance | Spec facts confirmed from the device itself |
| E1 (B0) | Read, copy and write bandwidth vs buffer size, hot vs cold | The real bandwidth ceiling (never use the spec number) and the smallest possible tau | Cold read somewhat below spec. A large hot-run plateau above DRAM speed until the buffer exceeds L2. Ada and Blackwell parts have big L2 caches |
| E2 (B1) | cuBLAS BF16 matrix-vector products on real Llama/Qwen layer shapes, plus a size sweep, in 3 timing modes | The production yardstick, plus fitted tau and R_inf for a real GEMV | Large layers near the B0 ceiling; small layers (K/V projections) clearly lower. The higher your bandwidth, the larger this drop should be |
| E3 | Nsight Compute smoke test | Confirms we can count DRAM bytes, so later claims are metric-backed | Usually works on discrete GPUs if profiling permissions allow |
| E4 | Library FP8 and INT4 kernels on the same shapes | First measured vs theoretical speedups | Below theory, especially on small shapes |
| E5 | Whole-token decode with an nsys trace | How many kernels a token takes, how much is matmul, how much is idle gap | Tokens/s below the bandwidth floor. Eager mode is more launch-bound than CUDA Graphs |

### The three timing modes

These split tau into its parts:

- `isolated`: synchronize, run one op, synchronize. Full latency of a single op.
- `eager`: a Python loop of back-to-back ops. Adds the CPU's launch cost when the CPU cannot stay ahead.
- `graph`: the same loop captured in a CUDA Graph. Removes the CPU cost, leaving only GPU-side gaps, ramp and tail.

Then `tau_eager - tau_graph` is roughly the exposed CPU launch cost, and `tau_graph` is what only fusion or persistent kernels can remove.

### Two rules that decide whether the numbers are real

1. **L2 rotation.** Your GPU's L2 cache is tens of MB, larger than many layer matrices. If you call the same matrix repeatedly, it stays in L2 and you measure cache speed, not memory speed. Every cell cycles through enough copies of the weights to exceed L2 by at least 4x.
2. **Count metadata in theoretical speedups.**

   | Format | Bits per weight | Theoretical speedup vs BF16 |
   |---|---|---|
   | FP8, per-tensor scale | 8 | 2.0x |
   | INT4, group 128, scale + zero | 4.25 | 3.76x |
   | INT4, group 128, scale only | 4.125 | 3.88x |

---

## Part 3: Running on a shared server

Others may use the server. A benchmark that ran while someone else was using the GPU is invalid. You can't stop other people, so the harness copes instead.

### Don't pause and resume a measurement

Freezing a run halfway (for example with SIGSTOP) has three drawbacks:

- the timing window across the pause is contaminated;
- after resuming, the cache contents, clocks and temperature differ;
- the paused process still holds GPU memory.

### What the harness does instead

1. **Short cells.** It splits the work into cells of about 15 seconds, each covering one configuration, and saves each result immediately.
2. **Idle check before each cell.** No other process on your GPU, GPU utilization near zero, low CPU load from other users. If the machine is busy, it waits and retries.
3. **Live monitor during each cell.** Any foreign activity marks the cell "contaminated" and puts it back in the queue.
4. **Bandwidth canary.** A ~30 ms memory-read probe runs before and after each cell and is compared against a reference taken while the machine was idle. A drop of more than 5% invalidates the cell.
5. **Resumable runs.** Restarting skips cells that are already clean, so an interruption costs at most one cell.
6. **At least two sessions** on different days or times. Results should agree within about 3%.

### Discrete-GPU specifics

- If the server has several GPUs, pin yours with `CUDA_VISIBLE_DEVICES`. Other people's jobs on *other* GPUs are mostly harmless to the bandwidth benchmarks. They can still affect `eager` mode (CPU contention) and PCIe, so the monitor also watches CPU load.
- GPU memory is separate from CPU memory here. CPU-only jobs are less of a threat than on the DGX Spark, where CPU and GPU share one memory bus.
- Run long sessions inside `tmux`, preferably at quiet hours. Tell the other users your time window.

---

## Part 4: Step by step

### Step 1: Identify your GPU exactly

"RTX 5000" can mean three different products. Run:

```
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,ecc.mode.current,persistence_mode --format=csv
```

Then send P2 the output. What changes by variant (verify all of these on your machine; the spec numbers are approximate):

| Variant | Compute cap | Memory | Notes |
|---|---|---|---|
| RTX 5000 Ada Generation | 8.9 | 32 GB GDDR6, about 576 GB/s | FP8 supported; no Programmatic Dependent Launch; large L2. ECC can be enabled, and ECC can reduce bandwidth, so record its state |
| RTX PRO 5000 Blackwell | 12.0 | 48 GB GDDR7, about 1.3 TB/s | FP8 and FP4 hardware; highest bandwidth, so the tau effect should be strongest; needs a recent driver and CUDA 12.8+ PyTorch build |
| Quadro RTX 5000 (Turing) | 7.5 | 16 GB GDDR6, about 448 GB/s | No BF16 tensor cores, so use FP16 as the 16-bit baseline and note it; no FP8; PyTorch int4 tinygemm needs compute cap 8.0+ (likely unavailable); 16 GB cannot hold a BF16 8B model, so skip E5 or use a smaller model after asking P2 |

The harness reads every hardware property (SM count, L2 size, memory) from the device at runtime. Nothing about your GPU is hardcoded.

### Step 2: Ask the admin these questions, before running

- Can I use the GPU exclusively during agreed windows?
- Am I allowed to lock clocks with `nvidia-smi -lgc` / `-lmc` (needs sudo)?
- Are GPU performance counters available to non-root users? Nsight Compute needs this.
- Is ECC on or off? Please don't change it between our sessions.
- Is Docker with NVIDIA's NGC containers available?

If clocks can't be locked, that's fine. The harness logs clocks and temperatures for every cell, so we can spot throttling afterwards.

### Step 3: Get the code

- **Preferred:** clone the team repo (`paia-bench`) at the **exact commit P2 gives you**. Identical code on every platform is what makes the comparison valid. Read its `README.md`.
- **If the repo isn't available:** start a Claude Code session in an empty directory, paste Appendix A, and let it build the harness phase by phase. The shapes, timing rules and result schema in Appendix A must stay exactly as written, so your results merge with P2's.

### Step 4: Environment

- Use a recent PyTorch with CUDA support matching your GPU. An NGC PyTorch container is the easiest route.
- Install `nvidia-ml-py`, `psutil`, `numpy`, `scipy`, `matplotlib`, `pyyaml`; plus `triton` if not bundled.
- Install Nsight Systems (`nsys`) and Nsight Compute (`ncu`) if they aren't already there.
- Optional, for E5 and Marlin: vLLM. Time-box any build problem to about 1 hour and record it.
- Request Hugging Face access to `meta-llama/Llama-3.1-8B-Instruct` (gated). `Qwen/Qwen2.5-7B-Instruct` is open. Each needs about 16 GB of disk.

### Step 5: Before measuring, write predictions

Fill in `predictions.md` with your guesses:

- the B0 cold-read bandwidth as a percentage of spec;
- tau in µs;
- the efficiency of the k_proj shape vs gate_proj;
- the FP8 and INT4 speedups.

Pre-registering guesses takes 10 minutes and makes our results more credible.

### Step 6: Run the phases in order, checking each gate

Use a platform tag like `rtx5000ada` (match your variant) and a session ID like `s1_2026-10-20`:

```
tmux new -s paia
python -m bench.env --platform-tag rtx5000ada --session s1
python -m bench.run --phase e1 --platform-tag rtx5000ada --session s1 --resume
python -m bench.run --phase e2 ...      # same flags
bash scripts/ncu_smoke.sh ...           # E3
python -m bench.run --phase e4 ...
bash scripts/nsys_token.sh ...          # E5
python -m bench.analyze --platform-tag rtx5000ada --session s1
```

(Exact flags may differ; follow the repo README.) If a run is interrupted, rerun the same command: it resumes.

| Gate | What to check before moving on |
|---|---|
| After E1 | Bandwidth-vs-size plot exists; cold-read plateau and tau fit reported; canary stable; 5 repeated cells agree within 2% |
| After E2 | Per-shape table and fit plot; cuBLAS kernel names recorded |
| After E3 | Measured DRAM bytes within about 5% of expected on the large shape, or the exact reason profiling failed |
| After E4 | Measured vs theoretical speedup table |
| After E5 | Tokens/s, kernels per token, matmul share and idle-gap time, for eager and graph modes |

### Step 7: Second session, then hand off

- Repeat the whole suite on another day as session `s2`, then run `analyze` again. It compares the sessions.
- Push `results/<platform_tag>/` (summaries, `cells/` JSON, `env.json`, `findings.md`) to a branch, or zip it and send it to P2. Do not include large raw `.nsys-rep` / `.ncu-rep` files unless asked.
- Message P2 with your GPU variant, anything surprising, and any phases that failed and why.

### Mistakes to avoid

- Using the spec-sheet bandwidth as the denominator. Always use your measured B0.
- Benchmarking one small matrix in a loop. You'd be measuring L2.
- Using Nsight Compute durations as performance numbers. ncu changes clocks by default; the harness uses `--clock-control none` and only reads counters from ncu.
- Mixing contaminated and clean cells, or deleting failed cells. Failures are information.
- Changing shapes or the result schema.

---

## Appendix A: Technical specification (also usable as a Claude Code brief)

> To Claude Code: you are helping P3 build and run a resumable benchmark harness on a discrete NVIDIA RTX 5000-class GPU on a shared server. Use library kernels and trivial measurement probes only; the team's custom kernels come later from P1. Verify every API, version and counter name on this machine instead of assuming; record findings in `findings.md`. Read all hardware properties at runtime. Never kill other users' processes; ask before any sudo or system install. Work phase by phase (E0 to E5, then the report) and stop at each gate for confirmation. Do not use em dashes in any file you write.

### A.1 Shared-machine protocol

- **Cells.** One configuration per cell, 15 s or less. Each cell writes `results/<platform_tag>/<session_id>/cells/<cell_id>.json` atomically (temp file plus rename); `cell_id` is a stable hash of the config.
- **Resume.** The runner skips cells with `status == "clean"` and retries contaminated cells up to 5 times. After that the status is `unclean_exhausted`.
- **Idle gate before each cell.** No foreign compute processes on the target GPU (NVML); device utilization below 5% for 2 s; other users' CPU use below 2 cores (psutil); canary OK. If any check fails, back off exponentially from 30 s up to 10 min, and log the reason.
- **Live monitor.** A 100 ms sampling thread records foreign processes, utilization, SM and memory clocks, temperature, power, throttle reasons, and other users' CPU. Any foreign activity marks the cell contaminated, with reasons.
- **Canary.**
  - Probe: a cold read over rotating buffers (512 MB total), 10 repetitions.
  - Reference: median of 20 idle runs at session start, requiring CV under 2%.
  - Check: before and after each cell; below 0.95 times the reference means contaminated.
  - Refresh: re-reference hourly when idle.
- **Courtesy lockfile** `/tmp/paia_bench.lock`, containing user, contact and end time.
- **Pinning.** Honor `CUDA_VISIBLE_DEVICES`.

### A.2 Timing

- CUDA events only.
- **Warmup:** at least 20 calls and at least 200 ms. **Samples:** at least 30.
- **Statistics:** median, p5, p95, mean, std and CV; `noisy` if CV exceeds 3%. Keep the raw samples.
- **Modes:**
  - `isolated`: sync, event, one op, event, sync.
  - `eager`: events around R back-to-back calls; per-op time = elapsed / R.
  - `graph`: R calls captured in a CUDA Graph; time one replay, then divide by R.
  - Choose R so that a loop lasts at least 5 ms. If an op can't be graph-captured, record that and skip `graph`.
- **L2 rotation:** `n_copies = max(2, ceil(4 * L2_bytes / op_bytes))` within `--mem-cap-gb` (default 12). If coverage falls short, record `l2_safe: false`.
- **Bytes per call:** weights + metadata + activations in + output out, stored separately and as a total. GB/s = total / median. `pct_of_b0` is relative to the session's measured cold-read R_inf.
- **Weights:** random weights for timing.
- **Kernel names:** record the CUDA kernel names once per shape via `torch.profiler`.
- **Clocks:** log `nvidia-smi` clock-lock status, ECC mode and persistence mode in `env.json`.

### A.3 Shapes

Weights are `[N_out, K_in]`. Verify against each model's HF `config.json`.

**Llama-3.1-8B** (hidden 4096, intermediate 14336, 8 KV heads, head_dim 128, vocab 128256):

| Name | N x K |
|---|---|
| q_proj | 4096 x 4096 |
| k_proj, v_proj | 1024 x 4096 |
| o_proj | 4096 x 4096 |
| gate_proj, up_proj | 14336 x 4096 |
| down_proj | 4096 x 14336 |
| qkv_fused | 6144 x 4096 |
| gate_up_fused | 28672 x 4096 |
| lm_head | 128256 x 4096 |

**Qwen2.5-7B** (hidden 3584, intermediate 18944, 4 KV heads, head_dim 128, vocab 152064):

| Name | N x K |
|---|---|
| q_proj | 3584 x 3584 |
| k_proj, v_proj | 512 x 3584 |
| o_proj | 3584 x 3584 |
| gate_proj, up_proj | 18944 x 3584 |
| down_proj | 3584 x 18944 |
| qkv_fused | 4608 x 3584 |
| gate_up_fused | 37888 x 3584 |
| lm_head | 152064 x 3584 |

Ignore Qwen's QKV bias for timing.

**Synthetic sweep:** K = 4096 and K = 14336; N a multiple of 128, chosen so BF16 weight bytes span about 0.25 MiB to 1 GiB in about 14 geometric steps.

**Batch sweep:** M in {1, 2, 4, 8, 16} on real shapes, graph mode only. M = 1 uses all modes.

Freeze all of this in `configs/shapes.yaml`.

### A.4 Experiments

- **E0.** Write `env.json`:
  - GPU name, compute capability, SM count, L2 bytes, memory, max clocks, ECC, persistence;
  - driver, CUDA, torch / triton / vLLM versions, cuBLAS version if obtainable;
  - `nvidia-smi -q`, `uname -a`, CPU, `pip freeze`, container tag;
  - ncu/nsys versions, perf-counter access, clock-lock result.

  Check whether these work: Triton, `torch._scaled_mm`, `torch.ops.aten._weight_int4pack_mm` (and its signature), vLLM/Marlin.
- **E1 (B0).**
  - Ops: read (a reduction; bytes = B), copy (`copy_`; bytes = 2B), write (`fill_`; bytes = B); optionally a minimal vectorized Triton read probe as a cross-check.
  - Sizes: 256 KiB to 4 GiB geometric, within the memory cap; hot and cold; all modes.
  - Outputs: GB/s vs size plot; cold-read plateau `R_inf_B0`; fit `t = tau + B/R` on cold graph-mode reads (least squares plus Theil-Sen) giving tau, R_inf, B_half and R^2; residual plot; hot L2 plateau and its falloff size.
- **E2 (B1).**
  - `torch.nn.functional.linear(x, W)` in BF16. On compute cap 7.5, use FP16 and flag it.
  - Synthetic sweep and real shapes, with the modes from A.3.
  - Outputs: per-shape table (median_us, GB/s, pct_of_b0); Hockney fit per K and mode; `tau_eager - tau_graph`; efficiency curve over the measured points; predicted efficiency of each real shape at BF16 and at 4-bit size.
- **E3.**
  - Run ncu with `--clock-control none` on gate_proj and k_proj. Collect, adapting names to `--query-metrics`: `gpu__time_duration.sum`, `dram__bytes_read.sum`, `dram__bytes_write.sum`, `lts__t_bytes.sum`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__warps_active.avg.pct_of_peak_sustained_active`, `launch__grid_size`, `launch__registers_per_thread`.
  - Compare DRAM bytes against expected bytes, or record the exact blocker.
- **E4.**
  - **FP8** (compute cap 8.9 or higher): `torch._scaled_mm` with `float8_e4m3fn` weights passed as `W.t()`, per-tensor float32 scales, FP8 activations, bf16 output. Pad M if required and record `padded_M`. Try row-wise scales if supported.
  - **INT4 tinygemm** (compute cap 8.0 or higher): g128, scale+zero in bf16. Adapt to the installed PyTorch signature.
  - **Marlin** via vLLM ops: optional, 1 hour time-box.
  - **Table:** BF16 time; quantized time; measured speedup; theoretical speedup from stored bytes; efficiency; and BW_q / BW_16, which should equal the efficiency.
- **E5.**
  - Qwen2.5-7B-Instruct, then Llama-3.1-8B-Instruct if access and memory allow.
  - Batch 1, greedy, 128 new tokens; decode tokens/s, excluding prefill.
  - Configs: vLLM `enforce_eager=True` vs `False`; or HF transformers eager vs a CUDA-graph / `torch.compile(mode="reduce-overhead")` variant.
  - nsys with CUDA and NVTX tracing. Per token: kernel launches, kernel time, idle gap, matmul share, top 10 kernels.
  - Compare against the floor: per-token weight bytes (excluding embedding) ÷ `R_inf_B0`.
- **Report.** Generate `summary/summary.md`, `cells.csv`, plots, fits as JSON, the list of unclean cells, and a cross-session comparison (flag median differences above 3%).

### A.5 Cell JSON (`schema_version: 1`)

```
schema_version, cell_id, session_id, platform_tag, phase, op, impl, precision,
shape {M, N, K}, model, layer_name, timing_mode, hot_or_cold, R_per_iter,
n_copies, rotation_bytes, l2_bytes, l2_safe,
bytes {weights, metadata, act_in, out, total},
samples_us [...], median_us, p5_us, p95_us, mean_us, std_us, cv, noisy,
achieved_gbps, pct_of_b0, kernel_names [...],
status (clean | contaminated | failed | unclean_exhausted), contamination_reasons [...], retries,
canary_before_gbps, canary_after_gbps, canary_ref_gbps,
telemetry {sm_clock_mhz {min, median, max}, mem_clock_mhz {...}, temp_c_max, power_w_median, throttle_reasons_seen [...]},
error, git_commit, git_dirty, env_ref, t_start, t_end
```

### A.6 Repo layout

```
paia-bench/
  README.md  findings.md  predictions.md
  configs/shapes.yaml  configs/platform_notes.yaml
  bench/env.py  bench/monitor.py  bench/timing.py  bench/ops/  bench/run.py  bench/analyze.py
  scripts/ncu_smoke.sh  scripts/nsys_token.sh
  results/<platform_tag>/<session_id>/{env.json, cells/, logs/, ncu/, nsys/, summary/}
```
