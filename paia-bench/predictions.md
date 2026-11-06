# Predictions (written BEFORE running E1/E2)

These predictions are committed before the corresponding experiment runs, so the results can be compared
against what theory expected. Do not edit a prediction after its experiment has run; add a note below it instead.

Platform: `dgxspark_gb10`. Predicted by Claude, at P2's request, from first principles and published specs.
Date committed: 2026-10-16.

**Inputs used**
- GB10 spec: 273 GB/s LPDDR5x (256-bit at 8533 MT/s); 48 SMs; 24 MiB L2; max SM clock 3003 MHz.
- Verified model configs (`configs/shapes.yaml`).

**Not used**
- No valid measurement exists yet. The only GB10 timings seen so far come from a `debug_` session taken while another user's job held the GPU at 96%. They are contaminated by design and were deliberately ignored here.

## E1 (B0) achievable bandwidth

| Quantity | Prediction | Reasoning |
|---|---|---|
| Cold read plateau R_inf_B0 (GB/s) | **225** (range 200 to 245) | Streaming reads on LPDDR5x typically reach 80 to 90% of the interface peak. Refresh, bank and row overheads are larger than on GDDR/HBM, and the GPU shares the memory controller with the CPU complex. |
| R_inf_B0 as % of spec bandwidth | **82%** (73 to 90%) | 225 / 273. |
| Copy GB/s (counting 2B) vs read | **0.85x read, about 190 GB/s** | Mixed read/write streams pay bus turnaround penalties on LPDDR, so they run slower than pure reads. |
| Write (fill) GB/s vs read | **0.90x read, about 200 GB/s** | Pure writes avoid turnaround, but write-allocate and partial-write overheads usually keep them a little below reads. |
| tau from cold graph-mode read fit (us) | **torch `sum`: 6 us; Triton single-kernel read: 3.5 us** | Back-to-back kernels in a graph cost about 1 to 2 us of launch gap, plus ramp-up and tail. `sum` launches two kernels per call (memset + reduce), so it pays about twice the fixed cost. |
| B_half = tau * R_inf (MiB) | **torch: about 1.3 MiB; Triton: about 0.75 MiB** | 6 us x 225 GB/s = 1.35 MB; 3.5 us x 225 GB/s = 0.79 MB. |
| Hot (L2) plateau GB/s | **about 1,000 GB/s** (600 to 1,500), peaking at 4 to 16 MiB | The theoretical L2 bandwidth is several TB/s. At L2-resident sizes, though, each op takes only a few us, so tau caps the observed rate: an 8 MiB read at 4 us tau plus about 4 us transfer gives about 1 TB/s. |
| Size where the hot curve falls off L2 (MiB) | **16 to 24 MiB** | Effective L2 capacity is below the nominal 24 MiB (address hashing, set conflicts, other resident lines). Between 24 and 48 MiB, hot should converge to cold. |

## E2 (B1) cuBLAS BF16 GEMV

| Quantity | Prediction | Reasoning |
|---|---|---|
| tau_graph (us) | **4 us** | A single `gemvx` kernel: launch gap, ramp-up and tail. Each CTA reduces along K, so the tail is short. |
| tau_isolated (us) | **7 us** | Adds the event and sync bracket overhead and an un-overlapped cold start (no previous kernel to hide the ramp-up behind). |
| tau_eager - tau_graph, exposed launch cost (us) | **4 us** | PyTorch `F.linear` dispatch costs about 6 to 10 us per call on the host. For small GEMVs the GPU finishes sooner than the next launch arrives, so part of the host cost shows up as GPU idle time. On large shapes it is hidden, so the fit attributes it to tau. |
| R_inf of GEMV as % of R_inf_B0 | **90%** | A GEMV is a near-pure stream of W. The loss comes from imperfect coalescing at the K-tile edges, the reduction epilogue and wave quantization on 48 SMs. |
| Efficiency of smallest real shape (k_proj) at BF16 | **Llama k_proj (8 MiB): 91%; Qwen k_proj (3.5 MiB): 82%** | eff = B / (B + B_half), with B_half = 4 us x 0.9 x 225 GB/s = 0.81 MB. |
| Efficiency of k_proj at 4-bit size (B/4) | **Llama: 72%; Qwen: 53%** | The same tau applied to a quarter of the bytes. This is the mechanism the project studies: quantization shrinks B but not tau. |
| Any shape where cuBLAS picks a non-GEMV kernel at M=1 | **None** | Phase 0 selfcheck saw `gemvx` on 6 shapes covering K = 3584, 4096 and 14336. lm_head (N = 128k to 152k) might trigger a different heuristic, but the prediction is still `gemvx`. |

## E4 library FP8 / INT4 (M=1)

| Impl | Theoretical speedup | Predicted measured speedup | Reasoning |
|---|---|---|---|
| FP8 `_scaled_mm` per-tensor | ~2.0x | **1.5x on large shapes, 1.1x on k_proj** | It dispatches a GEMM-tiled `nvjet` kernel (64-row tiles) rather than a GEMV. At M=1 that wastes tile work and adds a scale epilogue, and its fixed cost is likely higher than `gemvx`'s. |
| INT4 tinygemm g128 scale+zero | ~3.76x | **2.2x large, 1.3x small** | tinygemm was tuned for A100. On sm_121, the dequant ALU work per byte and an older tile layout limit how many bytes are in flight. |
| INT4 Marlin g128 scale-only | ~3.88x | **2.8x large, 1.6x small** | Marlin is designed for small batches and hides dequant behind loads, but it still pays a larger tau (workspace and reduction) that dominates the small shapes. |

Per the identity S_actual = (bytes_16 / bytes_q) x (BW_q / BW_16), the ratio "measured / theoretical" should equal BW_q / BW_16 for every row.

**Note added 2026-10-26, after E4 (`e4_s1`, `e4_s2`):** all three predictions were too pessimistic. Measured speedups over
cuBLAS BF16 exceed the byte ratio on most shapes (INT4 3.6 to 5.2x on large shapes), because the BF16 baseline runs at about
70% of B0 on 13 of 20 shapes, while the quantized library kernels stream large ops at 83 to 98% of B0. The predicted
mechanism for small shapes (a larger fixed cost per op) did hold: Qwen k_proj reached only 1.2x (Marlin) and 2.1x (tinygemm).

## E5 whole-token decode (BF16, batch 1)

Bytes per token, excluding the embedding table and counting the lm_head (not tied): Llama 15.01 GB, Qwen 14.14 GB.

| Quantity | Prediction | Reasoning |
|---|---|---|
| Bandwidth floor tokens/s (bytes/token / R_inf_B0) | **Llama 15.0 tok/s (66.7 ms/token); Qwen 15.9 tok/s** | 225 GB/s / 15.01 GB; 225 / 14.14. |
| vLLM eager tokens/s | **Llama 12.8 tok/s** (85% of floor) | GEMVs run at about 90% of B0, and 32 layers of non-matmul kernels plus attention add about 3 to 4 ms. At 67 ms per token, host launch costs are mostly hidden behind GPU work. |
| vLLM CUDA graph tokens/s | **Llama 13.5 tok/s** (90% of floor) | Graphs remove the residual launch gaps; the per-op tau inside the graph remains. |
| Kernels per token | **about 350 to 450** | About 10 to 12 kernels per layer (norms, QKV, RoPE, attention, o_proj, add+norm, gate_up, activation, down) x 32, plus lm_head and sampling. |
| Matmul share of kernel time | **about 90%** | Weight streaming dominates at batch 1. Attention with a short context is small. |
| GPU idle fraction between kernels (eager / graph) | **6% / 1.5%** | With per-token GPU time around 67 ms, the GPU stays well ahead of the host, so even eager mode exposes little idle time. |

## Hypothesis check

Because B_half = tau x R_inf, GB10 should be less tau-sensitive than a fast discrete GPU unless its tau is larger.

**Prediction:** GB10 graph-mode tau is about the same as a discrete Blackwell GPU's (about 3 to 5 us on both). Kernel launch and ramp-up are GPU front-end costs and scale weakly with memory technology.

So GB10's B_half (about 0.8 to 1.3 MiB) will be about 3 to 6x smaller than on an RTX 5000-class card with 3 to 6x the bandwidth. GB10 is therefore predicted to be much **less** tau-sensitive. That means its quantization speedups should come closer to the byte ratio, except on small ops such as k_proj and v_proj at 4-bit.

---

## Platform: `rtx6000ada`

Predicted by Claude, at P2's request, before any timing on this machine (E0 and selfcheck only, no E1 or E2 cell).
Date committed: 2026-10-22.

**Inputs used**
- RTX 6000 Ada Generation spec: 384-bit GDDR6 at 20 Gbps (NVML max memory clock 10001 MHz), so **960 GB/s**; 142 SMs;
  96 MB L2; max SM clock 3105 MHz, default application clock 2505 MHz; 300 W. SM count, L2 and bus width are checked
  against `env.json` before this commit.
- GB10 v2 sessions and the RTX PRO 5000 partial pilot (`findings.md`), used as priors.
- Stack: torch 2.11.0+cu128, cuBLAS 12.8.4.1 (host venv; driver 575 rules out the cu130 stacks of GB10 and P3).

**Not used:** no timing on this GPU exists yet.

### E1 (B0)

| Quantity | Prediction | Reasoning |
|---|---|---|
| Cold read plateau R_inf_B0 | **875 GB/s** (range 830 to 910) | The other discrete GDDR card (RTX PRO 5000) reached 92% of spec. GDDR6 has similar refresh and turnaround overheads to GDDR7, so about 91%. |
| R_inf_B0 as % of spec | **91%** (86 to 95%) | 875 / 960. |
| probe tau, Triton read, graph mode | **1.6 us** (1.2 to 2.5) | GB10 1.7 us, RTX PRO 5000 about 1.4 us. Graph-mode tau is a GPU front-end cost (launch gap, ramp, tail), and Ada's front end runs at a similar SM clock. 142 SMs means a slightly longer ramp to fill every SM. |
| probe tau, eager mode | **6.5 us** (5 to 8) | Eager is host-launch-bound at small sizes. The EPYC 9124 (Zen 4, 3.7 GHz max) has slower single-thread speed than the Core Ultra 9 285K (RTX PRO 5000 floor about 5.8 us), so the floor should be a little higher. |
| probe tau, isolated mode | **12 us** (10 to 15) | GB10 11.4 us: event and sync bracket overhead, dominated by host and driver round trips. |
| B_half (graph) | **1.3 MiB** (1.0 to 2.2) | 1.6 us x 875 GB/s = 1.40 MB = 1.34 MiB. About 3.4x GB10's 0.39 MiB, below the RTX PRO 5000's 1.61 MiB. |
| write (fill) vs read | **0.97x** | GDDR7 on the RTX PRO 5000 gave 1.00x; GDDR6 pure writes avoid bus turnaround, with a small write-allocate cost. |
| copy (2B counted) vs read | **0.90x** | RTX PRO 5000 0.89x: mixed read and write streams pay turnaround penalties on GDDR. |
| hot L2 peak (Triton read, graph) | **about 4 TB/s** (2.5 to 5.5) at 32 to 64 MiB | The L2 bandwidth of AD102 is several TB/s. At L2-resident sizes tau caps the observed rate: 32 MiB at 1.6 us tau plus 6.7 us at 5 TB/s gives about 4 TB/s. |
| hot falloff size | **drop between 64 and 128 MiB; hot equals cold by 256 MiB** | Nominal 96 MB L2; effective capacity is lower (hashing, two L2 partitions). Same shape as the RTX PRO 5000 (96 MiB L2, fell off by 128 MiB). |

### E2 (cuBLAS BF16)

| Quantity | Prediction | Reasoning |
|---|---|---|
| Kernel family at M=1 | **`gemvx` on every real shape** | GB10 and RTX PRO 5000 both dispatch `gemvx` at M=1. The heuristic in cuBLAS 12.8 for sm_89 may differ from 13.1, so confidence is moderate (about 80%). |
| v6 on the same shapes as GB10 | **Yes, mostly** (about 60% confidence): v6 on attention shapes and Qwen gate/up; v7 on Llama MLP and down_proj | The variant split looks tied to shape (K and N per CTA), but the cuBLAS version differs, so some shapes may switch variant. |
| v6 %B0 | **75%** (65 to 85%) | Little's law: the bytes in flight per SM needed for full speed are R_inf x latency / SMs, about 875 GB/s x 0.6 us / 142 = 3.7 KB here, against about 240 x 0.8 / 48 = 4.0 KB on GB10 (latencies are guesses). With 164 registers per thread and 64K registers per SM, occupancy is also 25% here (12 warps), so v6 should be about as short of bytes in flight as on GB10, slightly less. |
| v7 %B0 | **90 to 98%** | As on GB10. |
| lm_head %B0 | **80%** (70 to 95) | GB10 72 to 73% on a large-N v7 regime; its cause is unknown, so the prior is wide. |
| M sweep, t(M=2..16)/t(1) on v6 shapes | **0.80** (0.70 to 0.95) | If cuBLAS switches to an sm80 GEMM at M >= 2 that streams at about 90 to 95% of B0, the ratio is about 75/93. |
| M sweep on v7 shapes | **1.00 to 1.10** | As on GB10. |
| per-allocation spread (median / p90) | **0.15% / 0.6%** | Dedicated GDDR with no CPU traffic on the bus: the RTX PRO 5000 canary was 10x quieter than GB10's. Page-placement effects should be smaller than on LPDDR5x (GB10 0.45% / 1.3%). |
| Llama k_proj (8 MiB) efficiency, BF16 / at 4-bit size | **86% / 61%** | eff = B / (B + B_half) with B_half 1.3 MiB, using only the per-op cost. |

**Note added after drafting, before commit (2026-10-22):** the table above was written before `selfcheck.json` was read.
Selfcheck then showed that at M=1 Qwen gate_proj (18944x3584) dispatches `gemvx` v7 here (v6 on GB10), while Llama q/k_proj
are v6 as on GB10. The predictions were not changed; E2 will test them as written.

### E4 (library FP8 / INT4), `rtx6000ada`

Predicted by Claude before any E4 cell on this GPU (2026-10-25). Inputs: this platform's E1/E2 results of record
(B0 891.7 GB/s, probe tau 1.2 to 1.3 us, cuBLAS BF16 tau 2.45 us, BF16 per token 17.17 ms Llama / 16.20 ms Qwen in graph
mode with fused shapes) and GB10's gate-bypassed E4 debug hints (`findings.md`, not results). No Ada E4 number seen.
Stack for E4: torch 2.11.0+cu129, vLLM 0.26.0+cu129 (a separate venv; BF16 reference re-measured in the same venv).

| Quantity | Prediction | Reasoning |
|---|---|---|
| Large ops (gate/up/down, lm_head), %B0, graph M=1 | **FP8 per-tensor 80 to 90%, CUTLASS FP8 row-wise 85 to 95%, tinygemm 85 to 95%, Marlin 90 to 97%** | Tiled GEMMs stream well once there are many tiles; GB10 hints 83 to 98%. Ada's FP8 path is the native `sm89_xmma` kernel GB10 already used for large N. |
| Small ops (Qwen k/v 512 x 3584; Llama k/v 1024 x 4096), %B0 | **20 to 45% (FP8), 15 to 40% (INT4)** | At 0.9 to 4 MiB the per-op cost dominates. Library GEMMs at M=1 carry 5 to 10 us of fixed cost (tile setup, split-K or stream-K reduction, scale epilogue), against B/B0 of 1 to 4.5 us. GB10 hints: 24 to 65%. |
| Fitted per-op cost of the library kernels (graph) | **4 to 10 us**, i.e. 2 to 4x cuBLAS BF16 (2.45 us) and 3 to 8x the probe (1.25 us) | As above. This is the per-op gap a fused low-precision GEMV would close. |
| Measured speedup vs cuBLAS BF16, large ops | **FP8 1.7 to 1.9x (theory 2.0x); INT4 Marlin 3.2 to 3.6x (theory 3.88x); tinygemm 2.8 to 3.4x (theory 3.76x)** | Near the byte ratio when streaming dominates. |
| Measured speedup vs cuBLAS BF16, small ops | **0.6 to 1.3x for every library kernel** | The 4x byte saving is eaten by the larger fixed cost; some small shapes get slower than BF16. |
| Per-token GEMV time, best library kernel, graph, fused shapes | **FP8 about 9.5 ms Llama (1.8x, theory 2.0x); INT4 about 5.5 ms Llama (3.1x, theory 3.9x)** | Large ops dominate the bytes, so per-token efficiency is set by them, minus a per-op tax of 129 ops x (library tau - B0 time saving). |
| Gap to the 4-bit bandwidth limit per token (ideal 4.21 ms Llama) | **20 to 35%** | 129 GEMV ops x 5 to 10 us of fixed cost is 0.6 to 1.3 ms against 4.2 ms. If this holds, a fused low-precision GEMV with probe-like tau (4.37 ms) is worth 1.2 to 1.35x per token at 4-bit. |
| Dynamic activation quantization (`*_full`) | **per-tensor +8 to 14 us per op (3 extra kernels); per-token +3 to 6 us (1 kernel)** | GB10 hint +14 us per-tensor; Ada launches are cheaper in graph mode. |
| M sweep (graph), t(16)/t(1) | **1.0 to 1.15** | Tensor-core tiles at M = 1 already compute 16 or more rows; extra rows cost little until compute matters. |
| Conclusion to test | **A per-op gap exists for low precision on Ada even though BF16 has none** | If the small-op %B0 and the per-token gap come out near the BF16 numbers instead, there is no gap on this GPU and that will be reported. |
