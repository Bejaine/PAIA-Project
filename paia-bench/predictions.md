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
