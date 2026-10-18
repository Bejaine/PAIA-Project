"""Are 'cold' cells really cold? For a buffer size S, rotate through n = ceil(f * L2 / S) copies (f = 4, 8, 16)
reading each with torch.sum, exactly like the harness. Run under ncu; the last pass's kernels tell how many
bytes came from DRAM (lts__d_sectors_fill_sysmem x 32) vs the bytes read. Diagnostic tooling."""
import math, sys, torch
L2 = torch.cuda.get_device_properties(0).L2_cache_size
S = int(sys.argv[1]); f = int(sys.argv[2])
n = max(2, math.ceil(f * L2 / S))
bufs = [torch.rand(S // 4, device="cuda") for _ in range(n)]
for _ in range(3):                       # warm passes, as in warmup + rotation
    for b in bufs: b.sum()
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("measure")
for b in bufs: b.sum()                   # profiled pass
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print(f"S={S} f={f} n_copies={n}")
