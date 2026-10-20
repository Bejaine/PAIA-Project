"""Profiling target: a few M=1 F.linear calls on one BF16 weight (N, K from env). The weight exceeds L2 for the
shapes profiled, so the profiled call streams from DRAM. Used by scripts/ncu_gemv_variants.sh."""
import os
import torch
import torch.nn.functional as F
N, K = int(os.environ["N"]), int(os.environ["K"])
W = ((torch.rand(N, K, device="cuda") - 0.5) * 0.04).to(torch.bfloat16)
x = (torch.rand(1, K, device="cuda") - 0.5).to(torch.bfloat16)
for _ in range(6):
    F.linear(x, W)
torch.cuda.synchronize()
