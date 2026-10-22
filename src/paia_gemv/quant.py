"""Quantizers and reference dequantizers (torch, run on the GPU). Format definitions are in quant_host.h and must stay in
sync with it:  FP8 W8A16 = e4m3fn weights with one fp32 scale per output row (scale = amax/448);  INT4 W4A16 = unsigned
nibbles with zero point 8, group 128 along K, fp16 scale = fp16(amax/7), n = clamp(rint(w/scale), -7, 7) + 8, byte j =
weight 2j in the low nibble and 2j+1 in the high nibble."""
import torch


def quantize_fp8_rowwise(W):
    """W [N,K] float32 (cuda) -> (q uint8 [N,K], scale float32 [N])."""
    amax = W.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax)).float()
    q = (W / scale[:, None]).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(torch.uint8), scale


def dequant_fp8(q, scale):
    return q.view(torch.float8_e4m3fn).float() * scale[:, None]


def quantize_int4_g128(W):
    """W [N,K] float32 (cuda), K % 128 == 0 -> (packed uint8 [N,K/2], scales float16 [N,K/128])."""
    N, K = W.shape
    assert K % 128 == 0
    g = W.view(N, K // 128, 128)
    amax = g.abs().amax(dim=-1)
    s16 = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax)).to(torch.float16)
    sf = s16.float()
    sf = torch.where(sf > 0, sf, torch.ones_like(sf))
    n = (torch.round(g / sf[..., None]).clamp_(-7, 7) + 8).to(torch.int32).view(N, K)
    packed = (n[:, 0::2] | (n[:, 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), s16.contiguous()


def dequant_int4(packed, s16):
    N, half = packed.shape
    p = packed.to(torch.int32)
    n = torch.stack([p & 0xF, p >> 4], dim=-1).view(N, half * 2).float()
    return (n - 8.0) * s16.float().repeat_interleave(128, dim=1)
