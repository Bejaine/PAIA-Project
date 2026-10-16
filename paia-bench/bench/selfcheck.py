"""Functional checks for every library path the benchmarks use: correctness vs a reference, kernel dispatch,
and CUDA-graph capturability. Produces no timing numbers.

python -m bench.selfcheck --session S [--platform-tag T] [--only a,b]
Writes results/<platform_tag>/<session>/selfcheck.json. Run once per platform before any measurement.
"""
import argparse
import traceback

import torch
import torch.nn.functional as F

from bench.common import default_platform_tag, provenance, session_dir, write_json_atomic

DEV = "cuda"
REL_TOL = 3e-2  # bf16/fp8/int4 outputs vs fp32 reference of the same (de)quantized operands


def rel_err(y, ref):
    return ((y.float() - ref).abs().max() / ref.abs().max()).item()


def kernel_names(fn, n=2):
    from torch.profiler import ProfilerActivity, profile
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})


def graph_capturable(fn, reps=4):
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        g.replay(); torch.cuda.synchronize()
        return True
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:300]


def case(fn, ref):
    """Run fn, compare against ref, record dispatch and graph capture. Correctness is checked first:
    a kernel that silently returns garbage must never be timed."""
    y = fn(); torch.cuda.synchronize()
    err = rel_err(y, ref)
    return {"rel_err": err, "correct": bool(err < REL_TOL), "kernels": kernel_names(fn), "graph": graph_capturable(fn)}


def chk_bf16_cpu():
    torch.manual_seed(0)
    a, b = torch.randn(256, 512) * 0.1, torch.randn(512, 384) * 0.1
    got = (a.to(DEV, torch.bfloat16) @ b.to(DEV, torch.bfloat16)).float().cpu()
    err = rel_err(got, a @ b)
    return {"rel_err": err, "correct": err < REL_TOL}


def chk_e1_ops():
    a = torch.ones(16 << 20, dtype=torch.uint8, device=DEV)
    b = torch.empty_like(a)
    af = a.view(torch.float32)
    return {name: {"kernels": kernel_names(fn), "graph": graph_capturable(fn)}
            for name, fn in [("read_sum", lambda: af.sum()), ("copy", lambda: b.copy_(a)), ("write_fill", lambda: b.fill_(0))]}


def chk_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def read_k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.atomic_add(out_ptr, tl.sum(tl.load(x_ptr + offs, mask=offs < n, other=0.0), axis=0))

    n = 1 << 22
    x = torch.ones(n, device=DEV)
    o = torch.zeros(1, device=DEV)
    read_k[(triton.cdiv(n, 4096),)](x, o, n, BLOCK=4096); torch.cuda.synchronize()
    return {"triton": triton.__version__, "target": str(triton.runtime.driver.active.get_current_target()),
            "correct": o.item() == n, "graph": graph_capturable(lambda: read_k[(triton.cdiv(n, 4096),)](x, o, n, BLOCK=4096))}


def chk_cublas_bf16():
    out = {}
    for N, K in [(1024, 4096), (4096, 4096), (14336, 4096), (4096, 14336), (512, 3584), (18944, 3584)]:
        W = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02
        for M in (1, 16):
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
            out[f"{N}x{K}_M{M}"] = case(lambda: F.linear(x, W), x.float() @ W.float().t())
    return out


def _fp8_operands(N, K, M, rowwise):
    W = torch.randn(N, K, device=DEV) * 0.02
    x = torch.randn(M, K, device=DEV)
    if rowwise:
        sw = (W.abs().amax(1, keepdim=True) / 448.0).float()   # [N,1]
        sx = (x.abs().amax(1, keepdim=True) / 448.0).float()   # [M,1]
    else:
        sw = (W.abs().max() / 448.0).float()
        sx = (x.abs().max() / 448.0).float()
    Wq, xq = (W / sw).to(torch.float8_e4m3fn), (x / sx).to(torch.float8_e4m3fn)
    ref = (xq.float() * sx) @ (Wq.float() * sw).t()
    return Wq, xq, sw, sx, ref


def chk_scaled_mm_fp8():
    out = {"schema": str(torch.ops.aten._scaled_mm.default._schema)}
    for M in (1, 2, 4, 8, 16):
        Wq, xq, sw, sx, ref = _fp8_operands(4096, 4096, M, rowwise=False)
        out[f"tensorwise_M{M}"] = try_case(lambda: torch._scaled_mm(xq, Wq.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16), ref)
    for M in (1, 16):
        Wq, xq, sw, sx, ref = _fp8_operands(4096, 4096, M, rowwise=True)
        out[f"rowwise_M{M}"] = try_case(lambda: torch._scaled_mm(xq, Wq.t(), scale_a=sx, scale_b=sw.t(), out_dtype=torch.bfloat16), ref)
    return out


def chk_vllm_cutlass_fp8():
    from vllm import _custom_ops as ops
    cc = torch.cuda.get_device_capability()
    out = {"supports_fp8": ops.cutlass_scaled_mm_supports_fp8(cc[0] * 10 + cc[1])}
    for M in (1, 16):
        Wq, xq, sw, sx, ref = _fp8_operands(4096, 4096, M, rowwise=True)
        out[f"rowwise_M{M}"] = try_case(lambda: ops.cutlass_scaled_mm(xq, Wq.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16), ref)
    return out


def chk_tinygemm_int4():
    aten = torch.ops.aten
    out = {"convert_schema": [str(getattr(aten._convert_weight_to_int4pack, o)._schema) for o in aten._convert_weight_to_int4pack.overloads()],
           "mm_schema": [str(getattr(aten._weight_int4pack_mm, o)._schema) for o in aten._weight_int4pack_mm.overloads()]}
    N, K, G = 4096, 4096, 128
    W = torch.randn(N, K, device=DEV) * 0.02
    Wg = W.view(N, K // G, G)
    mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    scale = (mx - mn).clamp(min=1e-6) / 15.0
    zero = mn + 8 * scale  # tinygemm convention: w = (q - 8) * scale + zero
    q = ((Wg - mn) / scale).round().clamp(0, 15).to(torch.int32).view(N, K)
    deq = ((q.view(N, K // G, G).float() - 8) * scale + zero).view(N, K)
    packed = aten._convert_weight_to_int4pack((q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8).contiguous(), 8)
    sz = torch.stack([scale.squeeze(-1), zero.squeeze(-1)], -1).transpose(0, 1).contiguous().to(torch.bfloat16)
    out["packed"] = [list(packed.shape), str(packed.dtype)]
    out["scales_zeros"] = [list(sz.shape), str(sz.dtype)]
    for M in (1, 16):
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        out[f"M{M}"] = case(lambda: aten._weight_int4pack_mm(x, packed, G, sz), x.float() @ deq.t())
    return out


def chk_marlin_int4():
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types
    cc = torch.cuda.get_device_capability()
    out = {"supported": mu.check_marlin_supported(scalar_types.uint4b8, 128, has_zp=False, device_capability=cc[0] * 10 + cc[1])}
    dev = torch.device(DEV)
    for N, K in [(4096, 4096), (1024, 4096), (4096, 14336), (512, 3584)]:
        W = torch.randn(K, N, device=DEV, dtype=torch.bfloat16) * 0.02  # marlin_quantize takes [K, N]
        w_ref, q_w, s, g_idx, sort_idx, _ = marlin_quantize(W, scalar_types.uint4b8, 128, act_order=False)
        ws = mu.marlin_make_workspace_new(dev)
        for M in (1, 16):
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
            fn = lambda: mu.apply_gptq_marlin_linear(x, q_w, s, mu.marlin_make_empty_g_idx(dev), g_idx, sort_idx, ws,
                                                     scalar_types.uint4b8, N, K, is_k_full=True)
            out[f"{N}x{K}_M{M}"] = case(fn, x.float() @ w_ref.float())
    return out


def try_case(fn, ref):
    try:
        return case(fn, ref)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:400], "correct": False}


CHECKS = {
    "bf16_vs_cpu": chk_bf16_cpu,
    "e1_torch_ops": chk_e1_ops,
    "triton_read_probe": chk_triton,
    "cublas_bf16": chk_cublas_bf16,
    "scaled_mm_fp8": chk_scaled_mm_fp8,
    "vllm_cutlass_fp8": chk_vllm_cutlass_fp8,
    "tinygemm_int4": chk_tinygemm_int4,
    "marlin_int4": chk_marlin_int4,
}


def summarize(v):
    if not isinstance(v, dict):
        return v
    if "correct" in v or "error" in v:
        return {k: v[k] for k in ("correct", "graph", "error") if k in v}
    return {k: summarize(x) for k, x in v.items() if isinstance(x, dict)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--platform-tag", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    tag = a.platform_tag or default_platform_tag()
    names = a.only.split(",") if a.only else list(CHECKS)
    res = {"platform_tag": tag, "session_id": a.session, "provenance": provenance(),
           "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(), "checks": {}}
    for n in names:
        try:
            res["checks"][n] = {"ok": True, "value": CHECKS[n]()}
        except Exception:
            res["checks"][n] = {"ok": False, "error": traceback.format_exc()[-2000:]}
        print(f"[{n}] {summarize(res['checks'][n].get('value', res['checks'][n]))}", flush=True)
    out = session_dir(tag, a.session) / "selfcheck.json"
    write_json_atomic(out, res)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
