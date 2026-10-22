"""Compile-check the device code without a GPU or nvcc: NVRTC -> PTX -> ptxas, and print registers per variant.

Needs: pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 (any 12.9.x), or a CUDA toolkit.
Usage: python src/tools/nvrtc_check.py [--arch sm_121 sm_120 sm_89] [--variants 1,2,4 ...]
Prints, per variant: registers/thread, spills, resident warps per SM (65536 regs, <=48 warps; a lookup for GB10 and
RTX PRO 5000), and in-flight bytes per SM = warps * R*U*512. This is a static estimate; ncu is the measurement.
"""
import argparse, ctypes, glob, itertools, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.normpath(os.path.join(HERE, "..", "paia_gemv"))


def find(pattern):
    for base in sys.path + ["/usr/local/lib/python3.12/dist-packages", "/usr/local/cuda"]:
        hits = glob.glob(os.path.join(base, pattern))
        if hits:
            return hits[0]
    for hit in glob.glob("/usr/**/" + pattern, recursive=True):
        return hit
    raise SystemExit(f"cannot find {pattern}")


def nvrtc_ptx(src, arch):
    lib = ctypes.CDLL(find("nvidia/cuda_nvrtc/lib/libnvrtc.so.12"))
    prog = ctypes.c_void_p()
    assert lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"k.cu", 0, None, None) == 0
    inc = os.path.dirname(find("nvidia/cuda_runtime/include/cuda_bf16.h"))
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17", f"-I{inc}".encode(), f"-I{KERNEL_DIR}".encode()]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        raise SystemExit(f"NVRTC failed ({arch}):\n{log.value.decode()}")
    lib.nvrtcGetPTXSize(prog, ctypes.byref(n))
    ptx = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetPTX(prog, ptx)
    return ptx.value.decode(), log.value.decode()


def variants():
    for R, U, S, H, W in itertools.product((1, 2, 4), (1, 2, 4, 8), (0, 1), (0, 1), (4, 8)):
        yield R, U, W, S, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["sm_121", "sm_120", "sm_89"])
    args = ap.parse_args()
    vs = list(variants())
    src = '#include "gemv_bf16.cuh"\n' + "".join(
        f"template __global__ void paia::gemv_bf16_kernel<{R},{U},{W},{'true' if S else 'false'},{'true' if H else 'false'}>"
        "(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int);\n" for R, U, W, S, H in vs)
    ptxas = find("nvidia/cuda_nvcc/bin/ptxas")
    for arch in args.arch:
        comp = "compute_" + arch[3:]
        try:
            ptx, log = nvrtc_ptx(src, comp)
        except SystemExit as e:
            print(f"[{arch}] NVRTC did not accept {comp}: {str(e)[:200]}"); continue
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.ptx"); open(p, "w").write(ptx)
            r = subprocess.run([ptxas, f"-arch={arch}", "-v", p, "-o", os.path.join(d, "k.cubin")],
                               capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{arch}] ptxas FAILED:\n{r.stderr[:2000]}"); continue
        regs, spills = {}, {}
        cur = None
        for line in r.stderr.splitlines():
            m = re.search(r"Compiling entry function '(.+?)' for", line)
            if m: cur = m.group(1); continue
            m = re.search(r"Used (\d+) registers", line)
            if m and cur: regs[cur] = int(m.group(1))
            m = re.search(r"(\d+) bytes spill stores", line)
            if m and cur: spills[cur] = int(m.group(1))
        print(f"\n[{arch}] ptxas OK, {len(regs)} kernels. warps/SM = min(48, 65536 // (regs*32)); in-flight/SM = warps*R*U*512 B")
        print(f"{'R':>2} {'U':>2} {'W':>2} {'S':>1} {'H':>1} {'regs':>5} {'need*':>5} {'spill':>5} {'warps/SM':>8} {'occ%':>5} {'inflight KB/SM':>14}")
        print('   (* need = 4*U*(R+1): registers to hold all R*U weight and U x loads at once; regs >= need suggests the loads batch)')
        for R, U, W, S, H in vs:
            name = next((k for k in regs if f"gemv_bf16_kernel" in k and f"ILi{R}ELi{U}ELi{W}ELb{S}ELb{H}E" in k), None)
            if not name: continue
            g = regs[name]
            gran = ((g + 7) // 8) * 8                       # register allocation granularity
            warps = min(48, 65536 // (gran * 32))
            # whole blocks only
            warps = (warps // W) * W
            flag = '*' if g < 4*U*(R+1) else ' '   # regs below need: ptxas is NOT holding all loads in flight, so the in-flight column is an upper bound
            print(f"{R:>2} {U:>2} {W:>2} {S:>1} {H:>1} {g:>5} {4*U*(R+1):>5} {spills.get(name,0):>5} {warps:>8} {100*warps/48:>5.0f} {warps*R*U*512/1024:>13.1f}{flag}")


if __name__ == "__main__":
    main()
