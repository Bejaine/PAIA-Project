#!/usr/bin/env bash
# Build libpaia_gemv.so (static cudart, C ABI). Run where nvcc exists: inside paia-bench:25.11 (CUDA 13.0) on the
# Spark, or on turing with CUDA 12.9 (~/cuda-12.9). Output: src/build/<arch>/libpaia_gemv.so (git-ignored).
# GB10 is aarch64 and the RTX server is x86_64, so build once per CPU architecture; the .so does not depend on the
# torch version (no torch headers, static cudart).
# [check] that this nvcc accepts sm_121; if not, drop it and rely on compute_120 PTX (sm_121 is reported to run sm_120 code).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-nvcc}"
OUT="${OUT:-$HERE/../build/$(uname -m)}"
GENCODE="${GENCODE:--gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121 -gencode arch=compute_120,code=compute_120}"
mkdir -p "$OUT"
if ! $NVCC -O3 -std=c++17 -lineinfo -Xcompiler -fPIC -shared -cudart static $GENCODE \
  -Xptxas -v "$HERE/gemv_bf16.cu" -o "$OUT/libpaia_gemv.so" 2> "$OUT/ptxas_report.txt"; then
  echo "BUILD FAILED. First 40 lines of the compiler output:" >&2
  head -40 "$OUT/ptxas_report.txt" >&2
  exit 1
fi
echo "built $OUT/libpaia_gemv.so (register report: $OUT/ptxas_report.txt)"
