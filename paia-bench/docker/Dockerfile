# Single image for every phase (E0 to E5).
# Base: NGC vLLM 25.11 (torch 2.10.0a0+b558c98.nv25.11, cuBLAS 13.1.0, Triton 3.5.0, CUDA 13.0, vLLM 0.11.0).
# Its torch/cuBLAS/Triton build is identical to nvcr.io/nvidia/pytorch:25.11-py3; we only add the
# profilers and analysis libraries that the vLLM image lacks, copied from or pinned to the PyTorch image.
# 25.11 is the newest NGC release built on CUDA 13.0, i.e. the newest that runs natively on driver 580.
ARG NGC_TAG=25.11
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}-py3 AS pt
FROM nvcr.io/nvidia/vllm:${NGC_TAG}-py3

COPY --from=pt /usr/local/cuda-13.0/NsightSystems-cli-2025.5.1 /opt/nvidia/nsys
COPY --from=pt /opt/nvidia/nsight-compute /opt/nvidia/nsight-compute
RUN ln -sf /opt/nvidia/nsys/bin/nsys /usr/local/bin/nsys \
 && ln -sf /opt/nvidia/nsight-compute/2025.3.1/ncu /usr/local/bin/ncu

# Same versions as the PyTorch image. /etc/pip/constraint.txt (NGC default) keeps torch and friends pinned.
RUN pip install --no-cache-dir pandas==2.3.3 matplotlib==3.10.7 nvtx==0.2.13

# Fail the build if the core stack drifted from what was validated.
RUN python -c "import torch, triton, vllm; \
assert torch.__version__.startswith('2.10.0a0+b558c98'), torch.__version__; \
assert triton.__version__.startswith('3.5.0'), triton.__version__; \
assert vllm.__version__.startswith('0.11.0'), vllm.__version__; \
print('stack ok', torch.__version__, triton.__version__, vllm.__version__)" \
 && nsys --version && ncu --version | tail -1

LABEL org.paia.purpose="PAIA library baselines (P2)" org.paia.ngc_tag="${NGC_TAG}"
