#!/usr/bin/env bash
# Build the host venv used by scripts/hrun.sh. No sudo, nothing outside paia-bench/ (plus pip's user cache).
# The torch CUDA flavor follows the driver: >= 580 -> cu130, >= 570 -> cu128, else cu126 (no Blackwell).
# Override with PAIA_TORCH_INDEX=https://download.pytorch.org/whl/cuXXX. Add --vllm to also try vLLM (E4 Marlin, E5).
# The exact package set is frozen to $VENV/requirements.lock.txt; reuse the same venv for every session.
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PAIA_VENV:-$BENCH_DIR/.venv-host}"
PY="${PAIA_PYTHON:-python3}"
WITH_VLLM=0; [[ "${1:-}" == "--vllm" ]] && WITH_VLLM=1

drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
major="${drv%%.*}"
if [[ -z "${PAIA_TORCH_INDEX:-}" ]]; then
  if   (( major >= 580 )); then PAIA_TORCH_INDEX=https://download.pytorch.org/whl/cu130
  elif (( major >= 570 )); then PAIA_TORCH_INDEX=https://download.pytorch.org/whl/cu128
  else                          PAIA_TORCH_INDEX=https://download.pytorch.org/whl/cu126
  fi
fi
echo "driver $drv -> torch index $PAIA_TORCH_INDEX; venv $VENV"

[[ -x "$VENV/bin/python" ]] || "$PY" -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch --index-url "$PAIA_TORCH_INDEX"
pip install nvidia-ml-py psutil numpy scipy pandas matplotlib pyyaml nvtx transformers huggingface_hub
if (( WITH_VLLM )); then
  # vLLM pins its own torch; if this replaces the torch above, the arch check below still has to pass.
  pip install vllm || echo "vLLM install failed: record it in findings.md and continue without E4 Marlin / vLLM E5"
fi
pip freeze > "$VENV/requirements.lock.txt"

# The wheel must contain SASS (or compatible PTX) for this GPU; otherwise every kernel fails at launch.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${PAIA_GPU:-${CUDA_VISIBLE_DEVICES:-0}}" python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch sees no GPU (driver too old for this wheel?)"
cc = torch.cuda.get_device_capability()
arch = torch.cuda.get_arch_list()
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0), "cc", cc)
print("arch list", arch)
# SASS runs on any GPU of the same major version with an equal or higher minor (sm_86 code runs on cc 8.9).
sass_ok = any(a.startswith("sm_") and len(a) == 5 and a[3] == str(cc[0]) and int(a[4]) <= cc[1] for a in arch)
assert f"sm_{cc[0]}{cc[1]}" in arch or sass_ok or any(a.startswith("compute_") and int(a[8:]) <= cc[0] * 10 + cc[1] for a in arch), \
    f"this torch build has no code for sm_{cc[0]}{cc[1]}; pick another PAIA_TORCH_INDEX"
x = torch.randn(1024, 1024, device="cuda")
print("matmul ok", float((x @ x).sum()) == float((x @ x).sum()))
import triton; print("triton", triton.__version__)
EOF
echo "venv ready: scripts/hrun.sh python -m bench.env --session <s> --platform-tag <tag> --probe-ncu"
