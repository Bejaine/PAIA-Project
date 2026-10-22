#!/usr/bin/env bash
# Run a command on the host in the venv made by scripts/setup_venv.sh. Fallback for machines where the
# paia-bench image cannot run (driver < 580, no Docker access, or a GPU the image's torch was not built for).
# Sets the same provenance variables as scripts/drun.sh so env.json and cell records are complete.
#   PAIA_GPU : on multi-GPU hosts, the one GPU to use (nvidia-smi index or UUID). Required when there are several.
# Usage: scripts/hrun.sh <cmd...>     e.g. scripts/hrun.sh python -m bench.env --session s1 --platform-tag rtx5000ada
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PAIA_VENV:-$BENCH_DIR/.venv-host}"
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV; run scripts/setup_venv.sh first" >&2; exit 1; }
N_GPUS="$(nvidia-smi -L | wc -l)"
if [[ -n "${PAIA_GPU:-}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID   # so an index means the same GPU as in nvidia-smi
  export CUDA_VISIBLE_DEVICES="$PAIA_GPU"
elif [[ "$N_GPUS" -gt 1 && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "this host has $N_GPUS GPUs: set PAIA_GPU to the nvidia-smi index or UUID of yours" >&2; exit 1
fi
source "$VENV/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" PYTHONPATH="$BENCH_DIR:$BENCH_DIR/../src"
export PAIA_IMAGE="host-venv:$(basename "$VENV")"
export PAIA_IMAGE_ID="$(sha1sum "$VENV/requirements.lock.txt" 2>/dev/null | cut -c1-16 || echo unknown)"
export PAIA_GIT_COMMIT="$(git -C "$BENCH_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
export PAIA_GIT_DIRTY="$( [[ -n "$(git -C "$BENCH_DIR" status --porcelain -- . ':!results' 2>/dev/null)" ]] && echo 1 || echo 0)"
export PAIA_IN_TMUX="${TMUX:+1}" PAIA_HOST_USER="$(id -un)" PAIA_HOST="$(hostname)"
cd "$BENCH_DIR"
exec "$@"
