#!/usr/bin/env bash
# Run a command inside the benchmark image.
#   --pid=host     : the monitor must see other users' processes and NVML's host PIDs
#   --user         : results are owned by the invoking user, not root
#   /etc/passwd    : torch.profiler needs a passwd entry for the uid
#   /tmp           : shared so the courtesy lockfile /tmp/paia_bench.lock is visible to everyone
#   PAIA_GPU       : on multi-GPU hosts, the one GPU to expose (nvidia-smi index or UUID); default all
# Usage: scripts/drun.sh [docker-run-options...] -- <cmd...>
#        scripts/drun.sh <cmd...>
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${PAIA_IMAGE:-paia-bench:25.11}"
EXTRA=()
if [[ " $* " == *" -- "* ]]; then
  while [[ "$1" != "--" ]]; do EXTRA+=("$1"); shift; done
  shift
fi
mkdir -p "$BENCH_DIR/.container_home"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
TTY=(); [[ -t 0 && -t 1 ]] && TTY=(-it)
# .git lives above the mounted dir, so resolve provenance on the host.
GIT_COMMIT="$(git -C "$BENCH_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$( [[ -n "$(git -C "$BENCH_DIR" status --porcelain -- . ':!results' 2>/dev/null)" ]] && echo 1 || echo 0)"
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || echo unknown)"
GPUS="all"; [[ -n "${PAIA_GPU:-}" ]] && GPUS="\"device=${PAIA_GPU}\""
exec docker run --rm "${TTY[@]}" --gpus "$GPUS" --pid=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$BENCH_DIR":/bench -v "$BENCH_DIR/../src":/src -v "$HF_CACHE":/hf -v /tmp:/tmp \
  -e HOME=/bench/.container_home -e HF_HOME=/hf -e PYTHONPATH=/bench:/src -e PAIA_GEMV_LIB=/src/build/$(uname -m)/libpaia_gemv.so \
  -e PAIA_IMAGE="$IMAGE" -e PAIA_IMAGE_ID="$IMAGE_ID" \
  -e PAIA_GIT_COMMIT="$GIT_COMMIT" -e PAIA_GIT_DIRTY="$GIT_DIRTY" -e PAIA_IN_TMUX="${TMUX:+1}" -e PAIA_HOST_USER="$(id -un)" -e PAIA_HOST="$(hostname)" \
  -w /bench "${EXTRA[@]}" "$IMAGE" "$@"
