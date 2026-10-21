#!/usr/bin/env bash
# First-login survey of a new benchmark host. Read-only: no sudo, no installs, no GPU work (no CUDA context).
# Answers: which GPU variant is this, can the paia-bench image run here, which GPU is ours, who else uses
# the machine, can we read perf counters, is there room for models and images.
# Usage: scripts/preflight.sh [out_file]     (default: results/preflight/<host>_<timestamp>.txt)
set -uo pipefail
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$BENCH_DIR/results/preflight/$(hostname -s)_$(date +%Y%m%d_%H%M%S).txt}"
mkdir -p "$(dirname "$OUT")"
exec > >(tee "$OUT") 2>&1

sec() { printf '\n===== %s =====\n' "$*"; }
run() { printf '$ %s\n' "$*"; timeout 60 bash -c "$*" 2>&1 | sed 's/^/  /'; }
have() { command -v "$1" >/dev/null 2>&1; }
NOTES=()
note() { NOTES+=("$*"); }

sec "host"
run 'date -Is; hostname; uname -a; cat /etc/os-release | head -4'
run 'id; groups'

sec "GPUs"
run 'nvidia-smi -L'
run 'nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,memory.total,driver_version,ecc.mode.current,ecc.mode.pending,persistence_mode,mig.mode.current,display_active,power.limit,clocks.max.sm,clocks.max.mem --format=csv'
run 'nvidia-smi topo -m'
run 'nvidia-smi -q -d CLOCK | grep -A3 -E "Applications Clocks|Max Clocks" | head -20'

sec "who is using the GPUs right now"
run 'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv'
run 'nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv'
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  run "ps -o user=,pid=,etime=,pcpu=,cmd= -p $pid | cut -c1-160"
done
run 'pgrep -a Xorg; pgrep -a gnome-shell'

sec "CPU and other users"
run 'lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)|max MHz"'
run 'cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq 2>/dev/null | sort | uniq -c'
run 'uptime; who | awk "{print \$1}" | sort | uniq -c'
run "ps -eo user,pcpu --no-headers | awk '\$1!=\"$(id -un)\"{c[\$1]+=\$2} END{for(u in c) if(c[u]>5) printf \"%-16s %6.1f%% (~%.1f cores)\n\",u,c[u],c[u]/100}' | sort -k2 -nr | head"
run 'free -g'

sec "driver and image compatibility"
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
ccs="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')"
echo "  driver $drv, compute capabilities: $ccs"
if [[ -n "$drv" ]] && (( ${drv%%.*} >= 580 )); then
  note "driver $drv >= 580: the paia-bench:25.11 image (CUDA 13.0) can run natively"
else
  note "driver $drv < 580: the 25.11 image will not run natively; use scripts/setup_venv.sh + scripts/hrun.sh (or ask P2 for an older NGC tag)"
fi
for cc in $ccs; do
  case "$cc" in
    7.5) note "cc 7.5 (Turing, Quadro RTX 5000): NGC 25.11 torch has no sm_75 build -> venv path; no BF16 tensor cores (FP16 baseline), no FP8, no tinygemm INT4; 16 GB too small for BF16 8B E5" ;;
    8.9) note "cc 8.9 (RTX 5000 Ada / RTX 6000 Ada): FP8 yes, FP4 no; record ECC state (ECC costs bandwidth)" ;;
    12.0) note "cc 12.0 (RTX PRO 5000 Blackwell): needs CUDA 12.8+ torch; highest bandwidth, strongest tau effect expected" ;;
  esac
done

sec "containers"
if have docker; then
  run 'docker version --format "client {{.Client.Version}} server {{.Server.Version}}"'
  run 'docker info --format "runtimes: {{json .Runtimes}} default: {{.DefaultRuntime}} root: {{.DockerRootDir}}"'
  run 'docker image ls --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep -E "paia|nvcr|pytorch|vllm" | head'
  if docker info >/dev/null 2>&1; then note "docker usable without sudo"; else note "docker present but not usable by $(id -un) (not in docker group?): ask admin, or use the venv path"; fi
else
  note "no docker: use the venv path (scripts/setup_venv.sh)"
fi
have podman && run 'podman --version'
have nvidia-ctk && run 'nvidia-ctk --version | head -1'
have enroot && run 'enroot version'
have apptainer && run 'apptainer --version'
have singularity && run 'singularity --version'

sec "profilers and toolchain"
for t in ncu nsys nvcc tmux git python3; do
  if have "$t"; then echo "  $t: $(command -v $t)"; else echo "  $t: not in PATH"; fi
done
run 'ls -d /usr/local/cuda* /opt/nvidia/nsight-* 2>/dev/null'
run 'ncu --version 2>/dev/null | tail -1; nsys --version 2>/dev/null'
run 'python3 --version; python3 -c "import torch; print(\"host torch\", torch.__version__, torch.version.cuda)" 2>&1 | tail -1'
run 'grep RmProfilingAdminOnly /proc/driver/nvidia/params'
if grep -q 'RmProfilingAdminOnly: 0' /proc/driver/nvidia/params 2>/dev/null; then
  note "perf counters open to non-root: ncu (E3) should work"
else
  note "perf counters admin-only (or unreadable): E3 needs the admin to set NVreg_RestrictProfilingToAdminUsers=0, or record it as the blocker"
fi
have tmux || note "no tmux: real sessions refuse to run outside tmux; ask admin or build tmux in \$HOME"

sec "disk"
run "df -h $HOME /tmp $BENCH_DIR 2>/dev/null"
run 'du -sh ${HF_HOME:-$HOME/.cache/huggingface} 2>/dev/null; ls ${HF_HOME:-$HOME/.cache/huggingface}/hub 2>/dev/null | head'
avail_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
(( avail_gb < 60 )) && note "only ${avail_gb} GB free in \$HOME: two models (~31 GB) + the image (~30 GB) may not fit; set HF_HOME to a bigger disk"

sec "network (egress needed for image, wheels, models)"
for u in https://nvcr.io/v2/ https://download.pytorch.org/whl/ https://pypi.org/simple/ https://huggingface.co https://github.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$u"); echo "  $u -> ${code:-fail}"
done

sec "shared-machine courtesy"
run 'ls -l /tmp/paia_bench.lock 2>/dev/null && cat /tmp/paia_bench.lock'

sec "summary"
for n in "${NOTES[@]}"; do echo "  - $n"; done
name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$name" in
  *"RTX 6000 Ada"*)            tag=rtx6000ada ;;
  *"RTX PRO 6000"*)            tag=rtxpro6000bw ;;
  *"Quadro RTX 6000"*)         tag=quadrortx6000 ;;
  *"RTX 5000 Ada"*)            tag=rtx5000ada ;;
  *"RTX PRO 5000"*|*Blackwell*) tag=rtxpro5000bw ;;
  *"Quadro RTX 5000"*)         tag=quadrortx5000 ;;
  *) tag="(unknown: $name; agree a tag with P2)" ;;
esac
echo "  - suggested --platform-tag: $tag"
echo
echo "report written to $OUT"
