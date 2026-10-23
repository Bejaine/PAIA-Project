#!/usr/bin/env bash
# One SLURM job = one physical GPU for every session, so E1/E2 s1 and s2 are measured on the same card.
# Order: E0 env + selfcheck, monitor-perturbation diag, E1 s1, E2 s1, E1 s2, E2 s2, analysis.
# Runs under sbatch (no SSH session to lose), hence --no-tmux-check. The idle gate is unchanged.
# A node-state logger records load and our own uid's processes every 60 s, because the gate ignores our own uid.
#SBATCH -J paia-rtx6000ada
#SBATCH -p u22
#SBATCH -w node01
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=20G
#SBATCH -t 06:00:00
#SBATCH -o /home/bejaine.birju/paia/project/slurm-%j.out
set -uo pipefail
BENCH=/home/bejaine.birju/paia/project/PAIA-Project/paia-bench
cd "$BENCH"
export PATH=/usr/local/cuda-12.9/bin:$PATH
export PAIA_VENV=/scratch/bejaine.birju/paia/.venv-host
export PAIA_GPU="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)"
TAG=rtx6000ada
D=${PAIA_DATE:-$(date +%Y%m%d)}
CONTACT="Bejaine <bejainebirju@gmail.com>"
R=scripts/hrun.sh
ENV_S=env_${D}; E1S1=e1v2_s1_${D}; E2S1=e2v2_s1_${D}; E1S2=e1v2_s2_${D}; E2S2=e2v2_s2_${D}
STEPS="${PAIA_STEPS:-env diag e1s1 e2s1 e1s2 e2s2}"

mkdir -p results/$TAG/$ENV_S
NODELOG=results/$TAG/$ENV_S/node_state_${SLURM_JOB_ID}.log
(
  while true; do
    echo "=== $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
    ps -eo user,pid,pcpu,etime,cmd --sort=-pcpu --no-headers | head -6 | cut -c1-140
    nvidia-smi --query-gpu=uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu --format=csv,noheader
    sleep 60
  done
) > "$NODELOG" 2>&1 &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null' EXIT

echo "job $SLURM_JOB_ID on $(hostname) GPU $PAIA_GPU cpus $(taskset -pc $$ | cut -d: -f2) steps: $STEPS"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,ecc.mode.current,ecc.mode.pending,persistence_mode,pstate --format=csv
uptime

run() { echo; echo "##### $(date -Is) $*"; "$@"; echo "##### rc=$? $(date -Is)"; }
for step in $STEPS; do
  case $step in
    env)  run $R python -m bench.env --session $ENV_S --platform-tag $TAG --probe-ncu
          run $R python -m bench.selfcheck --session $ENV_S --platform-tag $TAG ;;
    diag) run $R python scripts/diag/monitor_perturbation.py results/$TAG/$ENV_S/diag_monitor_perturbation.json ;;
    e1s1) run $R python -m bench.run --phase e1 --session $E1S1 --platform-tag $TAG --contact "$CONTACT" --no-tmux-check
          run $R python -m bench.analyze --session $E1S1 --platform-tag $TAG ;;
    e2s1) run $R python -m bench.run --phase e2 --session $E2S1 --platform-tag $TAG --contact "$CONTACT" --no-tmux-check
          run $R python -m bench.analyze --session $E2S1 --platform-tag $TAG --e1-session $E1S1 ;;
    e1s2) run $R python -m bench.run --phase e1 --session $E1S2 --platform-tag $TAG --contact "$CONTACT" --no-tmux-check
          run $R python -m bench.analyze --session $E1S2 --platform-tag $TAG --compare $E1S1 ;;
    e2s2) run $R python -m bench.run --phase e2 --session $E2S2 --platform-tag $TAG --contact "$CONTACT" --no-tmux-check
          run $R python -m bench.analyze --session $E2S2 --platform-tag $TAG --e1-session $E1S1 --compare $E2S1 ;;
  esac
done
echo "ALL_DONE $(date -Is)"
