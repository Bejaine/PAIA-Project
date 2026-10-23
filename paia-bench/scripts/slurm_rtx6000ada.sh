#!/usr/bin/env bash
# One SLURM job = one physical GPU for every session, so E1/E2 s1 and s2 are measured on the same card.
# Order: venv (identical rebuild if missing on this node), resume test, E0 env + selfcheck, monitor-perturbation diag,
# E1 s1, E2 s1, E1 s2, E2 s2, analysis. Submit with -w <node> (the quietest node at submit time).
# Aborts if the card has ECC enabled (the sessions of record use ECC off) or the venv's lock hash differs.
# Runs under sbatch (no SSH session to lose), hence --no-tmux-check. The idle gate is unchanged.
# A node-state logger records load and our own uid's processes every 60 s, because the gate ignores our own uid.
#SBATCH -J paia-rtx6000ada
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
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
STEPS="${PAIA_STEPS:-venv test env diag e1s1 e2s1 e1s2 e2s2}"
REBUILD=/home/bejaine.birju/paia/project/venv_rebuild
LOCK_ID=ca97c3a32dcc3d75   # sha1 prefix of the node01 venv lock file (env_20261023 image_id)

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
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" != "NVIDIA RTX 6000 Ada Generation" ]]; then
  echo "ABORT: not an RTX 6000 Ada"; exit 2
fi
# Every session of this set must run on one card: record it on first use, refuse any other card later (resume after timeout).
CARD=results/$TAG/card_${D}.txt
if [[ -f $CARD && "$(cat $CARD)" != "$PAIA_GPU" ]]; then
  echo "ABORT: sessions *_${D} were started on $(cat $CARD), this job got $PAIA_GPU"; exit 6
fi
if [[ "$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | head -1)" != "Disabled" ]]; then
  echo "ABORT: ECC is enabled on this card; sessions of record use ECC off"; exit 3
fi
[[ -f $CARD ]] || echo "$PAIA_GPU" > $CARD

run() { echo; echo "##### $(date -Is) $*"; "$@"; local rc=$?; echo "##### rc=$rc $(date -Is)"; return $rc; }
for step in $STEPS; do
  case $step in
    venv) if [[ ! -x $PAIA_VENV/bin/python ]]; then
            run bash $REBUILD/fetch_wheels.sh || { echo "ABORT: wheel fetch failed"; exit 4; }
            run python3 -m venv $PAIA_VENV
            run $PAIA_VENV/bin/pip install --no-deps -r $REBUILD/requirements.lock.txt
            run bash scripts/setup_venv.sh
          fi
          id=$(sha1sum $PAIA_VENV/requirements.lock.txt | cut -c1-16); echo "venv lock id $id (want $LOCK_ID)"
          [[ "$id" == "$LOCK_ID" ]] || { echo "ABORT: venv differs from the node01 stack"; diff $REBUILD/requirements.lock.txt $PAIA_VENV/requirements.lock.txt; exit 5; } ;;
    test) run $R python -m tests.test_resume ;;
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
