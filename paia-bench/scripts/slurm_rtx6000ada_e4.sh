#!/usr/bin/env bash
# E4 on rtx6000ada: library FP8 / INT4 against cuBLAS BF16 on one card, in the E4 venv (torch 2.11.0+cu129, vLLM 0.26.0).
# The BF16 reference (E2) and B0 (E1) are re-measured in the same job, venv and card, so every speedup and %B0 compares
# like with like. Order: test, env+selfcheck, E1 s1, E2 s1, E4 s1, E2 s2, E4 s2, dispatch table, analysis.
# Same guards as slurm_rtx6000ada.sh: GPU model, ECC off, one card per session set, venv lock id. Gate unchanged.
#SBATCH -J paia-e4
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH -c 10
#SBATCH --mem=20G
#SBATCH -t 24:00:00
#SBATCH -o /home/bejaine.birju/paia/project/slurm-%j.out
set -uo pipefail
cd /home/bejaine.birju/paia/project/PAIA-Project/paia-bench
export PATH=/usr/local/cuda-12.9/bin:$PATH
export PAIA_VENV=/scratch/bejaine.birju/paia/.venv-e4
export PAIA_GPU="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)"
TAG=rtx6000ada; D=${PAIA_DATE:-$(date +%Y%m%d)}; R=scripts/hrun.sh
CONTACT="Bejaine <bejainebirju@gmail.com>"; LOCK_ID=9014db2581becb83
ENV_S=env_e4_${D}; E1=e1v2_s1_${D}; E2A=e2v2_s1_${D}; E4A=e4_s1_${D}; E2B=e2v2_s2_${D}; E4B=e4_s2_${D}
STEPS="${PAIA_STEPS:-test env e1 e2a e4a e2b e4b dispatch analyze}"
T=results/$TAG/dispatch_${D}/dispatch_table_e2_e4.json
mkdir -p results/$TAG/$ENV_S
( while true; do echo "=== $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"; ps -eo user,pid,pcpu,etime,cmd --sort=-pcpu --no-headers | head -6 | cut -c1-140
    nvidia-smi --query-gpu=uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu --format=csv,noheader; sleep 60; done
) > results/$TAG/$ENV_S/node_state_${SLURM_JOB_ID}.log 2>&1 &
trap "kill $! 2>/dev/null" EXIT
echo "job $SLURM_JOB_ID on $(hostname) GPU $PAIA_GPU cpus $(taskset -pc $$ | cut -d: -f2) steps: $STEPS"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,ecc.mode.current,ecc.mode.pending,persistence_mode --format=csv,noheader; uptime
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" == "NVIDIA RTX 6000 Ada Generation" ]] || { echo "ABORT: not an RTX 6000 Ada"; exit 2; }
CARD=results/$TAG/card_${D}.txt
[[ -f $CARD && "$(cat $CARD)" != "$PAIA_GPU" ]] && { echo "ABORT: sessions *_${D} use $(cat $CARD), got $PAIA_GPU"; exit 6; }
[[ "$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | head -1)" == "Disabled" ]] || { echo "ABORT: ECC enabled"; exit 3; }
[[ -x $PAIA_VENV/bin/python ]] || { echo "ABORT: no E4 venv on this node"; exit 4; }
id=$(sha1sum $PAIA_VENV/requirements.lock.txt | cut -c1-16); [[ "$id" == "$LOCK_ID" ]] || { echo "ABORT: venv lock $id != $LOCK_ID"; exit 5; }
[[ -f $CARD ]] || echo "$PAIA_GPU" > $CARD
run() { echo; echo "##### $(date -Is) $*"; "$@"; local rc=$?; echo "##### rc=$rc $(date -Is)"; return $rc; }
sess() { run $R python -m bench.run --phase $1 --session $2 --platform-tag $TAG --contact "$CONTACT" --no-tmux-check; }
for step in $STEPS; do
  case $step in
    test) run $R python -m tests.test_resume ;;
    env)  run $R python -m bench.env --session $ENV_S --platform-tag $TAG --probe-ncu
          run $R python -m bench.selfcheck --session $ENV_S --platform-tag $TAG ;;
    e1)   sess e1 $E1 ;;
    e2a)  sess e2 $E2A ;;
    e4a)  sess e4 $E4A ;;
    e2b)  sess e2 $E2B ;;
    e4b)  sess e4 $E4B ;;
    dispatch) mkdir -p $(dirname $T); run $R python scripts/diag/dispatch_table.py $T e2 e4 ;;
    analyze)
      run $R python -m bench.analyze --session $E1 --platform-tag $TAG
      run $R python -m bench.analyze --session $E2A --platform-tag $TAG --e1-session $E1 --dispatch-table $T
      run $R python -m bench.analyze --session $E2B --platform-tag $TAG --e1-session $E1 --dispatch-table $T --compare $E2A
      run $R python -m bench.analyze --session $E4A --platform-tag $TAG --e1-session $E1 --bf16-sessions $E2A --dispatch-table $T
      run $R python -m bench.analyze --session $E4B --platform-tag $TAG --e1-session $E1 --bf16-sessions $E2B --dispatch-table $T --compare $E4A ;;
  esac
done
echo "ALL_DONE $(date -Is)"
