#!/usr/bin/env bash
# E3 (focused ncu) on rtx6000ada plus a DCGM on/off A/B. Needs P2 to pause DCGM profiling on the node, from this job:
#   srun --jobid=<this job> --overlap --cpu-bind=none dcgmi profile --pause      (and later --resume)
# The job waits until ncu can read counters (DCGM paused), profiles, times the A/B probes while paused, then waits for
# DCGM to be resumed (ncu fails again) and times the same probes with DCGM active.
#SBATCH -J paia-e3
#SBATCH -p u22
#SBATCH -w node01
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH -c 10
#SBATCH --mem=20G
#SBATCH -t 03:00:00
#SBATCH -o /home/bejaine.birju/paia/project/slurm-%j.out
set -uo pipefail
cd /home/bejaine.birju/paia/project/PAIA-Project/paia-bench
export PATH=/usr/local/cuda-12.9/bin:$PATH PAIA_VENV=/scratch/bejaine.birju/paia/.venv-host
export PAIA_GPU="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)"
TAG=rtx6000ada; OUT=results/$TAG/e3_ncu_gemv_variants_$(date +%Y%m%d); mkdir -p $OUT
echo "job $SLURM_JOB_ID on $(hostname) GPU $PAIA_GPU"; nvidia-smi --query-gpu=name,uuid,pci.bus_id,ecc.mode.current --format=csv,noheader
probe() { scripts/hrun.sh ncu --clock-control none -c 1 --metrics gpu__time_duration.sum python -c "import torch; a=torch.empty(1<<20,device='cuda'); a.fill_(1); torch.cuda.synchronize()" 2>&1 | grep -q "gpu__time_duration.sum"; }
echo "WAITING_FOR_DCGM_PAUSE $(date -Is)"
for i in $(seq 1 240); do probe && break; sleep 15; done
probe || { echo "ABORT: counters never became available"; exit 7; }
echo "COUNTERS_AVAILABLE $(date -Is)"
scripts/hrun.sh scripts/ncu_gemv_variants_discrete.sh $TAG $OUT \
  "4096 4096 llama_q_proj_v6" "3584 3584 qwen_q_proj_v6" \
  "14336 4096 llama_gate_proj_v7" "4096 14336 llama_down_proj_v7" "18944 3584 qwen_gate_proj_v7"
scripts/hrun.sh ncu --query-metrics 2>/dev/null | grep -E "^dram__bytes_(read|write) " > $OUT/query_metrics_dram.txt
scripts/hrun.sh python scripts/diag/dcgm_ab.py dcgm_paused $OUT/dcgm_ab_paused.json 2>&1 | grep -vE "Warning|warn_once"
echo "E3_DONE_PLEASE_RESUME_DCGM $(date -Is)"
for i in $(seq 1 240); do probe || break; sleep 15; done
if probe; then echo "DCGM was not resumed within 60 min; skipping the active half of the A/B"; exit 0; fi
echo "DCGM_ACTIVE_AGAIN $(date -Is)"
sleep 30
scripts/hrun.sh python scripts/diag/dcgm_ab.py dcgm_active $OUT/dcgm_ab_active.json $OUT/dcgm_ab_paused.json 2>&1 | grep -vE "Warning|warn_once"
echo "ALL_DONE $(date -Is)"
