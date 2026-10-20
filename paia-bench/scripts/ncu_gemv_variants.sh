#!/usr/bin/env bash
# E3 (focused): why does cuBLAS's gemvx/v6 stream ~30% slower than gemvx/v7 on GB10?
# Profiles M=1 F.linear on two v6 shapes and two v7 shapes with Nsight Compute, clocks uncontrolled
# (--clock-control none) so they match the harness. ncu durations are NOT headline numbers.
# Usage (inside the image): scripts/ncu_gemv_variants.sh <out_dir>
set -euo pipefail
OUT="${1:-results/ncu_gemv_variants}"
mkdir -p "$OUT"
METRICS=gpu__time_duration.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_aperture_sysmem_op_write.sum,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,\
launch__grid_size,launch__block_size,launch__registers_per_thread,launch__waves_per_multiprocessor,\
launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
for shape in "4096 4096 llama_q_proj_v6" "18944 3584 qwen_gate_proj_v6" "14336 4096 llama_gate_proj_v7" "4096 14336 llama_down_proj_v7"; do
  set -- $shape
  N=$1 K=$2 ncu --clock-control none -k regex:^kernel$ -c 1 --launch-skip 3 --metrics "$METRICS" --csv \
    python scripts/diag/linear_once.py > "$OUT/$3.csv" 2> "$OUT/$3.err" || true
  echo "$3 done"
done
