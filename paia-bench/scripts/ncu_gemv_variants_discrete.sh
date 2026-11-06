#!/usr/bin/env bash
# E3 (focused) on a discrete GPU: the GB10 script (scripts/ncu_gemv_variants.sh) with two changes only.
#  1. Memory metrics come from configs/platform_notes.yaml for the platform tag (dram__* on discrete GPUs) instead of
#     GB10's L2 sysmem-aperture counters. Every other metric is the same list as on GB10.
#  2. Shapes are arguments ("N K label"), chosen from this platform's E2 dispatch, not GB10's.
# Same settings as GB10: --clock-control none, default cache control (caches flushed before the profiled kernel, so the
# call streams from DRAM even when the weight fits in L2), launch-skip 3. ncu durations are NOT headline numbers.
# Usage: scripts/ncu_gemv_variants_discrete.sh <tag> <out_dir> "N K label" ["N K label" ...]
set -euo pipefail
TAG="$1"; OUT="$2"; shift 2
mkdir -p "$OUT"
MEM=$(python -c "
from bench import config
m = config.platform_notes('$TAG')['ncu_metrics']
print(','.join([m['dram_read_bytes'], m['dram_write_bytes'], m['mem_throughput_pct']]))")
echo "memory metrics for $TAG: $MEM"
METRICS=gpu__time_duration.sum,$MEM,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
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
for shape in "$@"; do
  set -- $shape
  N=$1 K=$2 ncu --clock-control none -k regex:^kernel$ -c 1 --launch-skip 3 --metrics "$METRICS" --csv \
    python scripts/diag/linear_once.py > "$OUT/$3.csv" 2> "$OUT/$3.err" || true
  echo "$3 done: $(grep -c '"kernel"' "$OUT/$3.csv" 2>/dev/null || echo 0) kernel rows"
done
