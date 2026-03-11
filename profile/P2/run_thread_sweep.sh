#!/usr/bin/env bash
set -euo pipefail

STREAM_BIN=${STREAM_BIN:-/tmp/stream}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common_phys_cores.sh"

p2_require_stream_bin "$STREAM_BIN"

PHYS_CPU_LIST=${PHYS_CPU_LIST:-$(p2_get_physical_cpu_list)}
MAX_THREADS=$(p2_count_cpus_in_list "$PHYS_CPU_LIST")
THREAD_LIST=${THREAD_LIST:-$(p2_default_thread_list "$MAX_THREADS")}

echo "threads triad_mb_per_s"
for t in $THREAD_LIST; do
  if (( t > MAX_THREADS )); then
    continue
  fi
  RUN_CPU_LIST=$(p2_first_n_cpus "$PHYS_CPU_LIST" "$t")
  bw=$(taskset -c "$RUN_CPU_LIST" numactl --interleave=all env OMP_NUM_THREADS="$t" OMP_PROC_BIND=close OMP_PLACES=cores "$STREAM_BIN" | awk '/Triad:/{print $2}')
  echo "$t $bw"
done
