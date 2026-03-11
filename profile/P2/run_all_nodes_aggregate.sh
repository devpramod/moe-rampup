#!/usr/bin/env bash
set -euo pipefail

STREAM_BIN=${STREAM_BIN:-/tmp/stream}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common_phys_cores.sh"

p2_require_stream_bin "$STREAM_BIN"

PHYS_CPU_LIST=${PHYS_CPU_LIST:-$(p2_get_physical_cpu_list)}
MAX_THREADS=$(p2_count_cpus_in_list "$PHYS_CPU_LIST")
THREADS=${THREADS:-$MAX_THREADS}

if (( THREADS > MAX_THREADS )); then
  echo "Requested THREADS=$THREADS exceeds physical core count ($MAX_THREADS). Capping to $MAX_THREADS."
  THREADS=$MAX_THREADS
fi

RUN_CPU_LIST=$(p2_first_n_cpus "$PHYS_CPU_LIST" "$THREADS")

echo "[P2] All-node aggregate bandwidth"
echo "STREAM_BIN=$STREAM_BIN THREADS=$THREADS"
echo "RUN_CPU_LIST=$RUN_CPU_LIST"
taskset -c "$RUN_CPU_LIST" \
  numactl --interleave=all \
  env OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores "$STREAM_BIN" \
  | egrep 'Copy:|Scale:|Add:|Triad:'
