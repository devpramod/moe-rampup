#!/usr/bin/env bash
set -euo pipefail

STREAM_BIN=${STREAM_BIN:-/tmp/stream}
CPU_NODE=${CPU_NODE:-0}
MEM_NODE=${MEM_NODE:-1}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common_phys_cores.sh"

p2_require_stream_bin "$STREAM_BIN"

PHYS_CPU_LIST=${PHYS_CPU_LIST:-$(p2_get_physical_cpu_list "$CPU_NODE")}
MAX_THREADS=$(p2_count_cpus_in_list "$PHYS_CPU_LIST")
THREADS=${THREADS:-$MAX_THREADS}

if (( THREADS > MAX_THREADS )); then
  echo "Requested THREADS=$THREADS exceeds physical cores on CPU_NODE=$CPU_NODE ($MAX_THREADS). Capping to $MAX_THREADS."
  THREADS=$MAX_THREADS
fi

RUN_CPU_LIST=$(p2_first_n_cpus "$PHYS_CPU_LIST" "$THREADS")

echo "[P2] Cross-NUMA run (remote memory)"
echo "STREAM_BIN=$STREAM_BIN THREADS=$THREADS CPU_NODE=$CPU_NODE MEM_NODE=$MEM_NODE"
echo "RUN_CPU_LIST=$RUN_CPU_LIST"
taskset -c "$RUN_CPU_LIST" \
  numactl --membind="$MEM_NODE" \
  env OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores "$STREAM_BIN" \
  | egrep 'Copy:|Scale:|Add:|Triad:'
