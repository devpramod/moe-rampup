#!/usr/bin/env bash

p2_require_stream_bin() {
  local stream_bin="$1"
  if [[ ! -x "$stream_bin" ]]; then
    echo "STREAM binary not found or not executable: $stream_bin"
    echo "Build it first with ./smoke_test.sh"
    exit 1
  fi
}

p2_get_physical_cpu_list() {
  local numa_node="${1:-}"
  if [[ -n "$numa_node" ]]; then
    lscpu -e=CPU,CORE,SOCKET,NODE | awk -v node="$numa_node" '
      NR == 1 { next }
      $4 == node {
        key = $2 ":" $3
        if (!(key in seen)) {
          seen[key] = 1
          cpus[++n] = $1
        }
      }
      END {
        for (i = 1; i <= n; i++) {
          printf "%s", cpus[i]
          if (i < n) printf ","
        }
        printf "\n"
      }
    '
  else
    lscpu -e=CPU,CORE,SOCKET,NODE | awk '
      NR == 1 { next }
      {
        key = $2 ":" $3
        if (!(key in seen)) {
          seen[key] = 1
          cpus[++n] = $1
        }
      }
      END {
        for (i = 1; i <= n; i++) {
          printf "%s", cpus[i]
          if (i < n) printf ","
        }
        printf "\n"
      }
    '
  fi
}

p2_count_cpus_in_list() {
  local cpu_list="$1"
  if [[ -z "$cpu_list" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<< "$cpu_list"
}

p2_first_n_cpus() {
  local cpu_list="$1"
  local n="$2"
  IFS=',' read -r -a cpus <<< "$cpu_list"
  local out=()
  local i
  for ((i = 0; i < n && i < ${#cpus[@]}; i++)); do
    out+=("${cpus[$i]}")
  done
  (IFS=','; echo "${out[*]}")
}

p2_default_thread_list() {
  local max_threads="$1"
  local candidates=(1 2 4 8 16 32 40 48 64 80 96 128 160 192 256)
  local out=()
  local c
  for c in "${candidates[@]}"; do
    if (( c <= max_threads )); then
      out+=("$c")
    fi
  done
  if (( ${#out[@]} == 0 || out[${#out[@]}-1] != max_threads )); then
    out+=("$max_threads")
  fi
  echo "${out[*]}"
}
