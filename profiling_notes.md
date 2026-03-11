# Profiling Notes

## Redirect Memo — EMR (P1)

Date: 2026-02-23
Profiling exercise: P1
Key finding: EMR has simple 2-node NUMA (distance 10/20), so cross-socket memory traffic is the primary topology penalty to avoid.
Implication for kernel track: Prioritize NUMA-safe execution order and expert/thread pinning assumptions before deeper kernel micro-tuning.
Implication for vLLM contribution: Add or improve CPU startup guidance for binding (`VLLM_CPU_OMP_THREADS_BIND`) and locality checks to prevent remote-memory-heavy configs.

## Redirect Memo — GNR (P1)

Date: 2026-02-23
Profiling exercise: P1
Key finding: GNR has 6 NUMA nodes with distance tiers 10/12/21, so both intra-socket and cross-socket placement materially affect MoE memory behavior.
Implication for kernel track: Reprioritize K23 and K13 earlier because locality-aware expert scheduling and prepack placement are likely higher impact than pure SIMD/AMX tuning.
Implication for vLLM contribution: Propose topology-aware thread and memory placement defaults (per-NUMA grouping) plus warnings when expert weights are accessed across far NUMA nodes.

## Redirect Memo — EMR (P2, physical-core pinned)

Date: 2026-02-24
Profiling exercise: P2
Key finding: With physical-core pinning, local NUMA Triad bandwidth is 255454.9 MB/s vs remote 142356.7 MB/s (~1.79x local/remote), while 80-core interleave-all reaches 359633.8 MB/s and sweep peak is 367451.3 MB/s at 80 threads.
Implication for kernel track: Prioritize NUMA-local expert execution and placement first (K23/K13), and treat interleave-all numbers as an upper bound rather than a locality-safe target for MoE decode.
Implication for vLLM contribution: Add a CPU benchmarking recipe and startup warnings that report local-vs-remote BW delta and recommend locality-first thread/memory pinning defaults.

## Redirect Memo — Xeon 6 (P2, physical-core pinned)

Date: 2026-02-24
Profiling exercise: P2
Key finding: With physical-core pinning on Xeon 6, local NUMA Triad is 160723.1 MB/s and remote is 146627.3 MB/s (~1.10x local/remote), while aggregate Triad at 192 threads is 380766.7 MB/s and thread-sweep peak reaches 623707.6 MB/s at 160 threads.
Implication for kernel track: Prioritize scaling behavior and thread-count tuning (around 128-160 threads from this sweep) in addition to NUMA locality, since remote penalty appears much smaller than on EMR in this run.
Implication for vLLM contribution: Add a platform-specific tuning guide that benchmarks local/remote penalty and sweep peak before setting default CPU thread binding, because optimal thread count differs materially by platform.

## P2 Quick Comparison — EMR vs Xeon 6

- NUMA sensitivity: EMR shows strong local/remote gap (~1.79x), while Xeon 6 shows a much smaller gap in this run (~1.10x).
- Best observed sweep Triad: EMR peaks at 367451.3 MB/s (80 threads), Xeon 6 peaks at 623707.6 MB/s (160 threads).
- Tuning implication: EMR should bias harder toward strict locality, while Xeon 6 should include broader thread-count sweeps to find the throughput knee before setting defaults.

## Redirect Memo — GNR (P3, perf stat with GLM-4.7-Flash)

Date: 2026-03-09
Profiling exercise: P3
Key finding: Prefill (`input-len=2048, output-len=1`) shows IPC 1.73 with LLC miss rate 30.16% and branch miss rate 2.68%, while decode shows IPC 0.69 with LLC miss rate 30.87% and branch miss rate 7.31%, indicating decode is much more stall/control-flow limited.
Implication for kernel track: Prioritize decode-path work first (K15-K18, K23, K13): expert execution locality, routing/permutation efficiency, and fusion before large-M prefill GEMM micro-tuning.
Implication for vLLM contribution: Add CPU profiling guidance that requires separate prefill/decode windows and recommends SNC/NUMA-aware binding templates plus branch/control-overhead checks for MoE decode.

## P3 Quick Comparison — Prefill vs Decode (GLM-4.7-Flash)

- IPC: prefill 1.73 vs decode 0.69 (large decode drop).
- LLC load miss rate: prefill 30.16% vs decode 30.87% (similar ratio, decode still throughput-limited).
- Branch mispredict rate: prefill 2.68% vs decode 7.31% (higher decode control-flow overhead).
- Tuning implication: optimize decode first for memory locality and routing/control overhead, then broader compute micro-optimizations.

