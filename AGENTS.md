# Road to MoE: CPU Kernel & Profiling Mastery on Xeon 6

## Target: Qwen3-Next-80B-A3B-Instruct on vLLM CPU

> A progressive difficulty curriculum modeled after Blind 75, designed for someone
> who understands MoE conceptually but has never written intrinsics.

---

## Part 0: Architecture Maps — Two MoE Models, Two Sets of Lessons

You will work with **two models**: GLM-4.7-Flash for profiling (already validated
on your vLLM CPU setup) and Qwen3-Next-80B-A3B as the long-term kernel target.
They use fundamentally different architectures, which is ideal for learning breadth.

---

### Model A: GLM-4.7-Flash (Profiling Target)

**Architecture**: `Glm4MoeLiteForCausalLM` — essentially DeepSeek V3 architecture, renamed.

```
config.json key values:
  hidden_size: 2048          n_routed_experts: 64
  intermediate_size: 10240   num_experts_per_tok: 4
  moe_intermediate_size: 1536  n_shared_experts: 1
  num_hidden_layers: 47      first_k_dense_replace: 1  (layer 0 is dense FFN)
  num_attention_heads: 20    num_key_value_heads: 20
  q_lora_rank: 768           kv_lora_rank: 512
  qk_nope_head_dim: 192      qk_rope_head_dim: 64
  v_head_dim: 256            vocab_size: 154880
  max_position_embeddings: 202752
  num_nextn_predict_layers: 1  (MTP)
  topk_method: "noaux_tc"    routed_scaling_factor: 1.8
```

```
Input Token Embedding (154,880 vocab → 2048 hidden)
     │
     ▼
  Layer 0: Dense (standard attention + dense FFN, no MoE)
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Layers 1–46 (×46):                             │
│    ├─ RMSNorm                                   │
│    ├─ MLA Attention (Multi-Head Latent Attn)     │
│    │   ├─ Compress Q: hidden→768 (q_lora_rank)  │
│    │   ├─ Compress KV: hidden→512 (kv_lora_rank)│
│    │   ├─ KV cache stores COMPRESSED latents     │
│    │   ├─ 20 heads, nope_dim=192, rope_dim=64   │
│    │   └─ v_head_dim=256                         │
│    ├─ RMSNorm                                   │
│    └─ MoE FFN (64 experts, pick 4 + 1 shared)   │
│        ├─ Gate: hidden→64 expert scores          │
│        ├─ Top-4 selection + softmax normalize    │
│        ├─ Each expert: SwiGLU(2048→1536→2048)    │
│        └─ Shared expert: same dims, always active│
└─────────────────────────────────────────────────┘
     │
     ▼
  MTP Head (1 speculative token)
     │
     ▼
  Output Logits
```

**GLM-4.7-Flash Kernel Families:**

| Family | Layers | What It Does | CPU Bottleneck |
|--------|--------|-------------|----------------|
| **MLA Attention** | 47 (all) | Latent-compressed QKV, paged attention decode | Memory BW (KV cache is compressed but still streamed) |
| **MoE FFN** | 46 of 47 | Gate → route → 4+1 experts → combine | Memory BW (streaming expert weights) |
| **Dense FFN** | 1 (layer 0) | Standard SwiGLU FFN | Memory BW (single large FFN) |
| **Normalization + Misc** | 94 RMSNorm, embeddings, MTP | Supporting ops | Fused into adjacent ops ideally |

**GLM-4.7-Flash Memory Footprint:**

| Config | Weight Size | Active Weights/Token | BW Needed @ 50 tok/s |
|--------|------------|---------------------|---------------------|
| BF16 | ~60 GB | ~6.5 GB | 325 GB/s |
| INT8 | ~30 GB | ~3.3 GB | 165 GB/s |
| INT4 | ~15 GB | ~1.6 GB | 80 GB/s |

**Why GLM-4.7-Flash is great for profiling practice:**
- 30B total / 3B active → fits on a single socket in BF16 (~60GB)
- MLA compresses KV cache by ~4× vs standard attention → interesting to profile
- 64 experts / top-4 is moderate sparsity — more experts activate per token than Qwen3-Next
- Already validated on your vLLM CPU setup
- DeepSeek V3-style architecture → the most widely deployed MoE design

**MLA vs Standard Attention (Key Difference):**
Standard attention stores full K,V per head per layer.
MLA compresses KV into a latent space (kv_lora_rank=512 instead of full head dims).
At decode time: load compressed KV → decompress → compute attention.
The compression/decompression is extra compute but dramatically reduces KV cache memory.

---

### Model B: Qwen3-Next-80B-A3B (Kernel Target)

```
Input Token Embedding (151,936 vocab → 2048 hidden)
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Repeating Block (×12)                          │
│                                                 │
│  Layer A (×3): Gated DeltaNet → MoE FFN         │
│    ├─ RMSNorm                                   │
│    ├─ Gated DeltaNet Attention (recurrent, O(n))│
│    ├─ RMSNorm                                   │
│    └─ MoE FFN (512 experts, pick 10 + 1 shared) │
│                                                 │
│  Layer B (×1): Gated Attention → MoE FFN        │
│    ├─ RMSNorm                                   │
│    ├─ Gated Full Attention (KV cache, O(n²))    │
│    ├─ RMSNorm                                   │
│    └─ MoE FFN (512 experts, pick 10 + 1 shared) │
└─────────────────────────────────────────────────┘
     │
     ▼
  MTP Head (multi-token prediction)
     │
     ▼
  Output Logits
```

**Qwen3-Next Kernel Families:**

| Family | Layers | What It Does | CPU Bottleneck |
|--------|--------|-------------|----------------|
| **Gated DeltaNet** | 36 of 48 | Recurrent linear attention, state update | Compute (AMX matmul for state transitions) |
| **Gated Full Attention** | 12 of 48 | Standard MHA with KV cache | Memory BW (paged attention decode) |
| **MoE FFN** | 48 of 48 | Gate → route → 10+1 experts → combine | Memory BW (streaming expert weights) |
| **Normalization + Misc** | 96 RMSNorm, embeddings, MTP | Supporting ops | Fused into adjacent ops ideally |

**Qwen3-Next Memory Footprint:**

| Config | Weight Size | Active Weights/Token | BW Needed @ 50 tok/s |
|--------|------------|---------------------|---------------------|
| BF16 | ~160 GB | ~7.8 GB | 390 GB/s |
| INT8 | ~80 GB | ~3.9 GB | 195 GB/s |
| INT4 (MoE only) | ~50 GB | ~2.4 GB | 120 GB/s |

---

### Side-by-Side Comparison

| Aspect | GLM-4.7-Flash | Qwen3-Next-80B-A3B |
|--------|--------------|-------------------|
| **Attention** | MLA (latent compression, all layers) | Gated DeltaNet (36L) + Gated Attn (12L) |
| **MoE config** | 64 experts, top-4 + 1 shared | 512 experts, top-10 + 1 shared |
| **Sparsity** | ~1:16 activation ratio | ~1:50 activation ratio |
| **KV cache** | Compressed via MLA (kv_lora_rank=512) | Only 12/48 layers need KV cache |
| **Total params** | 30B (3B active) | 80B (3.9B active) |
| **Weight fit** | Single socket BF16 | Needs 2 sockets BF16, 1 socket INT8 |
| **vLLM CPU status** | Validated on your machine | May need testing |

**Hardware bandwidth reference:**
Xeon 6 dual-socket with MRDIMM-8800: ~1.5–1.7 TB/s aggregate bandwidth.
Xeon 5th gen (EMR) with DDR5-5600: ~0.6–0.7 TB/s aggregate bandwidth.

**Key insight**: At INT8, a dual-socket Xeon 6 has ~8× the bandwidth needed per GLM-4.7-Flash
token, meaning you could theoretically sustain ~400+ tok/s decode if your kernels achieve
near-peak bandwidth efficiency. The SGLang team demonstrated 85% bandwidth efficiency for
INT8 MoE on Xeon 6, so the ceiling is real.

---

## Part 1: The Kernel Track — "LeetCPU for MoE"

### How This Works

Each problem has:
- **Problem statement**: what to implement, input/output spec
- **Test protocol**: how to verify correctness and measure perf
- **Performance target**: what "good" looks like on your hardware
- **Editorial hints**: key techniques, what to read
- **ISA focus**: which instruction sets to use

Problems are grouped into tiers. Complete each tier before moving on.
Within each tier, do problems in order — they build on each other.

### Profiling → Kernel Feedback Loop

The profiling track (P-series) and kernel track (K-series) are NOT independent.
Profiling findings should actively redirect your kernel priorities. Here's how:

**After P4 + P4b (Week 3):** You'll have VTune hotspots AND a Perfetto trace
showing exactly which operators dominate GLM-4.7-Flash decode time. Use this to
answer: "Which kernel would give the biggest speedup if I made it 2× faster?"

- If MoE expert execution dominates → prioritize K15-K18, consider pulling K17
  (single expert forward) into Week 4 alongside K9-K10 instead of waiting for Week 7
- If attention dominates → prioritize K21 (paged attention) and K26 (MLA)
- If framework/scheduling overhead is >15% → pause kernel work, read vLLM's
  ModelRunner and Scheduler code, file a bug or optimization PR

**After P6 (Week 4):** You'll know the expert memory access pattern. This directly
determines whether K13 (weight prepacking) or K8 (scatter/gather) matters more.
If hot experts stay L3-resident, prepacking layout for cache line efficiency is
critical. If experts always miss L3, raw bandwidth optimization (prefetching) matters.

**After P7 + P7b (Week 5):** You'll know the prefill/decode crossover point.
If decode is >90% of your use case's latency, skip optimizing prefill GEMM tiling
(K11 large-M cases) and focus on M=1 decode kernels. If prefill matters, spend
more time on K11's cache blocking for large M.

**The rule:** After each profiling exercise, spend 30 minutes writing a "redirect
memo" — a paragraph answering: "Based on what I just measured, should I change
the order of upcoming K-problems? Should I skip any? Should I spend extra time
on any?" Keep these memos in a journal. They're more valuable than the profiles
themselves, because they demonstrate systems-level thinking.

---

### Tier 0: SIMD Foundations (Problems K1–K8)

*Goal: Get comfortable with AVX512 loads, stores, arithmetic, and the mental model of "process 16 floats at once."*

#### K1: Vector Add (BF16) ★☆☆☆☆
```
Problem: Implement elementwise addition of two BF16 arrays.
Input:  float* a, float* b, float* c, int n  (BF16 stored as uint16_t)
Output: c[i] = a[i] + b[i] for all i

Constraints:
- n is a multiple of 32
- Use AVX512-BF16 intrinsics (_mm512_cvtpbh_ps, _mm512_cvtne2ps_pbh)
- Handle the BF16↔FP32 conversion explicitly

Test: Compare against scalar C++ reference. Measure GB/s throughput.
Target: >80% of peak memory bandwidth (stream triad equivalent)

Editorial:
- Start by writing the scalar version first
- Then write it with _mm512_loadu_ps / _mm512_storeu_ps for FP32
- Then add BF16 conversion: load BF16 → convert to FP32 → add → convert back → store
- Key intrinsics: _mm512_loadu_si512, _mm512_cvtpbh_ps, _mm512_cvtneps_pbh
- The BF16 path matters because Qwen3-Next runs in BF16 on CPU by default
- Read: Intel Intrinsics Guide (search "512" + "bfloat16")
```

#### K2: Vector Multiply-Add (FMA) ★☆☆☆☆
```
Problem: Implement c[i] = a[i] * b[i] + c[i] (fused multiply-add)
Input:  float* a, float* b, float* c, int n
Output: c updated in-place

Constraints:
- Use _mm512_fmadd_ps
- n is a multiple of 16

Test: Compare against scalar. Measure GFLOPS.
Target: Understand FMA throughput vs memory bandwidth on your chip.

Editorial:
- This is the most fundamental compute operation — every GEMM is built from FMA
- On Granite Rapids, each core can issue 2× FMA per cycle (each on 512-bit)
- Calculate theoretical peak: cores × FMA_per_cycle × 2 (mul+add) × 16 (FP32 lanes) × freq
- Compare your achieved GFLOPS to peak — the gap is your optimization opportunity
- Try varying n to see when you become compute-bound vs memory-bound
```

#### K3: Horizontal Sum / Reduction ★★☆☆☆
```
Problem: Compute the sum of all elements in an FP32 array.
Input:  float* x, int n
Output: float sum

Constraints:
- n can be up to 10M
- Use _mm512_reduce_add_ps or manual reduction with _mm512_add_ps
- Also implement max reduction (_mm512_reduce_max_ps)

Test: Compare against std::accumulate. Measure throughput.
Target: >70% of memory read bandwidth

Editorial:
- Reduction is fundamental to softmax, RMSNorm, and MoE gating
- Try two approaches: (1) accumulate into a single __m512 register, reduce at end
  (2) use multiple accumulators to hide latency
- The multi-accumulator version should be ~2-3x faster due to instruction pipelining
- Key pattern: "unroll and jam" — process 4 vectors per loop iteration
```

#### K4: SiLU Activation (Vectorized) ★★☆☆☆
```
Problem: Implement SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
Input:  float* x, float* y, int n
Output: y[i] = silu(x[i])

Constraints:
- Use AVX512 throughout
- You'll need to implement exp() approximation in AVX512
- Acceptable error: < 1e-5 relative to scalar

Test: Compare accuracy and throughput against scalar and std::exp.
Target: >5x speedup over scalar

Editorial:
- SiLU is the activation inside every MoE expert's SwiGLU FFN
- exp() doesn't exist as a single intrinsic — you need a polynomial approximation
- Common approach: Cody-Waite range reduction + polynomial
- Alternatively, use _mm512_exp_ps from SVML (Intel Short Vector Math Library)
  but understand it's a library call, not a single instruction
- This problem teaches: what happens when you need transcendentals in SIMD
- Reference: Cephes library, or look at how oneDNN implements eltwise ops
```

#### K5: RMSNorm ★★☆☆☆
```
Problem: Implement RMS Layer Normalization
Input:  float* x, float* weight, float* out, int hidden_size
        (single token, hidden_size = 2048 for Qwen3-Next)
Output: out[i] = (x[i] / rms) * weight[i]
        where rms = sqrt(mean(x[i]^2) + eps)

Constraints:
- hidden_size = 2048 (fixed for Qwen3-Next)
- eps = 1e-6
- Use AVX512

Test: Compare against PyTorch's torch.nn.RMSNorm
Target: This should be memory-bound. Measure against bandwidth ceiling.

Editorial:
- RMSNorm is called 96 times per token in Qwen3-Next (before each attn + each MoE)
- Two-pass approach: (1) compute sum of squares, (2) normalize and scale
- For hidden_size=2048, that's 128 FP32 vectors — fits comfortably in L1
- Advanced: fuse with subsequent operation (e.g., RMSNorm + quantize to INT8)
- This is your first "real model kernel"
- Read: vLLM csrc/cpu/layernorm.cpp
```

#### K6: Softmax (Online / Numerically Stable) ★★★☆☆
```
Problem: Implement softmax over a vector using the online (streaming) algorithm
Input:  float* x, float* out, int n
Output: out[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))

Constraints:
- Use the 3-pass safe softmax OR the 2-pass online softmax
- Must be numerically stable (subtract max first)
- n = 512 (number of experts in Qwen3-Next's MoE gate)

Test: Compare against torch.softmax. Check for numerical accuracy.
Target: Understand the memory access pattern (3 passes vs 2 passes)

Editorial:
- The MoE gating layer computes softmax over 512 expert logits per token
- 3-pass: find max, compute exp(x-max) and sum, divide
- 2-pass (online): single pass for max+sum using the recurrence relation
- For n=512, this fits in L1 cache, so the 3-pass version is fine
- But learning the online algorithm prepares you for attention softmax
- Reference: "Online normalizer calculation for softmax" (Milakov & Gimelshein)
```

#### K7: Top-K Selection (k=10 from 512) ★★★☆☆
```
Problem: Select the top-10 values and indices from a 512-element array
Input:  float* scores, int n=512, int k=10
Output: int top_indices[10], float top_values[10]

Constraints:
- Must be exact (no approximate top-k)
- Use AVX512 sorting networks or partial sort

Test: Compare against std::partial_sort
Target: <1μs for n=512, k=10

Editorial:
- This is the MoE expert selection step — called 48 times per token
- For small n like 512, a simple approach works: maintain a min-heap of size k
- AVX512 approach: use sorting networks for small arrays, or use
  _mm512_cmp_ps_mask + compress to filter
- The AVX512 conflict detection instructions (_mm512_conflict_epi32)
  can help with scatter-based approaches
- Advanced: fused top-k + softmax (compute gating weights only for selected experts)
- Read: How vLLM's moe_align_block_size_kernel works (GPU version, then think about CPU)
```

#### K8: Scatter / Gather Operations ★★★☆☆
```
Problem: Implement gather and scatter with AVX512
(a) Gather: out[i] = table[indices[i]]  (16 simultaneous lookups)
(b) Scatter: table[indices[i]] = values[i]  (16 simultaneous stores)

Input:  float* table, int* indices, float* values, int n_ops
Output: Gathered values or updated table

Constraints:
- Use _mm512_i32gather_ps and _mm512_i32scatter_ps
- Handle potential index conflicts in scatter

Test: Compare against scalar loop
Target: Understand gather/scatter throughput limitations

Editorial:
- MoE routing requires gathering expert weights based on token→expert assignment
- Gather is ~4-6 cycles latency on Granite Rapids (much slower than contiguous load)
- This teaches you WHY expert weight layout matters — contiguous experts are faster
- Scatter conflicts require masking — two tokens routed to same expert need serialization
- This directly maps to the "token permutation" step in MoE
```

---

### Tier 1: AMX Fundamentals (Problems K9–K14)

*Goal: Learn to use Intel AMX tiles for matrix multiplication — the workhorse for MoE expert linear layers.*

#### K9: AMX Tile Load and Store ★★☆☆☆
```
Problem: Load data into AMX tiles and store back
Input:  BF16 matrix A[16][32], BF16 matrix B[32][16]  (in memory)
Output: Successfully loaded into tiles and stored back identically

Constraints:
- Use _tile_loadd / _tile_stored
- Configure tiles with _tile_loadconfig
- Understand the tile register file: 8 tiles, each 1KB (16 rows × 64 bytes)

Test: Round-trip: load → store → compare with original
Target: Understand tile dimensions and stride parameters

Editorial:
- AMX has 8 tile registers (tmm0-tmm7)
- Each tile: up to 16 rows × 64 bytes = 1024 bytes
- For BF16: 16 rows × 32 columns (since BF16 = 2 bytes)
- For INT8: 16 rows × 64 columns
- _tile_loadconfig takes a 64-byte config struct specifying each tile's rows/cols
- The stride parameter in _tile_loadd is the row pitch in bytes (not elements!)
- This is pure mechanics but CRITICAL — wrong tile config = silent wrong results
- Read: Intel AMX programming reference, Chapter 3
```

#### K10: AMX GEMM — BF16 Matmul ★★★☆☆
```
Problem: Multiply two matrices using AMX BF16 tiles
Input:  BF16 A[M][K], BF16 B[K][N]  where M=16, K=32, N=16
Output: FP32 C[M][N] = A × B

Constraints:
- Use _tile_dpbf16ps(dst_tile, src1_tile, src2_tile)
- This computes: C_tile += A_tile * B_tile  (BF16 inputs, FP32 accumulation)
- A is row-major, B must be in "VNNI format" (pairs of BF16 interleaved)

Test: Compare against naive matmul. Verify FP32 accuracy.
Target: >80% of AMX peak TFLOPS for this tile size

Editorial:
- _tile_dpbf16ps is THE key instruction — this is what makes AMX fast
- A tile: 16×32 BF16 (16 rows, 32 cols = 64 bytes/row)
- B tile: 16×32 BF16 but stored in VNNI format (column-major with 2-element interleave)
- C tile: 16×16 FP32 (16 rows, 16 cols = 64 bytes/row)
- One _tile_dpbf16ps: 16×16×32 = 8192 BF16 FMAs
- On Granite Rapids at ~2.4 GHz: theoretical ~20 TFLOPS BF16 per core
- The VNNI repacking of B is essential — without it, results are garbage
- This maps directly to MoE expert linear layers (expert_weight × activation)
```

#### K11: Tiled GEMM — Larger Matrices ★★★★☆
```
Problem: Implement GEMM for arbitrary M, K, N using AMX tiles
Input:  BF16 A[M][K], BF16 B[K][N]
Output: FP32 C[M][N]

Constraints:
- M, K, N can be multiples of 16 (up to 4096)
- Tile over the matrices: process in 16×32×16 blocks
- Handle the K-dimension accumulation (multiple tile multiplies per output block)

Test: Compare against oneDNN matmul. Measure TFLOPS.
Target: >60% of oneDNN performance (this is a GOOD result for a first attempt)

Editorial:
- The tiling loop: for each (m_block, n_block): for each k_block: tile_dpbf16ps
- Key optimization: tile reuse. A tile can be reused across N blocks,
  B tile can be reused across M blocks
- L1/L2 cache blocking matters enormously here:
  - L1 = 48KB (Granite Rapids): can hold ~3 tiles
  - L2 = 2MB: can hold ~2000 tiles
- Optimal loop order depends on M/N/K relative to cache sizes
- For MoE: M = batch_size (1-32 for decode), K = 2048, N = expert_width
  → Small M means you're memory-bound on weight loading
- Compare your implementation against:
  (1) naive scalar (2) your AMX version (3) oneDNN matmul
```

#### K12: INT8 GEMM with AMX ★★★★☆
```
Problem: Implement INT8 matmul using AMX
Input:  INT8 A[M][K], INT8 B[K][N], FP32 scale_a, FP32 scale_b
Output: FP32 C[M][N] = (A × B) * scale_a * scale_b

Constraints:
- Use _tile_dpbssd (signed×signed) or _tile_dpbusd (unsigned×signed)
- B must be in VNNI format for INT8 (4-element interleave, not 2)
- Handle the U8S8 vs S8S8 distinction

Test: Compare against BF16 matmul for accuracy. Measure TFLOPS.
Target: ~2x throughput vs BF16 (INT8 tiles are 64 cols vs 32 cols)

Editorial:
- INT8 AMX tile: A is 16×64 (4× wider than BF16), B is 16×64
- One _tile_dpbusd: 16×16×64 = 16384 INT8 ops (2× BF16)
- U8S8 (unsigned A × signed B) is natively supported
- S8S8 requires the "compensation trick": A_u8 = A_s8 + 128,
  then subtract 128×B column sums from result
- The SGLang team uses this exact trick for their Xeon 6 MoE kernels
- This matters because quantized MoE experts use INT8/UINT8
- Also: per-channel vs per-tensor quantization affects scale application
```

#### K13: Weight Prepacking (VNNI Format) ★★★☆☆
```
Problem: Repack a weight matrix from row-major to AMX VNNI format
Input:  BF16 W[K][N]  (row-major)
Output: BF16 W_packed[N/16][K/32][16][32]  (VNNI tiled layout)

Also implement:
Input:  INT8 W[K][N]  (row-major)
Output: INT8 W_packed[N/16][K/64][16][64]  (VNNI tiled layout for INT8)

Constraints:
- This is a one-time preprocessing step (done at model load)
- Must be fast enough to not dominate model loading time

Test: Verify that GEMM with prepacked weights gives same result
Target: Understand WHY prepacking helps (hint: sequential memory access in tiles)

Editorial:
- VNNI format interleaves adjacent elements in K dimension:
  For BF16: pairs of K values are interleaved (K=0,K=1 together)
  For INT8: quads of K values are interleaved (K=0,1,2,3 together)
- Without prepacking, each _tile_loadd hits non-contiguous memory
- With prepacking, tile loads are sequential cache-line reads
- vLLM CPU uses VLLM_CPU_MOE_PREPACK=1 to do exactly this via IPEX
- For 512 MoE experts × expert_size, prepacking is significant memory layout work
- This is a prerequisite for good MoE kernel performance
```

#### K14: Fused GEMM + Bias + Activation ★★★★☆
```
Problem: Implement GEMM followed by bias addition and SiLU activation
Input:  BF16 A[M][K], BF16 W[K][N], FP32 bias[N]
Output: FP32 out[M][N] = SiLU(A × W + bias)

Constraints:
- Fuse the bias add and SiLU into the GEMM epilogue
- When writing output tiles, apply bias and activation before storing

Test: Compare against separate GEMM + bias + SiLU
Target: Fusion should improve performance by reducing memory round-trips

Editorial:
- In a standard (unfused) implementation: GEMM writes C to memory,
  then bias+SiLU reads C back. That's 2× memory traffic for C.
- Fused: after accumulating each output tile, apply bias+SiLU in registers,
  then store once. Saves one full read+write of the output matrix.
- For MoE decode (M=1-32), the output is small and fits in cache,
  so fusion matters less. But for prefill (M=hundreds), it's significant.
- This pattern is the basis of the SwiGLU FFN: gate=SiLU(x·W_gate), up=x·W_up,
  output = gate ⊙ up, then projected by W_down
- Read: how oneDNN post-ops work (eltwise, binary fusion with matmul)
```

---

### Tier 2: MoE-Specific Kernels (Problems K15–K22)

*Goal: Build the actual MoE pipeline — gating, routing, expert execution, and combining.*

#### K15: MoE Gating Network ★★★☆☆
```
Problem: Implement the MoE gating computation
Input:  BF16 hidden[batch_size][2048]   (hidden states)
        BF16 gate_weight[2048][512]     (gate projection to 512 experts)
Output: float scores[batch_size][512]   (expert scores after softmax)
        int   top_k_ids[batch_size][10] (selected expert indices)
        float top_k_weights[batch_size][10] (gating weights for selected experts)

Constraints:
- Compute: logits = hidden × gate_weight
- Apply softmax over 512 experts
- Select top-10 experts per token
- Normalize top-k weights to sum to 1
- Batch size: 1-32 for decode, 64-512 for prefill

Test: Compare against PyTorch reference
Target: <10μs for batch_size=1 (this should not be the bottleneck)

Editorial:
- For decode (batch=1): logits is 1×512 — tiny matmul, memory-bound
- The gate weight is 2048×512 = 2MB in BF16 — fits in L2 cache
- Fuse: matmul → softmax → top-k into a single kernel
- The top-k step can use a simple partial sort for k=10, n=512
- Key: The selected expert IDs determine which expert weights to load next
  → This is where the "routing" decision happens
- Store results in a format ready for the next step (expert dispatch)
```

#### K16: Token-to-Expert Permutation ★★★★☆
```
Problem: Given routing decisions, permute tokens for grouped expert execution
Input:  int top_k_ids[batch_size][10]     (which experts each token uses)
        BF16 hidden[batch_size][2048]     (token hidden states)
        int num_experts = 512
Output: BF16 permuted_hidden[total_tokens][2048]  (sorted by expert)
        int expert_offsets[513]  (start index for each expert's tokens)
        int source_indices[total_tokens]  (mapping back to original tokens)

Constraints:
- A token appears once for each of its 10 selected experts
- total_tokens = batch_size × 10
- Output must be sorted by expert_id for coalesced weight access

Test: Verify all tokens are accounted for. Verify correct expert grouping.
Target: O(total_tokens + num_experts) — single pass with counting sort

Editorial:
- This is the "align block size" / "token permutation" step from vLLM MoE
- Counting sort is ideal: count tokens per expert → prefix sum → scatter
- For CPU: the scatter step benefits from K8's gather/scatter intrinsics
- With 512 experts and batch=1, each expert gets 10/512 ≈ 0.02 tokens on average
  → Most experts get 0 tokens! Only 10 out of 512 are active
- This extreme sparsity (1:50 ratio) means you should SKIP inactive experts entirely
- The expert_offsets array enables this: if offset[e] == offset[e+1], skip expert e
- Critical insight: For decode with small batch, you're running ~10 tiny matmuls
```

#### K17: Single Expert Forward (SwiGLU FFN) ★★★★☆
```
Problem: Implement one expert's FFN computation
Input:  BF16 x[M][2048]        (M tokens routed to this expert)
        BF16 W_gate[2048][D]   (gate projection)
        BF16 W_up[2048][D]     (up projection)
        BF16 W_down[D][2048]   (down projection)
        D = expert hidden dimension (varies by model)
Output: BF16 out[M][2048]

Computation:
  gate = SiLU(x × W_gate)       # [M, D]
  up   = x × W_up               # [M, D]
  hidden = gate ⊙ up            # elementwise multiply [M, D]
  out  = hidden × W_down         # [M, 2048]

Constraints:
- Fuse SiLU into the gate GEMM (use K14's technique)
- Fuse the elementwise multiply into the down projection if possible
- Handle M=1 (single token decode) efficiently

Test: Compare against PyTorch MLP reference
Target: For M=1, this should be memory-bandwidth-bound on weight loading

Editorial:
- Three GEMMs + one activation + one elementwise multiply
- Total weight loading per expert: W_gate + W_up + W_down
- For M=1 (decode): arithmetic intensity is ~1 FMA per weight loaded
  → Pure memory bandwidth bound. Speed = weight_size / memory_bandwidth
- Optimization strategy for M=1:
  (1) Fuse W_gate and W_up into a single wider matmul (load x once)
  (2) Stream weights through AMX, overlap load of next tile with compute of current
  (3) Use software prefetching (_mm_prefetch) to hide memory latency
- For M>8 (prefill): starts becoming compute-bound, tiling matters more
- Advanced: prepack W_gate and W_up interleaved for better cache utilization
```

#### K18: Full MoE Layer (Decode Path) ★★★★★
```
Problem: Implement a complete MoE forward pass for decode
Input:  BF16 hidden[batch_size][2048]
        BF16 gate_weight[2048][512]
        BF16 expert_weights[512][3]  (W_gate, W_up, W_down per expert)
        batch_size = 1-32

Pipeline:
  1. Gating: compute expert scores and top-k selection (K15)
  2. Permutation: sort tokens by expert (K16)
  3. Expert Execution: run SwiGLU for each active expert (K17)
  4. Combine: weighted sum of expert outputs back to token positions

Output: BF16 out[batch_size][2048]

Constraints:
- Must correctly weight expert outputs by gating scores
- Must handle the shared expert (always active, always computed)
- For batch_size=1: only 10+1 experts are active

Test: Compare against HuggingFace Qwen3Next MoE implementation
Target: Time the full MoE layer. Identify which step dominates.

Editorial:
- For decode (batch=1-4), step 3 (expert execution) dominates by far
  because you're loading 10 expert weight sets from memory
- Shared expert: compute unconditionally, add to output before combining
- The combine step: out[token] = Σ (gate_weight[e] × expert_out[token][e])
  This is a small weighted reduction — not a bottleneck
- KEY OPTIMIZATION: Expert execution order matters for cache locality
  → Process experts whose weights are adjacent in memory consecutively
  → Consider "expert parallelism" across OMP threads
- Each expert's SwiGLU is independent → embarrassingly parallel
- Thread strategy: assign different experts to different OMP threads
  vs. parallelize within each expert's GEMM
- For small batch: inter-expert parallelism is better
  For large batch: intra-expert parallelism wins
```

#### K19: MoE Layer with INT8 Quantization ★★★★★
```
Problem: Same as K18 but with INT8 quantized expert weights
Input:  BF16 hidden[batch_size][2048]
        BF16 gate_weight[2048][512]  (gate stays in BF16)
        INT8 expert_weights_quantized[512][3]  (per-channel quantized)
        FP32 scales[512][3][N_channels]  (quantization scales)
        INT8 zero_points[512][3][N_channels]  (optional)

Additional steps:
  - Dynamic quantization of activations (BF16 → INT8) before expert GEMM
  - Dequantization of output (INT8 accumulator → BF16) after expert GEMM

Test: Compare accuracy against BF16 version. Measure speedup.
Target: ~1.8-2x throughput over BF16 (from halved weight memory traffic)

Editorial:
- INT8 MoE is the practical sweet spot for CPU inference
- Dynamic quantization: find min/max of activation per-token, compute scale
  Then quantize: x_int8 = round(x_bf16 / scale)
- FUSE the quantization into the permutation step (K16):
  while scattering tokens to expert groups, also quantize them
- The SGLang team fuses "quantization from BF16 to UINT8 with fetching of activation"
  — this is exactly what you should do
- U8S8 vs S8S8 on AVX512-VNNI vs AMX:
  AVX512-VNNI: only supports U8S8 → need the +128 compensation
  AMX: supports both U8S8 (dpbusd) and S8S8 (dpbssd)
- Per-channel scales: each output channel of the weight has its own scale
  → Apply scales after the INT32 accumulation, before storing
- Memory savings: 80GB → 40GB for expert weights alone
```

#### K20: Gated DeltaNet Recurrence ★★★★★
```
Problem: Implement the Gated DeltaNet attention recurrence
This is the LINEAR attention that replaces standard attention in 36/48 layers.

Input:  BF16 q[1][d_head], k[1][d_head], v[1][d_head]  (single step decode)
        BF16 alpha[1][d_head]  (gating for state update)
        BF16 beta[1][d_head]   (gating for output)
        FP32 state[d_head][d_head]  (recurrent state matrix from previous step)
Output: FP32 state_new[d_head][d_head]  (updated state)
        BF16 output[1][d_head]  (attention output for this step)

Recurrence (per head):
  state_new = alpha ⊙ state + beta ⊙ (k^T × v)   (rank-1 update with gating)
  output = q × state_new

Where ⊙ is element-wise gating (broadcast over state dimensions)

Constraints:
- d_head is typically 64 or 128
- Multi-head: repeat for all heads, parallelize across heads
- State is FP32 (accumulated over many steps, needs precision)
- q, k, v come from linear projections (separate kernel)

Test: Compare against Python reference implementation of DeltaNet
Target: This should be compute-bound (state update is a matrix operation)

Editorial:
- This is NOT standard attention — there is NO KV cache for DeltaNet layers
- Instead, there's a RECURRENT STATE per head, updated each step
- state is d_head × d_head = 64×64 = 16KB per head (FP32)
  With ~32 heads × 36 layers = ~18MB total recurrent state
  → Fits in L2/L3 cache! Very different from KV cache.
- The k^T × v is a rank-1 outer product: d_head × d_head matrix from two vectors
- alpha gating: element-wise scale of the old state (forget gate analog)
- beta gating: element-wise scale of the new information
- For AMX: the outer product k^T × v maps to a tile multiply with M=d_head, K=1, N=d_head
  But K=1 is too small for AMX efficiency → batch multiple steps if possible
- Key optimization: For prefill, batch the recurrence over sequence chunks
- For decode (single step): each head's update is small, parallelize across heads
- This is the most novel kernel in the whole model — least existing reference code
```

#### K21: Paged Attention (Decode, for Gated Attention Layers) ★★★★★
```
Problem: Implement CPU paged attention for the 12 standard attention layers
Input:  BF16 q[batch][num_heads][d_head]
        BF16 kv_cache[num_blocks][2][num_heads][block_size][d_head]
        int block_tables[batch][max_blocks]
        int context_lens[batch]
Output: BF16 out[batch][num_heads][d_head]

Constraints:
- block_size = 128 (vLLM CPU default)
- d_head = 128 (typical for Qwen3-Next)
- Only 12 layers use this (Gated Attention layers), rest use DeltaNet (K20)
- Use AVX512 for the attention score computation and softmax

Test: Compare against vLLM's existing CPU paged attention kernel
Target: Match or beat vLLM's existing implementation

Editorial:
- Only 12 out of 48 layers need KV cache — massive memory savings vs standard model
- The KV cache for 12 layers is ~4× smaller than a full 48-layer model
- Implementation follows FlashAttention algorithm adapted for CPU:
  (1) Loop over KV blocks for each query
  (2) Compute attention scores: Q × K^T (this is a small GEMV per block)
  (3) Online softmax: track running max and sum
  (4) Weighted sum of V using attention weights
- For CPU: the key optimization is cache blocking — each KV block (128 × d_head)
  should fit in L1/L2
- On Granite Rapids L1=48KB: one block of K or V = 128×128×2 = 32KB ✓
- Reference: vLLM csrc/cpu/attention.cpp
- Advanced: fuse RoPE into the attention computation (avoid separate pass)
```

#### K22: Fused MoE + RMSNorm + Residual ★★★★★
```
Problem: Fuse the residual connection, RMSNorm, and MoE dispatch
Input:  BF16 residual[batch][2048]     (residual stream)
        BF16 attn_output[batch][2048]  (attention layer output)
        BF16 rmsnorm_weight[2048]      (RMSNorm parameters)
        + all MoE weights
Output: BF16 new_residual[batch][2048] (updated residual after MoE)

Computation:
  hidden = residual + attn_output          # residual add
  normed = RMSNorm(hidden)                 # normalize
  moe_out = MoE(normed)                    # full MoE layer
  new_residual = hidden + moe_out          # another residual add

Constraints:
- Fuse the first three operations into a single pass:
  residual add → RMSNorm → quantize for MoE dispatch
- Fuse the final residual add with the MoE combine step

Test: Compare against unfused version
Target: Measure memory bandwidth savings from fusion

Editorial:
- Without fusion: hidden is written to memory, then read back for RMSNorm,
  then normed is written, then read back for MoE gating
- With fusion: residual add + RMSNorm + quantize happens in registers,
  output goes directly to MoE gate computation
- For hidden_size=2048, this is 4KB per token — trivially fits in L1
- The real saving is eliminating 2 unnecessary memory round trips
- For the final residual: MoE combine + residual add is also fusible
- This is how production kernels work — never do single ops in isolation
- Read: how the SGLang Intel team structures their fused kernels
```

---

### Tier 3: System-Level Kernels (Problems K23–K25)

*Goal: Full model layer execution with proper parallelism and memory management.*

#### K23: Multi-Expert Parallel Execution with OMP ★★★★★
```
Problem: Execute 10 active MoE experts in parallel using OpenMP
Input:  Token data + 10 expert weight sets
Output: Combined expert outputs

Design decisions:
- Thread binding: which OMP threads handle which experts?
- NUMA-awareness: are expert weights on the right NUMA node?
- Load balancing: experts may have different token counts

Constraints:
- Use #pragma omp parallel with proper scheduling
- Measure and minimize thread synchronization overhead
- Test with 1, 2, 4, 8, 16, 32 threads

Test: Compare throughput vs sequential expert execution
Target: Near-linear scaling up to memory bandwidth saturation

Editorial:
- For decode (batch=1): 10 independent expert GEMMs → parallelize across experts
- Each expert GEMM is small: M=1, K=2048, N=D → memory-bound
- With 10 experts on 32+ cores: each expert gets ~3 threads
- But memory bandwidth is shared! More threads don't help once BW is saturated
- The optimal thread count depends on memory bandwidth / per-expert BW need
- NUMA matters: if expert weights are on NUMA node 0 but thread runs on node 1,
  you get 50% bandwidth (cross-NUMA access)
- Strategy: preload experts to specific NUMA nodes, pin threads accordingly
- Advanced: overlap expert[i] computation with expert[i+1] weight prefetching
```

#### K24: Full Transformer Layer (DeltaNet variant) ★★★★★
```
Problem: Implement one complete "Gated DeltaNet → MoE" layer
Input:  BF16 residual[batch][2048], recurrent state, all weights
Output: Updated residual, updated recurrent state

This combines:
  - RMSNorm
  - Q/K/V projections (linear layers)
  - Gated DeltaNet recurrence (K20)
  - Output projection
  - Residual add
  - RMSNorm
  - Full MoE layer (K18 or K19)
  - Residual add

Constraints:
- Profile each sub-component's contribution to total time
- Identify the actual bottleneck on your hardware

Target: Understand the full layer's performance profile

Editorial:
- This is where you see the full picture
- For decode: MoE dominates (expert weight loading)
- For prefill: DeltaNet recurrence may dominate (compute-heavy state updates)
- The Q/K/V projections are standard GEMMs — use AMX
- Profile with: perf stat, VTune, or manual timing per sub-component
- Create a roofline model: plot each sub-component's arithmetic intensity
  vs achieved throughput to identify what limits each one
```

#### K25: MTP (Multi-Token Prediction) Head ★★★★☆
```
Problem: Implement the multi-token prediction mechanism
Input:  BF16 hidden[batch][2048] from last layer
Output: Multiple next-token predictions for speculative decoding

Constraints:
- MTP predicts N tokens ahead (typically 1 for GLM-4.7-Flash, 3-4 for Qwen3-Next)
- Each prediction requires a lightweight forward pass
- Must be fast enough that speculation saves more time than it costs

Test: Verify predictions match HuggingFace implementation
Target: MTP overhead should be <30% of single token decode time

Editorial:
- MTP is used by both GLM-4.7-Flash (1 speculative token) and Qwen3-Next
- On CPU: MTP is especially valuable because decode is memory-bound
  → Predicting 3 tokens costs ~1.3x but gives ~3x tokens if predictions are right
- Implementation: run the lightweight MTP forward, get top candidates,
  verify in the next full forward pass
- This is relatively straightforward — the hard part is integration with
  vLLM's scheduling, not the kernel itself
```

#### K26: MLA Attention (Multi-Head Latent Attention) ★★★★★
```
Problem: Implement MLA decode attention as used in GLM-4.7-Flash
This is the DeepSeek V3-style attention with latent KV compression.

Input:  BF16 hidden[batch][2048]
        BF16 Wq_compress[2048][768]    (q_lora_rank)
        BF16 Wq_up[768][5120]          (expand to num_heads × (nope_dim + rope_dim))
        BF16 Wkv_compress[2048][512]   (kv_lora_rank, single projection)
        BF16 Wkv_up[512][???]          (expand to KV heads)
        Compressed KV cache: [num_blocks][block_size][512]  (latent vectors)
        Block tables, context lengths (same as paged attention)

Output: BF16 out[batch][2048]

Computation:
  1. Compress hidden → q_latent (768-dim)
  2. Expand q_latent → Q heads: nope (192-dim) + rope (64-dim) per head
  3. Apply RoPE to the rope portion of Q
  4. For cached KV: the cache stores COMPRESSED latents (512-dim)
     → Decompress at attention time OR absorb decompression into Q projection
  5. Compute attention scores, softmax, weighted sum
  6. Output projection

Constraints:
- 20 attention heads
- nope_head_dim=192 (no positional encoding), rope_head_dim=64
- v_head_dim=256
- KV cache stores compressed latents (512-dim), NOT full K and V
- The "absorption trick": fold W_kv_up into W_q, avoiding runtime decompression

Test: Compare against HuggingFace GLM-4.7-Flash MLA implementation
Target: Understand the compute/memory tradeoff of MLA vs standard attention

Editorial:
- MLA is the most widely deployed efficient attention mechanism (DeepSeek V3, GLM-4.7)
- Key insight: KV cache is 512-dim instead of 20_heads × (192+64+256) = ~10240-dim
  → ~20× KV cache compression! This is why MLA enables long context on CPU.
- The "absorption trick" is critical for efficiency:
  Instead of: load compressed KV → decompress → compute attention
  You can: fold the decompression matrix into Q → compute attention on compressed KV
  This trades a larger Q projection for smaller KV cache loads
- On CPU: the reduced KV cache size means more context fits in L3 cache
  → Attention becomes less memory-bound for moderate context lengths
- This problem is important because you'll PROFILE this kernel on GLM-4.7-Flash
  → Understanding how it works helps you interpret VTune hotspots
- Reference: DeepSeek V2/V3 papers, vLLM model_executor/models/glm4_moe_lite.py
```

---

### Appendix A: Gated DeltaNet Reference Implementations

The Gated DeltaNet recurrence (K20) is the highest-uncertainty kernel. Here are the
reference implementations ordered from most readable to most optimized:

#### 1. Sebastian Raschka's From-Scratch Implementation (START HERE)
```
Repo: github.com/rasbt/LLMs-from-scratch
Path: ch04/08_deltanet/
URL:  https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/

What you get:
- Pure PyTorch implementation of Gated DeltaNet, no Triton/CUDA
- Step-by-step notebook walking through the recurrence
- Clear separation of: Q/K/V projections → gating → state update → output
- Comparison with standard attention, discussion of the 3:1 hybrid ratio
- Explains the relationship to Mamba2 gating

Why start here:
- Written for pedagogy, not performance
- Python loops over timesteps — maps directly to the scalar recurrence
- The gating variables (alpha = decay, beta = update) are explicit
- You can run this on CPU trivially

The core recurrence (simplified from Raschka's code):
  for t in range(seq_len):
      S = alpha[t] * S + beta[t] * (k[t].unsqueeze(-1) @ v[t].unsqueeze(-2))
      o[t] = q[t] @ S

Where S is the [d_head × d_head] recurrent state per head.
```

#### 2. NVlabs Official Gated DeltaNet Repository
```
Repo: github.com/NVlabs/GatedDeltaNet
URL:  https://github.com/NVlabs/GatedDeltaNet

What you get:
- ICLR 2025 official code
- Training scripts on SlimPajama
- Links to FLA-based optimized kernels
- Model configs for 0.4B, 1.3B parameter models

Key file: gated_delta_net/gated_delta_net.py
- Contains the layer implementation with short convolutions
- Shows how alpha (decay) and beta (update) gates are computed:
  alpha = sigmoid(a_proj(x))   # per-head scalar
  beta  = sigmoid(b_proj(x))   # per-head scalar
- Modes: 'chunk' (training, parallel), 'fused_recurrent' (inference, sequential)

For CPU kernel work, focus on the 'fused_recurrent' mode — that's what
decode inference uses.
```

#### 3. FLA (Flash Linear Attention) Library — Optimized Triton Kernels
```
Repo: github.com/fla-org/flash-linear-attention
URL:  https://github.com/fla-org/flash-linear-attention

Key paths:
  fla/layers/gated_delta_net.py          — Layer wrapper
  fla/ops/gated_delta_rule/              — Kernel implementations
    ├── recurrent.py                     — Fused recurrent (decode path)
    ├── chunk.py                         — Chunkwise parallel (prefill path)
    └── (various Triton kernels)

What you get:
- Production-quality Triton GPU kernels
- Chunkwise algorithm that parallelizes the recurrence across chunks
- Fused recurrent kernel for single-step decode
- Benchmarks showing DeltaNet throughput competitive with Mamba2

For CPU porting:
- Study the recurrent.py kernel — this is the decode path
- The chunkwise algorithm in chunk.py is for training/prefill
- Key operation: state = gate * state + beta * outer_product(k, v)
  then output = q @ state
- The Triton kernels use tile-based computation — maps to AMX tiles
- The WY representation optimization (for chunkwise) is specific to
  parallelizing the sequential recurrence — understand it for prefill kernels
```

#### 4. Qwen3-Next's Actual Implementation
```
Repo: HuggingFace model files for Qwen/Qwen3-Next-80B-A3B-Instruct
Path: modeling_qwen3next.py (in the HF model repo)

What you get:
- The exact Gated DeltaNet variant used in Qwen3-Next
- How it integrates with the 3:1 hybrid ratio
- The specific gate parameterization Qwen3 chose
- How the output gating (SiLU-based) wraps the attention output

Cross-reference with the FLA implementation to understand differences.
```

#### 5. DeltaNet Theory — Blog Posts (Read for Understanding)
```
Songlin Yang's blog series (the algorithm inventor):
  Part 1: https://sustcsonglin.github.io/blog/2024/deltanet-1/
  Part 2: https://sustcsonglin.github.io/blog/2024/deltanet-2/

Part 1 covers:
- Why linear attention fails (memory overload without forgetting)
- The delta rule: S = S + beta * k^T(v - S^Tk) = "update toward correct output"
- Connection to Test-Time-Training (TTT)
- Why DeltaNet excels at associative recall (MQAR benchmark)

Part 2 covers:
- The chunkwise parallelization algorithm (key for prefill)
- WY representation for efficient parallel prefix scan
- Hardware efficiency: why chunkwise beats parallel scan for matrix-valued states
- Throughput benchmarks vs Mamba2, GLA, standard attention

The paper: "Parallelizing Linear Transformers with the Delta Rule
over Sequence Length" (NeurIPS 2024, arXiv:2406.06484)

Gated DeltaNet paper: "Gated Delta Networks: Improving Mamba2
with Delta Rule" (ICLR 2025, arXiv:2412.06464)
```

#### CPU Porting Strategy for Gated DeltaNet (K20)
```
The decode path is straightforward to port to CPU:

1. Start with Raschka's Python loop → verify understanding
2. Translate to C++ with naive loops → verify correctness
3. Vectorize the outer product k^T × v with AVX512
   (this is a rank-1 update: d_head multiplies by d_head = d_head² ops)
4. Vectorize the state scaling (alpha * S) with AVX512
   (elementwise multiply of d_head × d_head matrix)
5. Vectorize the output q × S with AVX512
   (matrix-vector multiply: d_head × d_head matrix times d_head vector)
6. Try AMX for the state update when d_head=64 or 128:
   The state S fits in a few AMX tiles (64×64 = 4 tiles, 128×128 = 64 tiles)

Key realization: For decode (single step), the Gated DeltaNet recurrence
per head is:
  - 1 outer product: O(d²) — d=64 means 4096 FMAs
  - 1 elementwise scale: O(d²) — 4096 multiplies
  - 1 matvec: O(d²) — 4096 FMAs
  - Total: ~12K FLOPs per head per step

With 32 heads × 36 layers = 1152 state updates per token
Total: ~14M FLOPs — this is COMPUTE, not memory bandwidth!

Compare to MoE: ~8B FLOPs but memory-bound.
DeltaNet is the opposite: small but compute-dense.
The state (d×d per head) fits in L1/L2 cache → no memory bottleneck.
This is AMX territory: tile multiply is perfect for the outer product and matvec.
```

---

## Part 2: The Profiling Track — "System Design for MoE Inference"

### How This Works

Each problem involves profiling a real running system, identifying bottlenecks,
and making measurable improvements. You need an actual Xeon machine for these.

---

### Tier 0: Tool Proficiency (Problems P1–P5)

#### P1: Hardware Topology Discovery ★☆☆☆☆
```
Task: Map your machine's full hardware topology
Tools: lscpu, lstopo (hwloc), numactl --hardware, dmidecode

Deliverables:
1. How many sockets, cores, threads?
2. Draw the NUMA topology: which cores → which NUMA node → which memory?
3. What is the L1/L2/L3 cache size per core and shared?
4. What ISA extensions are available? (grep for avx512, amx in /proc/cpuinfo)
5. What is the theoretical memory bandwidth per socket?
   (MRDIMM channels × frequency × bytes/transfer)
6. What's different between your Xeon 6 and EMR machines?

Why: You can't optimize what you don't understand. Every profiling decision
depends on this topology.
```

#### P2: Memory Bandwidth Measurement ★★☆☆☆
```
Task: Measure actual memory bandwidth on your system
Tools: Intel MLC (Memory Latency Checker), STREAM benchmark

Deliverables:
1. Run STREAM on a single NUMA node. Record bandwidth.
2. Run STREAM across all NUMA nodes. Record bandwidth.
3. Measure cross-NUMA bandwidth (allocate on node 0, read from node 1)
4. Measure with different numbers of threads (1, 2, 4, 8, 16, 32, all)
5. Plot: bandwidth vs thread count, identify saturation point

Target measurements to record:
- Single-socket sequential read bandwidth
- Dual-socket aggregate bandwidth
- Cross-NUMA penalty ratio (should be ~1.5-2x slower)

Why: Your MoE kernel's ceiling is directly determined by these numbers.
The "85% bandwidth efficiency" target from the SGLang team is relative to
these measured numbers, not theoretical peak.
```

#### P3: perf stat Basics ★★☆☆☆
```
Task: Profile a simple vLLM CPU inference run with perf stat
Setup: Run vLLM with GLM-4.7-Flash (zai-org/GLM-4.7-Flash) on CPU

Commands:
  perf stat -e cycles,instructions,cache-references,cache-misses,\
    LLC-loads,LLC-load-misses,\
    branches,branch-misses \
    -p <vllm_pid> sleep 10

Deliverables:
1. What is the IPC (instructions per cycle)?
2. What is the LLC (last level cache) miss rate?
3. What is the branch misprediction rate?
4. How do these change between prefill vs decode phase?
   (Run separate profiling windows for each)

Why: IPC < 1.0 typically means memory-bound. LLC miss rate tells you if
model weights are streaming through cache or fitting. Branch misprediction
indicates control flow overhead (MoE routing decisions).
```

#### P4: VTune / perf record Hotspot Analysis ★★★☆☆
```
Task: Identify the top-10 hottest functions in vLLM CPU MoE inference
Tools: Intel VTune Profiler (free) or perf record + perf report

Setup: Run vLLM serving GLM-4.7-Flash on CPU, send a sustained workload

Deliverables:
1. Top-10 functions by CPU time (both self and inclusive)
2. Classify each as: GEMM / attention / MoE routing / framework overhead / other
3. What % of time is in oneDNN kernels?
4. What % of time is in IPEX kernels?
5. What % of time is Python/framework overhead?
6. What % of time is in memory allocation (tcmalloc)?

Why: This tells you WHERE to focus optimization. If 80% of time is in oneDNN
matmul, your kernel improvements need to beat oneDNN. If 20% is framework
overhead, that's a different problem entirely.
```

#### P4b: vLLM Torch Profiler Trace Analysis ★★★☆☆
```
Task: Capture and analyze a vLLM torch.profiler trace of GLM-4.7-Flash
Tools: vLLM built-in profiler, Perfetto (ui.perfetto.dev)

Setup:
  # Option 1: Offline latency benchmark with profiling
  VLLM_TORCH_PROFILER_DIR=./traces \
    vllm bench latency \
    --model zai-org/GLM-4.7-Flash \
    --num-iters-warmup 3 \
    --num-iters 1 \
    --batch-size 1 \
    --input-len 128 \
    --output-len 32

  # Option 2: Server with profiling (captures real serving behavior)
  VLLM_TORCH_PROFILER_DIR=./traces \
    vllm serve zai-org/GLM-4.7-Flash \
    --profiler-config '{"profiler": "torch"}'
  # Then trigger profiling:
  curl -X POST http://localhost:8000/start_profile
  # Send a few requests...
  curl -X POST http://localhost:8000/stop_profile

  # View: Open the .json.gz trace in https://ui.perfetto.dev

Deliverables:
1. Screenshot/annotation of one full decode step in Perfetto timeline
2. Identify: which aten ops correspond to MoE gating, expert matmul,
   attention, RMSNorm (label them in the timeline)
3. Measure wall-clock time for each component within a single decode step
4. Identify gaps: time between compute regions where nothing happens
   (these are scheduling overhead, Python GIL, memory allocation)
5. Compare this timeline view against P4's aggregate VTune numbers.
   Do they tell the same story? (They should, but from different angles.)

Why: This is the tool you'll use daily as a vLLM contributor. VTune gives
aggregates; the Perfetto trace gives you the TIMELINE — you can see the
exact sequence of operations, spot serialization, and identify fusion
opportunities (adjacent ops that could be merged to avoid memory round-trips).
The trace format is identical for CPU and GPU (GPU adds CUDA kernel lanes),
so this skill transfers directly.
```

#### P5: OMP Thread Binding Experiment ★★★☆☆
```
Task: Measure the impact of different OMP thread binding strategies
Setup: Same vLLM GLM-4.7-Flash model, varying VLLM_CPU_OMP_THREADS_BIND

Experiments:
1. Default (no binding): measure throughput
2. Bind to one NUMA node: VLLM_CPU_OMP_THREADS_BIND=0-31
3. Bind across NUMA nodes: VLLM_CPU_OMP_THREADS_BIND=0-63
4. Leave cores for framework: VLLM_CPU_OMP_THREADS_BIND=0-30 (reserve 31)
5. Compact binding: cores 0-31 (one thread per core)
6. Spread binding: alternating cores across NUMA nodes

Deliverables:
1. Throughput (tok/s) for each configuration
2. Latency (time to first token, time per output token) for each
3. CPU utilization per core (htop or perf)
4. Explain why the best configuration is the best

Why: Thread binding is the #1 configuration knob for CPU inference.
Wrong binding → 2-3x performance loss from NUMA effects.
```

---

### Tier 1: MoE-Specific Profiling (Problems P6–P10)

#### P6: MoE Expert Weight Memory Layout Analysis ★★★☆☆
```
Task: Analyze how MoE expert weights are laid out in memory
Tools: perf mem, Intel VTune memory access analysis, custom instrumentation
Model: GLM-4.7-Flash (64 experts, top-4 + 1 shared)

Questions to answer:
1. Where are 64 expert weight tensors physically in memory? (NUMA node?)
2. Are expert weights contiguous or scattered across pages?
3. When 4+1 experts activate, how much total memory is touched?
4. What is the cache hit rate for expert weights during decode?
   (Hot experts that repeat across tokens may stay cached)
5. Does the expert access pattern exhibit temporal locality?
   (Some experts are "popular" — measure the frequency distribution)

Deliverables:
- Memory heatmap: which address ranges are accessed per MoE layer
- Expert frequency histogram over a real conversation
- Cache residency analysis: what stays in L3 between tokens?

Why: With only 64 experts (vs 512 in Qwen3-Next), there's a much higher
chance that frequently-used experts stay resident in L3 cache. If the top-10
most popular experts handle 50%+ of tokens, they may NEVER leave cache,
making effective bandwidth much higher than DRAM bandwidth.
This fundamentally changes the optimization strategy.
```

#### P7: Prefill vs Decode Profiling ★★★★☆
```
Task: Create detailed profiles for both prefill and decode phases
Setup: Send prompts of varying length (128, 512, 2048, 8192 tokens)

For each phase, measure:
1. Time per layer (attention + MoE, separated)
2. Memory bandwidth utilization (perf events for memory controller)
3. AMX utilization (if using AMX kernels)
4. Thread efficiency (how much time in barriers vs compute)

Deliverables:
- Table: operation × prompt_length → time_ms
- Identify the crossover point where MoE stops being memory-bound
  and becomes compute-bound (should happen around batch 8-16)
- Create a roofline plot for your system with each operation plotted

Why: Prefill and decode have fundamentally different optimization strategies.
Decode is memory-bound (optimize for bandwidth). Prefill is mixed (optimize
for compute AND bandwidth). Understanding the crossover is essential.
```

#### P7b: Prefill vs Decode Trace Comparison ★★★★☆
```
Task: Capture vLLM torch profiler traces for prefill-heavy and decode-heavy
workloads and visually compare operator shapes in Perfetto

Setup:
  # Trace 1: Prefill-dominated
  VLLM_TORCH_PROFILER_DIR=./traces_prefill \
    vllm bench latency \
    --model zai-org/GLM-4.7-Flash \
    --num-iters-warmup 3 --num-iters 1 \
    --batch-size 1 --input-len 2048 --output-len 1

  # Trace 2: Decode-dominated
  VLLM_TORCH_PROFILER_DIR=./traces_decode \
    vllm bench latency \
    --model zai-org/GLM-4.7-Flash \
    --num-iters-warmup 3 --num-iters 1 \
    --batch-size 1 --input-len 32 --output-len 256

Deliverables:
1. Open both traces in Perfetto. Screenshot one forward pass from each.
2. In the prefill trace: identify the large GEMM blocks (MoE experts
   processing 2048 tokens at once). These should be wide, dense compute.
3. In the decode trace: identify the repeated thin GEMM blocks (MoE
   experts processing 1 token at a time). These should be short with
   memory-dominated gaps between them.
4. Measure: what fraction of wall-clock time is actual compute vs
   memory stalls in each trace?
5. If the MoE layer looks different between prefill and decode, describe
   how — this is the prefill/decode asymmetry that drives kernel design.
6. REDIRECT MEMO: Based on these traces, which phase (prefill or decode)
   should you optimize first for your expected workload? Does this change
   which K-problems you should prioritize?

Why: This exercise connects P7's quantitative measurements to visual
evidence. Seeing the shape difference between a prefill forward pass
and a decode forward pass in Perfetto builds an intuition that no
amount of aggregate profiling can give you. This is also the exact
workflow GPU engineers use (with Nsight instead of Perfetto for CPU).
```

#### P8: tcmalloc vs jemalloc vs System Malloc ★★★☆☆
```
Task: Measure the impact of memory allocator on MoE inference
Setup: Run the same workload with:
  (a) LD_PRELOAD=libtcmalloc_minimal.so
  (b) LD_PRELOAD=libjemalloc.so
  (c) No preload (glibc malloc)

Deliverables:
1. Throughput comparison
2. Memory usage comparison (RSS, peak allocation)
3. malloc/free time (use LD_PRELOAD profiling or VTune)
4. Memory fragmentation over time (sustained serving for 10 minutes)

Why: MoE models allocate and free many intermediate buffers during
expert execution. A slow allocator can add 10-20% overhead.
vLLM recommends tcmalloc for a reason — measure why.
```

#### P9: Tensor Parallelism Across NUMA Nodes ★★★★★
```
Task: Profile vLLM CPU with tensor parallelism on a 2-socket machine
Setup: VLLM_CPU_OMP_THREADS_BIND=0-31|32-63 (2 TP ranks)
Model: GLM-4.7-Flash (60GB BF16 — fits on one socket, but test TP anyway)

Analyze:
1. How are model weights partitioned between the two sockets?
2. What is the communication overhead between TP ranks?
   (shared memory? gloo? what backend does vLLM CPU use?)
3. How does MoE expert assignment change with TP?
   (With 64 experts: does each socket get 32?)
4. Is the all-reduce after MoE the bottleneck?
5. Measure: 1-socket throughput × 2 vs actual 2-socket TP throughput
   What is the TP efficiency?

Deliverables:
- Breakdown: compute time vs communication time per layer
- Analysis: is TP helping or hurting for GLM-4.7-Flash at various batch sizes?
  (Since the model fits on 1 socket, TP may hurt due to communication overhead
   unless batch sizes are large enough to benefit from doubled memory bandwidth)

Why: For GLM-4.7-Flash, TP may not help (model fits in 1 socket).
But for Qwen3-Next-80B (160GB BF16), you NEED TP.
Understanding TP overhead here prepares you for the larger model.
```

#### P10: Batch Size vs Throughput vs Latency ★★★★★
```
Task: Create a comprehensive batch_size → performance curve for MoE decode
Setup: Use vLLM benchmarking tools, sweep batch_size from 1 to 128

Measure at each batch size:
1. Tokens per second (throughput)
2. Time per output token (latency)
3. Time to first token (TTFT)
4. Memory bandwidth utilization
5. AMX utilization (if available)

Deliverables:
- Plot: batch_size vs throughput (should be sublinear then plateau)
- Plot: batch_size vs latency (should increase once BW saturates)
- Identify the "sweet spot" batch size for your deployment
- Calculate: what batch size makes MoE compute-bound?
  (hint: arithmetic_intensity = batch_size × 2 / bytes_per_weight)

Why: This is the fundamental deployment parameter. Too small → wasted BW.
Too large → unacceptable latency. The right answer depends on your hardware.
```

---

### Tier 2: Advanced System Profiling (Problems P11–P15)

#### P11: Expert Load Balancing Analysis ★★★★☆
```
Task: Measure expert utilization patterns in GLM-4.7-Flash
Setup: Instrument the MoE gating to log expert selection over 10K tokens

Questions:
1. Is expert selection uniform or skewed? (64 experts, top-4)
2. Are there "hot" experts that are selected >10× more than average?
3. How does the pattern change for different prompt types (code vs chat vs reasoning)?
4. What's the max and min token count per expert in a batch?
5. If you cached the top-10 hottest experts' weights in L3, what cache hit rate would you get?
6. Does the shared expert's output dominate, or is the routed contribution significant?

Why: Load imbalance is the #1 MoE efficiency problem at scale.
With 64 experts (vs 512), the distribution is likely MORE uniform per token,
but there will still be global hotspots. Understanding GLM-4.7-Flash's
patterns here directly informs Qwen3-Next optimization (where 512 experts
makes the problem dramatically harder).
```

#### P12: End-to-End Serving Profile ★★★★★
```
Task: Profile a realistic serving scenario (not just isolated inference)
Setup: vLLM serve GLM-4.7-Flash, send concurrent requests with varying loads

Measure:
1. What % of time is in model forward pass vs scheduling vs tokenization?
2. How does continuous batching interact with MoE? (Dynamic batch sizes)
3. Memory pressure: how does KV cache + recurrent state + request buffers
   interact? Does GC / memory pressure cause stalls?
4. What's the maximum QPS before latency degrades unacceptably?

Why: Real deployment is not a simple benchmark loop. The scheduling overhead,
memory management, and request batching all interact with MoE's unique
memory access patterns.
```

#### P13: Compare vLLM CPU vs SGLang CPU on Your Hardware ★★★★☆
```
Task: Benchmark both serving frameworks on the same model + hardware
Setup: Install both vLLM and SGLang with CPU backends

Compare:
1. Throughput at various batch sizes
2. Latency (TTFT and TPOT)
3. Memory usage
4. Kernel hotspots (what kernels does each framework call?)
5. Which framework better utilizes AMX?

Why: SGLang's Intel team has made significant CPU MoE optimizations.
Understanding the delta between frameworks reveals optimization opportunities.
If SGLang is faster, study their kernels. If vLLM is faster, study their scheduling.
```

#### P14: MRDIMM vs DDR5 Bandwidth Impact ★★★★☆
```
Task: If you have both Xeon 6 (MRDIMM) and EMR (DDR5), compare them
Setup: Same model, same configurations, different hardware

Measure:
1. Raw memory bandwidth (STREAM) on each platform
2. MoE decode throughput on each platform
3. How much of the bandwidth difference translates to inference speedup?
4. At what batch size does the EMR become compute-bound?
5. Is the AMX performance different? (Granite Rapids may have improved AMX)

Why: This directly answers "is Xeon 6 MRDIMM worth it for MoE inference?"
The answer determines your deployment recommendation.
```

#### P15: Identifying SOTA Improvement Opportunities ★★★★★
```
Task: Synthesize everything you've learned to identify specific kernel
improvements that would measurably improve MoE inference on Xeon 6.
Focus on GLM-4.7-Flash (your profiled model) but generalize to Qwen3-Next.

Deliverables:
1. Ranked list of optimization opportunities by expected impact
2. For each: what kernel to replace, what the current implementation does,
   what your improved version would do, and expected speedup
3. Prototype ONE improvement and measure the actual impact
4. Write a design doc for the highest-impact optimization

This is the capstone. It transitions you from "understanding the system"
to "contributing SOTA improvements."
```

---

## Part 3: The Integration Track — Code Reading Assignments

*These are reading-only exercises. Do them in parallel with the other tracks.*

### Week 1-2: Orientation
```
Read and annotate:
□ vLLM docs: CPU installation page — understand every environment variable
□ vLLM csrc/cpu/ directory — catalog every .cpp file and its purpose
□ vLLM model file for GLM-4.7-Flash:
  vllm/model_executor/models/glm4_moe_lite.py
  — trace how MLA attention and MoE are dispatched to CPU kernels
  — note: this is essentially DeepSeek V3 architecture renamed
□ vLLM model file for Qwen3-Next (likely in vllm/model_executor/models/)
  — trace how Gated DeltaNet + MoE is dispatched (hybrid attention)
□ IPEX MoE module: find ipex.llm.modules.GatedMLPMOE source code
  — understand what VLLM_CPU_MOE_PREPACK does
□ GLM-4.7-Flash config.json on HuggingFace — understand every field
  (especially q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
   v_head_dim — these define the MLA compression)
```

### Week 3-4: Kernel Deep Dive
```
Read and annotate:
□ oneDNN matmul primitive source: how does it select AMX vs AVX512?
□ vLLM csrc/cpu/attention.cpp — the existing paged attention kernel
  — how does this handle MLA's compressed KV cache?
□ The SGLang blog on Xeon 6 DeepSeek R1 optimization:
  https://lmsys.org/blog/2025-07-14-intel-xeon-optimization/
  — study their MoE kernel design, FP8 emulation, attention fusion
  — note: DeepSeek R1 uses the SAME MLA architecture as GLM-4.7-Flash
□ Gated DeltaNet references (see Appendix A in this document):
  — Start with Raschka's implementation, then FLA fused_recurrent
□ flash-linear-attention repository — reference Gated DeltaNet implementation
  https://github.com/fla-org/flash-linear-attention
  Key file: fla/ops/gated_delta_rule/recurrent.py
```

### Week 5-6: System Integration
```
Read and annotate:
□ vLLM's MoE modular kernel system (docs/design/moe_kernel_features)
  — understand how expert kernels plug into the framework
□ vLLM's CPU model runner — how does continuous batching work on CPU?
□ How does vLLM's scheduler interact with MoE batch formation?
□ oneDNN graph API — how does fusion work at the oneDNN level?
```

### Week 7-8: Research Literature
```
Read:
□ Original Gated DeltaNet paper (arXiv:2412.06464, ICLR 2025)
□ DeltaNet parallelization paper (arXiv:2406.06484, NeurIPS 2024)
□ Songlin Yang's DeltaNet blog (Parts 1 & 2 — see Appendix A)
□ DeepSeek V2 paper — the MLA architecture (same as GLM-4.7-Flash uses)
□ "Online normalizer calculation for softmax" (Milakov & Gimelshein)
□ Intel AMX ISA Programming Reference (for K9-K14)
□ DeepSeek MoE paper — their expert parallelism approach
□ Qwen3-Next technical blog — architecture rationale
□ Sebastian Raschka's Gated DeltaNet walkthrough (see Appendix A)
```

---

## Suggested Weekly Schedule

| Week | Kernel Track | Profiling Track | Integration Track | Feedback Loop |
|------|-------------|----------------|-------------------|---------------|
| 1 | K1-K3 (SIMD basics) | P1-P2 (hw discovery, both machines) | Orientation reads | — |
| 2 | K4-K6 (SiLU, RMSNorm, softmax) | P3 (perf stat w/ GLM-4.7-Flash) | Orientation reads | — |
| 3 | K7-K8 (top-k, scatter/gather) | P4, P4b (VTune + vLLM Perfetto trace) | Kernel deep dive | **REDIRECT #1**: Write memo. Which ops dominate? Reorder K9+ if needed. |
| 4 | K9-K10 (AMX basics) | P5-P6 (OMP binding, expert memory) | Kernel deep dive | **REDIRECT #2**: Expert memory findings → reprioritize K13 vs K8? |
| 5 | K11-K12 (tiled GEMM, INT8) | P7, P7b (prefill vs decode + traces) | System integration | **REDIRECT #3**: Which phase matters? Skip large-M K11 cases if decode-only. |
| 6 | K13-K14 (prepacking, fusion) | P8-P9 (malloc, TP) | System integration | — |
| 7 | K15-K17 (MoE gating, routing, single expert) | P10 (batch sweep) | Research literature | — |
| 8 | K18-K19 (full MoE layer, BF16+INT8) | P11-P12 (load balance, E2E) | Research literature | **REDIRECT #4**: E2E profile vs kernel perf. Is kernel work or system work the bottleneck now? |
| 9 | K20-K21 (DeltaNet, paged attn) | P13 (vLLM vs SGLang) | — | — |
| 10 | K22-K24 (fusion, full layer) | P14-P15 (HW compare, SOTA) | — | — |
| 11 | K25-K26 (MTP, MLA attention) | Prototype first SOTA improvement | — | — |
| 12 | Revisit weakest areas | Prototype continued | — | Final synthesis memo |

**Redirect memo format** (keep these in a running journal):
```
Date: ___
Profiling exercise: P__
Key finding: [one sentence, e.g. "MoE expert matmul is 62% of decode time"]
Implication for kernel track: [what to reprioritize]
Implication for vLLM contribution: [what PR or issue this suggests]
```

---

## After This Curriculum: The Agent/RL Path

Once you complete this curriculum, you'll have:
1. Deep understanding of every kernel in the Qwen3-Next forward pass
2. Ability to profile and identify bottlenecks on Xeon hardware
3. Working implementations of each kernel component

To transition to "agents creating SOTA kernels via RL":

**Constrained and verifiable domains you'll be ready to define:**
- "Improve the BF16 MoE expert GEMM to achieve >X% of peak AMX throughput
   for M=1, K=2048, N=D" — measurable, auto-gradeable
- "Fuse RMSNorm + quantize + scatter to reduce memory traffic by >Y%"
   — measurable via perf counters
- "Find the optimal OMP thread ↔ expert assignment for batch_size=B
   on a 2-socket system" — measurable via throughput

**Each becomes an RL reward signal:**
- Throughput (tok/s) on a fixed benchmark → reward
- Memory bandwidth efficiency (%) → reward
- Latency percentiles → reward

The kernel problems in this curriculum are already in the right format
for an agent: problem spec + test cases + performance metric. You just
need to wrap them in an RL loop where the agent writes C++ intrinsics code,
compiles, runs benchmarks, and receives the throughput as reward.

---

## Key Reference Links

### Models
- GLM-4.7-Flash Model Card: https://huggingface.co/zai-org/GLM-4.7-Flash
- GLM-4.7-Flash config.json: https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json
- Qwen3-Next Model Card: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct

### Frameworks & Kernels
- vLLM CPU Backend: https://docs.vllm.ai/en/stable/getting_started/installation/cpu/
- vLLM Profiling Docs: https://docs.vllm.ai/en/stable/contributing/profiling/
- vLLM Performance Dashboard: linked from profiling docs (public Perfetto traces)
- vLLM MoE Kernel Features: https://docs.vllm.ai/en/latest/design/moe_kernel_features/
- vLLM GLM-4.7-Flash model impl: vllm/model_executor/models/glm4_moe_lite.py
- oneDNN Source: https://github.com/oneapi-src/oneDNN
- IPEX Source: https://github.com/intel/intel-extension-for-pytorch

### Gated DeltaNet References (see Appendix A for details)
- Raschka's From-Scratch: https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet
- Raschka's Blog Walkthrough: https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/
- NVlabs Official: https://github.com/NVlabs/GatedDeltaNet
- FLA Library: https://github.com/fla-org/flash-linear-attention
- DeltaNet Blog Part 1: https://sustcsonglin.github.io/blog/2024/deltanet-1/
- DeltaNet Blog Part 2: https://sustcsonglin.github.io/blog/2024/deltanet-2/
- DeltaNet Paper (NeurIPS 2024): https://arxiv.org/abs/2406.06484
- Gated DeltaNet Paper (ICLR 2025): https://arxiv.org/abs/2412.06464

### Intel Hardware & ISA
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Intel AMX Programming Reference: search "Intel AMX ISA" on Intel developer site
- SGLang Xeon 6 Blog: https://lmsys.org/blog/2025-07-14-intel-xeon-optimization/

### Profiling Tools
- Perfetto Trace Viewer: https://ui.perfetto.dev (for viewing vLLM torch profiler traces)
- vLLM Profiling Example: examples/offline_inference/simple_profiling.py in vLLM repo
- Intel VTune Profiler: free download from Intel oneAPI toolkit

### MLA (Multi-Head Latent Attention)
- DeepSeek V2 Paper (introduces MLA): https://arxiv.org/abs/2405.04434
- llama.cpp GLM-4.7-Flash support (confirmed "just DeepSeek renamed"): https://github.com/ggml-org/llama.cpp/pull/18936

### Problem Format Reference
- Tensara (GPU, for problem format reference): https://tensara.org