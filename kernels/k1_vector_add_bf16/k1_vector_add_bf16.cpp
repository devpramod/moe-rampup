// ================================================================
// K1: Vector Add (BF16)  ★☆☆☆☆
// ================================================================
//
// PROBLEM
// -------
// Implement elementwise addition of two BF16 arrays.
//
//   c[i] = a[i] + b[i]   for all i in [0, n)
//
// BF16 values are stored as uint16_t in memory.
//
// CONSTRAINTS
// -----------
//   - n is a multiple of 32 (no tail handling needed)
//   - Use AVX512-BF16 intrinsics for the SIMD version
//   - Handle the BF16 ↔ FP32 conversion explicitly
//
// PIPELINE (think about this before coding)
// ------------------------------------------
//   Memory:  [bf16 × 32]  = 64 bytes = one 512-bit load
//                 │
//       split into two halves of 16
//              /            \
//     cvtpbh_ps           cvtpbh_ps      ← 16 BF16 → 16 FP32
//          │                   │
//      add_ps              add_ps
//          │                   │
//     cvtneps_pbh         cvtneps_pbh    ← 16 FP32 → 16 BF16
//              \            /
//       combine into 32 BF16
//                 │
//        store 64 bytes
//
// KEY INTRINSICS
// --------------
//   _mm512_loadu_si512    — load 64 bytes (32 BF16 values) as __m512i
//   _mm256_loadu_si256    — load 32 bytes (16 BF16 values) as __m256i
//   _mm512_cvtpbh_ps      — convert 16 BF16 (__m256bh) → 16 FP32 (__m512)
//   _mm512_cvtneps_pbh    — convert 16 FP32 (__m512) → 16 BF16 (__m256bh)
//   _mm512_add_ps         — add 16 FP32 values
//   _mm512_storeu_ps      — store 16 FP32 values
//
// NOTE: __m256bh is the type for 16 packed BF16 values.
//       You can cast __m256i ↔ __m256bh with _mm256_castsi256_bh()
//       or just load directly into the right type.
//
// TEST
// ----
//   Build:  cd kernels && mkdir -p build && cd build && cmake .. && make
//   Run:    ./k1_vector_add_bf16
//
// PERFORMANCE TARGET
// ------------------
//   >80% of peak memory bandwidth (STREAM triad equivalent).
//   On your Xeon 8568Y+ dual-socket with DDR5-5600:
//     Theoretical peak ≈ 600+ GB/s aggregate (both sockets)
//     Single-socket ≈ 300+ GB/s
//     80% target ≈ 240+ GB/s (single socket, for large n)
//
//   Bytes moved per element: read a(2B) + read b(2B) + write c(2B) = 6 bytes
//   GB/s = (n × 6) / (time_in_seconds × 1e9)
//
// ================================================================

#include "../common/common.h"
#include <cstring>
#include <algorithm>

// ================================================================
// Stage 1: SCALAR REFERENCE (provided — your correctness oracle)
// ================================================================
void vector_add_bf16_scalar(const uint16_t* a, const uint16_t* b,
                            uint16_t* c, int n) {
    for (int i = 0; i < n; i++) {
        float fa = bf16_to_fp32(a[i]);
        float fb = bf16_to_fp32(b[i]);
        float fc = fa + fb;
        c[i] = fp32_to_bf16(fc);
    }
}

// ================================================================
// Stage 2: AVX512 FP32 version (warm-up — no BF16 conversion)
//          Operates on float arrays, not BF16.
//          Purpose: get comfortable with _mm512 load/add/store
// ================================================================
void vector_add_fp32_avx512(const float* a, const float* b,
                            float* c, int n) {
    // TODO: Implement this first as a warm-up
    // Process 16 floats per iteration using:
    //   __m512 va = _mm512_loadu_ps(a + i);
    //   __m512 vb = _mm512_loadu_ps(b + i);
    //   __m512 vc = _mm512_add_ps(va, vb);
    //   _mm512_storeu_ps(c + i, vc);

    // REMOVE THIS FALLBACK ONCE YOU IMPLEMENT:
    // for (int i = 0; i < n; i++) {
    //     c[i] = a[i] + b[i];
    // }
    for (int i = 0; i < n ; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(c + i, vc);
    }
    
}

// ================================================================
// Stage 3: AVX512 BF16 version (the actual K1 solution)
//          Operates on uint16_t arrays (BF16 packed).
// ================================================================
void vector_add_bf16_avx512(const uint16_t* a, const uint16_t* b,
                            uint16_t* c, int n) {
    // TODO: YOUR IMPLEMENTATION HERE
    //
    // Hints:
    //   1. Loop in steps of 32 (32 BF16 values = 64 bytes = one __m512i)
    //      OR loop in steps of 16 (16 BF16 = 32 bytes = one __m256i / __m256bh)
    //
    //   2. For each chunk of 16 BF16 values:
    //      a) Load 16 BF16 from a:  __m256i raw_a = _mm256_loadu_si256(...)
    //      b) Cast to BF16 type:    __m256bh bh_a = (__m256bh)raw_a
    //         OR directly:          __m256bh bh_a = _mm256_loadu_si256(...)  cast
    //      c) Convert to FP32:      __m512 fp_a = _mm512_cvtpbh_ps(bh_a)
    //      d) Repeat for b
    //      e) Add:                  __m512 fp_c = _mm512_add_ps(fp_a, fp_b)
    //      f) Convert back to BF16: __m256bh bh_c = _mm512_cvtneps_pbh(fp_c)
    //      g) Store:                _mm256_storeu_si256((__m256i*)(c+i), (__m256i)bh_c)
    //
    //   3. The __m256bh ↔ __m256i casts might need reinterpret_cast or
    //      C-style casts depending on your compiler. GCC 11+ handles this.
    for (int i = 0; i < n; i+= 16) {
        __m256i raw_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256bh bh_a = (__m256bh)raw_a;
        __m512 fp_a = _mm512_cvtpbh_ps(bh_a);

        __m256i raw_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256bh bh_b = (__m256bh)raw_b;
        __m512 fp_b = _mm512_cvtpbh_ps(bh_b);

        __m512 fp_c = _mm512_add_ps(fp_a, fp_b);
        __m256bh bh_c = _mm512_cvtneps_pbh(fp_c);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), (__m256i)bh_c);
         
    }

    // REMOVE THIS FALLBACK ONCE YOU IMPLEMENT:
    //vector_add_bf16_scalar(a, b, c, n);
}

// ================================================================
// Verification: compare two BF16 result arrays
// ================================================================
bool verify_bf16(const uint16_t* ref, const uint16_t* test, int n,
                 const char* label) {
    int errors = 0;
    float max_rel_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float fref  = bf16_to_fp32(ref[i]);
        float ftest = bf16_to_fp32(test[i]);
        float abs_err = fabsf(fref - ftest);
        float rel_err = (fref != 0.0f) ? abs_err / fabsf(fref) : abs_err;
        max_rel_err = std::max(max_rel_err, rel_err);

        if (ref[i] != test[i]) {  // BF16 add should be bit-exact vs scalar
            errors++;
            if (errors <= 5) {
                printf("    MISMATCH at i=%d: ref=0x%04x (%.6f) vs test=0x%04x (%.6f)\n",
                       i, ref[i], fref, test[i], ftest);
            }
        }
    }

    if (errors == 0) {
        printf("  ✓ %-20s PASS  (max rel err = %.2e)\n", label, max_rel_err);
    } else {
        printf("  ✗ %-20s FAIL  (%d / %d mismatches)\n", label, errors, n);
    }
    return errors == 0;
}

// ================================================================
// Benchmark: measure GB/s throughput
// ================================================================
double benchmark_bf16(void (*fn)(const uint16_t*, const uint16_t*, uint16_t*, int),
                      const uint16_t* a, const uint16_t* b, uint16_t* c, int n,
                      int warmup_iters, int bench_iters) {
    // Warmup — fill caches, stabilize CPU frequency
    for (int i = 0; i < warmup_iters; i++) {
        fn(a, b, c, n);
    }

    Timer t;
    for (int i = 0; i < bench_iters; i++) {
        fn(a, b, c, n);
    }
    double elapsed_ms = t.elapsed_ms();
    double elapsed_s = elapsed_ms / 1000.0;

    // Bytes moved: read a + read b + write c = 3 arrays × n × 2 bytes
    double total_bytes = (double)n * 6.0 * bench_iters;
    double gbps = total_bytes / (elapsed_s * 1e9);

    return gbps;
}

// ================================================================
// Main — runs all stages and reports results
// ================================================================
int main(int argc, char** argv) {
    // Default: 64M elements (128 MB per array in BF16)
    // This is large enough to exceed all cache levels and measure DRAM bandwidth.
    int n = (argc > 1) ? atoi(argv[1]) : 64 * 1024 * 1024;

    // Ensure n is a multiple of 32
    n = (n / 32) * 32;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  K1: Vector Add (BF16)                                 ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("  Array size:     %d elements (%.1f MB per array in BF16)\n",
           n, (double)n * 2.0 / (1024 * 1024));
    printf("  Total data:     %.1f MB (3 arrays × read a + read b + write c)\n",
           (double)n * 6.0 / (1024 * 1024));
    printf("\n");

    // ── Allocate aligned arrays ──────────────────────────────────
    uint16_t* a   = (uint16_t*)aligned_alloc_64(n * sizeof(uint16_t));
    uint16_t* b   = (uint16_t*)aligned_alloc_64(n * sizeof(uint16_t));
    uint16_t* c_ref   = (uint16_t*)aligned_alloc_64(n * sizeof(uint16_t));
    uint16_t* c_avx   = (uint16_t*)aligned_alloc_64(n * sizeof(uint16_t));

    // ── Fill with random BF16 values ─────────────────────────────
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int i = 0; i < n; i++) {
        a[i] = fp32_to_bf16(dist(rng));
        b[i] = fp32_to_bf16(dist(rng));
    }

    // ── Stage 1: Scalar reference ────────────────────────────────
    print_separator();
    printf("  Stage 1: Scalar Reference\n");
    print_separator();

    vector_add_bf16_scalar(a, b, c_ref, n);

    {
        int warmup = 3, iters = 5;
        double gbps = benchmark_bf16(vector_add_bf16_scalar, a, b, c_ref, n,
                                     warmup, iters);
        print_result("Scalar throughput:", gbps, "GB/s");
    }

    // ── Stage 2: AVX512 FP32 (warm-up) ──────────────────────────
    print_separator();
    printf("  Stage 2: AVX512 FP32 (warm-up)\n");
    print_separator();

    {
        // Allocate FP32 arrays, fill from the same BF16 data
        float* fa = (float*)aligned_alloc_64(n * sizeof(float));
        float* fb = (float*)aligned_alloc_64(n * sizeof(float));
        float* fc_ref = (float*)aligned_alloc_64(n * sizeof(float));
        float* fc_avx = (float*)aligned_alloc_64(n * sizeof(float));

        for (int i = 0; i < n; i++) {
            fa[i] = bf16_to_fp32(a[i]);
            fb[i] = bf16_to_fp32(b[i]);
            fc_ref[i] = fa[i] + fb[i];
        }

        vector_add_fp32_avx512(fa, fb, fc_avx, n);

        // Verify
        int errors = 0;
        for (int i = 0; i < n; i++) {
            if (fc_ref[i] != fc_avx[i]) {
                errors++;
                if (errors <= 5)
                    printf("    MISMATCH at i=%d: ref=%.6f vs avx=%.6f\n",
                           i, fc_ref[i], fc_avx[i]);
            }
        }
        if (errors == 0)
            printf("  ✓ %-20s PASS\n", "AVX512 FP32");
        else
            printf("  ✗ %-20s FAIL  (%d / %d mismatches)\n", "AVX512 FP32", errors, n);

        // Benchmark: 3 arrays × n × 4 bytes (FP32)
        for (int w = 0; w < 5; w++) vector_add_fp32_avx512(fa, fb, fc_avx, n);
        Timer t2;
        int iters = 20;
        for (int it = 0; it < iters; it++) vector_add_fp32_avx512(fa, fb, fc_avx, n);
        double elapsed_s = t2.elapsed_ms() / 1000.0;
        double gbps = (double)n * 12.0 * iters / (elapsed_s * 1e9);  // 3 arrays × 4 bytes
        print_result("AVX512 FP32 throughput:", gbps, "GB/s");

        free(fa); free(fb); free(fc_ref); free(fc_avx);
    }

    // ── Stage 3: AVX512 BF16 ─────────────────────────────────────
    print_separator();
    printf("  Stage 3: AVX512 BF16\n");
    print_separator();

    memset(c_avx, 0, n * sizeof(uint16_t));
    vector_add_bf16_avx512(a, b, c_avx, n);
    verify_bf16(c_ref, c_avx, n, "AVX512 BF16");

    {
        int warmup = 5, iters = 20;
        double gbps = benchmark_bf16(vector_add_bf16_avx512, a, b, c_avx, n,
                                     warmup, iters);
        print_result("AVX512 BF16 throughput:", gbps, "GB/s");
    }

    // ── Summary ──────────────────────────────────────────────────
    printf("\n");
    print_separator();
    printf("  Performance Target: >80%% of peak memory bandwidth\n");
    printf("  Your Xeon 8568Y+ (EMR): ~300 GB/s single-socket\n");
    printf("  80%% target: ~240 GB/s\n");
    print_separator();
    printf("\n");
    printf("  NEXT STEPS:\n");
    printf("  1. Implement vector_add_fp32_avx512() as warm-up (optional)\n");
    printf("  2. Implement vector_add_bf16_avx512() — the real exercise\n");
    printf("  3. Try different array sizes:  ./k1_vector_add_bf16 1048576\n");
    printf("  4. Once passing, try: vary n from 1K to 256M to see\n");
    printf("     when you transition from cache-bound to DRAM-bound\n");
    printf("\n");

    free(a);
    free(b);
    free(c_ref);
    free(c_avx);

    return 0;
}
