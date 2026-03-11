// ================================================================
// K2: Vector Multiply-Add (FMA)  ★☆☆☆☆
// ================================================================
//
// PROBLEM
// -------
// Implement: c[i] = a[i] * b[i] + c[i]
//
// CONSTRAINTS
// -----------
//   - Use _mm512_fmadd_ps
//   - n is a multiple of 16
//
// TEST
// ----
//   Build:  cd kernels/build && cmake .. && make -j4
//   Run:    ./k2_vector_fma
//           ./k2_vector_fma 1048576
//
// TARGET
// ------
// Understand achieved GFLOPS and bandwidth behavior vs array size.
//
// ================================================================

#include "../common/common.h"
#include <algorithm>
#include <cstring>

void vector_fma_scalar(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i] + c[i];
    }
}

void vector_fma_avx512(const float* a, const float* b, float* c, int n) {
    // TODO: Implement AVX512 FMA path
    // for (int i = 0; i < n; i += (?)) {
    //     __m512 va = _mm512_loadu_ps(a + (?));
    //     __m512 vb = _mm512_loadu_ps(b + (?));
    //     __m512 vc = _mm512_loadu_ps(c + (?));
    //     vc = _mm512_fmadd_ps(va, vb, vc);
    //     _mm512_storeu_ps(c + (?), vc);
    // }
    for (int i = 0; i < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_loadu_ps(c + i);
        vc = _mm512_fmadd_ps(va, vb, vc);
        _mm512_storeu_ps(c + i, vc);
    }
    //vector_fma_scalar(a, b, c, n);
}

bool verify_fp32(const float* ref, const float* test, int n, const char* label) {
    int errors = 0;
    float max_rel_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float abs_err = fabsf(ref[i] - test[i]);
        float denom = std::max(1.0f, fabsf(ref[i]));
        float rel_err = abs_err / denom;
        max_rel_err = std::max(max_rel_err, rel_err);

        if (rel_err > 1e-5f) {
            errors++;
            if (errors <= 5) {
                printf("    MISMATCH at i=%d: ref=%.8f vs test=%.8f (rel=%.3e)\n",
                       i, ref[i], test[i], rel_err);
            }
        }
    }

    if (errors == 0) {
        printf("  ✓ %-20s PASS  (max rel err = %.2e)\n", label, max_rel_err);
        return true;
    }

    printf("  ✗ %-20s FAIL  (%d / %d mismatches, max rel err = %.2e)\n",
           label, errors, n, max_rel_err);
    return false;
}

double benchmark_fma(void (*fn)(const float*, const float*, float*, int),
                     const float* a, const float* b, float* c, int n,
                     int warmup_iters, int bench_iters,
                     double* out_gflops, double* out_gbps) {
    for (int i = 0; i < warmup_iters; i++) {
        fn(a, b, c, n);
    }

    Timer t;
    for (int i = 0; i < bench_iters; i++) {
        fn(a, b, c, n);
    }
    double elapsed_s = t.elapsed_ms() / 1000.0;

    // FLOPs per element: 2 (1 mul + 1 add)
    double flops = (double)n * 2.0 * bench_iters;
    *out_gflops = flops / (elapsed_s * 1e9);

    // Logical bytes per element: read a(4) + read b(4) + read c(4) + write c(4) = 16
    double bytes = (double)n * 16.0 * bench_iters;
    *out_gbps = bytes / (elapsed_s * 1e9);

    return elapsed_s;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 16 * 1024 * 1024;  // 16M elements
    n = (n / 16) * 16;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  K2: Vector Multiply-Add (FMA)                         ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("  Array size:     %d elements (%.1f MB per array FP32)\n",
           n, (double)n * sizeof(float) / (1024 * 1024));
    printf("  Total data:     %.1f MB logical per iteration\n",
           (double)n * 16.0 / (1024 * 1024));
    printf("\n");

    float* a = (float*)aligned_alloc_64(n * sizeof(float));
    float* b = (float*)aligned_alloc_64(n * sizeof(float));
    float* c_init = (float*)aligned_alloc_64(n * sizeof(float));
    float* c_ref = (float*)aligned_alloc_64(n * sizeof(float));
    float* c_avx = (float*)aligned_alloc_64(n * sizeof(float));

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (int i = 0; i < n; i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
        c_init[i] = dist(rng);
    }

    memcpy(c_ref, c_init, n * sizeof(float));
    memcpy(c_avx, c_init, n * sizeof(float));

    print_separator();
    printf("  Correctness\n");
    print_separator();

    vector_fma_scalar(a, b, c_ref, n);
    vector_fma_avx512(a, b, c_avx, n);
    verify_fp32(c_ref, c_avx, n, "AVX512 FMA");

    print_separator();
    printf("  Performance\n");
    print_separator();

    {
        memcpy(c_ref, c_init, n * sizeof(float));
        double gflops = 0.0, gbps = 0.0;
        benchmark_fma(vector_fma_scalar, a, b, c_ref, n, 2, 8, &gflops, &gbps);
        print_result("Scalar GFLOPS:", gflops, "GFLOP/s");
        print_result("Scalar throughput:", gbps, "GB/s");
    }

    {
        memcpy(c_avx, c_init, n * sizeof(float));
        double gflops = 0.0, gbps = 0.0;
        benchmark_fma(vector_fma_avx512, a, b, c_avx, n, 3, 20, &gflops, &gbps);
        print_result("AVX512 GFLOPS:", gflops, "GFLOP/s");
        print_result("AVX512 throughput:", gbps, "GB/s");
    }

    printf("\n");
    printf("  NEXT STEPS:\n");
    printf("  1. Implement vector_fma_avx512() with _mm512_fmadd_ps\n");
    printf("  2. Compare 1M vs 64M elements\n");
    printf("  3. Observe when behavior becomes memory-bound\n\n");

    free(a);
    free(b);
    free(c_init);
    free(c_ref);
    free(c_avx);

    return 0;
}
