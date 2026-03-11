#include "../common/common.h"
#include <algorithm>

// ================================================================
// K4: SiLU Activation (Vectorized)  ★★☆☆☆
// ================================================================
//
// Problem:
//   y[i] = SiLU(x[i]) = x[i] / (1 + exp(-x[i]))
//
// Scaffold goals:
//   1) Keep scalar reference for correctness.
//   2) Provide AVX512 loop skeleton over 16 FP32 lanes.
//   3) Leave clear TODO points for fast exp approximation.
//
// Notes:
//   - This scaffold intentionally uses a simple per-lane expf fallback inside
//     the AVX512 path so you can iterate safely.
//   - Next step is replacing fallback with:
//       (a) polynomial exp approximation, or
//       (b) SVML _mm512_exp_ps path if toolchain supports it.

inline float silu_scalar_elem(float x) {
    return x / (1.0f + std::exp(-x));
}

void silu_scalar(const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] = silu_scalar_elem(x[i]);
    }
}

void silu_avx512(const float* x, float* y, int n) {
    int i = 0;

    for (; i <= n - 16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);

        // Polynomial coefficients for 2^f approximation
        const auto c0 = _mm512_set1_ps(1.0f);
        const auto c1 = _mm512_set1_ps(1.0f);
        const auto c2 = _mm512_set1_ps(0.5f);
        const auto c3 = _mm512_set1_ps(0.16666667f);
        const auto c4 = _mm512_set1_ps(0.04166667f);
        const auto c5 = _mm512_set1_ps(0.008333333f);

        // TODO(K4): replace this fallback with true vector exp implementation.
        // Option A: polynomial/Cody-Waite exp approximation in AVX512.
        // Option B: SVML _mm512_exp_ps if available in your toolchain.
        // alignas(64) float lane_x[16];
        // alignas(64) float lane_exp_neg[16];
        // _mm512_store_ps(lane_x, vx);
        // for (int lane = 0; lane < 16; ++lane) {
        //     lane_exp_neg[lane] = std::exp(-lane_x[lane]);
        // }
        // __m512 vexp_neg = _mm512_load_ps(lane_exp_neg);

        // __m512 vone = _mm512_set1_ps(1.0f);
        // __m512 vden = _mm512_add_ps(vone, vexp_neg);
        // __m512 vy = _mm512_div_ps(vx, vden);
        // _mm512_storeu_ps(y + i, vy);

        // Negate x once (we want exp(-x))
        auto vneg_x = _mm512_sub_ps(_mm512_setzero_ps(), vx);

        // === Phase 1: Range Reduction ===
        auto vlinv = _mm512_set1_ps(1.4426950408f);        // 1/ln2
        auto vn = _mm512_roundscale_ps(
            _mm512_mul_ps(vneg_x, vlinv),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
        );
        auto vl = _mm512_set1_ps(0.6931471805f);           // ln2
        auto f = _mm512_fnmadd_ps(vn, vl, vneg_x);        // f = -(n*ln2) + (-x)

        // === Phase 2: Polynomial approximation of 2^f (Horner's method) ===
        auto p5 = _mm512_fmadd_ps(f, c5, c4);
        auto p4 = _mm512_fmadd_ps(f, p5, c3);
        auto p3 = _mm512_fmadd_ps(f, p4, c2);
        auto p2 = _mm512_fmadd_ps(f, p3, c1);
        auto p1 = _mm512_fmadd_ps(f, p2, c0);

        // === Phase 3: Reconstruction ===
        auto vexp_neg = _mm512_scalef_ps(p1, vn);          // exp(-x) = 2^f * 2^n

        auto vone = _mm512_set1_ps(1.0f);
        auto vden = _mm512_add_ps(vone, vexp_neg);
        auto vy = _mm512_div_ps(vx, vden);
        _mm512_storeu_ps(y + i, vy);
    }

    for (; i < n; ++i) {
        y[i] = silu_scalar_elem(x[i]);
    }
}

bool verify_fp32(const float* ref, const float* test, int n,
                 float rtol, float atol, const char* label) {
    int errors = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (int i = 0; i < n; ++i) {
        float abs_err = std::fabs(ref[i] - test[i]);
        float rel_err = abs_err / std::max(std::fabs(ref[i]), 1e-12f);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);

        if (abs_err > (atol + rtol * std::fabs(ref[i]))) {
            errors++;
            if (errors <= 5) {
                printf("    MISMATCH at i=%d: ref=%.8f vs test=%.8f (abs=%.3e rel=%.3e)\n",
                       i, ref[i], test[i], abs_err, rel_err);
            }
        }
    }

    if (errors == 0) {
        printf("  ✓ %-20s PASS  (max abs err = %.2e, max rel err = %.2e)\n",
               label, max_abs_err, max_rel_err);
        return true;
    }

    printf("  ✗ %-20s FAIL  (%d / %d mismatches)\n", label, errors, n);
    return false;
}

double benchmark_silu(void (*fn)(const float*, float*, int),
                      const float* x, float* y, int n,
                      int warmup_iters, int bench_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        fn(x, y, n);
    }

    Timer t;
    for (int i = 0; i < bench_iters; ++i) {
        fn(x, y, n);
    }
    double elapsed_s = t.elapsed_ms() / 1000.0;

    // Traffic model: read x (4B) + write y (4B) = 8B/element.
    double total_bytes = static_cast<double>(n) * 8.0 * bench_iters;
    return total_bytes / (elapsed_s * 1e9);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 16 * 1024 * 1024;
    n = (n / 16) * 16;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  K4: SiLU Activation (Vectorized)                      ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("  Array size:     %d elements (%.1f MB per FP32 array)\n",
           n, static_cast<double>(n) * sizeof(float) / (1024.0 * 1024.0));
    printf("  Total data:     %.1f MB logical per iteration (read x + write y)\n\n",
           static_cast<double>(n) * 8.0 / (1024.0 * 1024.0));

    float* x = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
    float* y_ref = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
    float* y_avx = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-12.0f, 12.0f);
    for (int i = 0; i < n; ++i) {
        x[i] = dist(rng);
    }

    print_separator();
    printf("  Correctness\n");
    print_separator();

    silu_scalar(x, y_ref, n);
    silu_avx512(x, y_avx, n);
    verify_fp32(y_ref, y_avx, n, 1e-5f, 1e-6f, "AVX512 SiLU");

    print_separator();
    printf("  Performance\n");
    print_separator();

    {
        double gbps = benchmark_silu(silu_scalar, x, y_ref, n, 2, 8);
        print_result("Scalar throughput:", gbps, "GB/s");
    }
    {
        double gbps = benchmark_silu(silu_avx512, x, y_avx, n, 3, 20);
        print_result("AVX512 throughput:", gbps, "GB/s");
    }

    printf("\n");
    printf("  NEXT STEPS:\n");
    printf("  1. Implement vector exp approximation (AVX512) in silu_avx512()\n");
    printf("  2. Re-check max relative error target (<1e-5)\n");
    printf("  3. Compare speedup vs scalar at multiple n values\n\n");

    free(x);
    free(y_ref);
    free(y_avx);
    return 0;
}
