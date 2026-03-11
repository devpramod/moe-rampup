#include "../common/common.h"
#include <algorithm>

// ================================================================
// K5: RMSNorm  ★★☆☆☆
// ================================================================
//
// Problem:
//   out[i] = (x[i] / rms) * weight[i]
//   rms = sqrt(mean(x[i]^2) + eps)
//
// Constraints:
//   - hidden_size = 2048 (default for Qwen3-Next)
//   - eps = 1e-6
//   - Use AVX512 in the optimized path
//
// Scaffold intent:
//   - Keep a clean scalar reference as the correctness oracle.
//   - Provide a guided AVX512 TODO block so you can implement it yourself.
//   - Keep benchmark + correctness plumbing ready.

void rmsnorm_scalar(const float* x, const float* weight, float* out,
                    int hidden_size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        sum_sq += x[i] * x[i];
    }

    float mean_sq = sum_sq / static_cast<float>(hidden_size);
    float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

    for (int i = 0; i < hidden_size; ++i) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

void rmsnorm_avx512(const float* x, const float* weight, float* out,
                    int hidden_size, float eps) {
    // TODO(K5): Implement AVX512 two-pass RMSNorm.
    //
    // Suggested steps:
    //   1) First pass (reduction): compute sum(x[i]^2)
    //      - Use __m512 accumulators and _mm512_fmadd_ps(vx, vx, acc)
    //      - Horizontal reduce to scalar sum_sq
    //
    //   2) Compute inv_rms = rsqrt(mean_sq + eps)
    //      - Start with scalar: inv_rms = 1.0f / sqrtf(...)
    //      - Optional later: try _mm512_rsqrt14_ps refinement
    //
    //   3) Second pass: out = x * inv_rms * weight
    //      - Broadcast inv_rms with _mm512_set1_ps
    //      - Vector multiply and store
    //
    // NOTE:
    //   hidden_size is expected to be multiple of 16 in this track (2048),
    //   but keep a scalar tail for robustness.

    //rmsnorm_scalar(x, weight, out, hidden_size, eps);
    __m512 acc = _mm512_setzero_ps();                     
    int i = 0;
    for (; i <= hidden_size -16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        acc = _mm512_fmadd_ps(vx, vx, acc);
    }

    float sum_sq = _mm512_reduce_add_ps(acc);

    for (; i < hidden_size; ++i) {
        sum_sq += x[i] * x[i];
    }
    
    float mean_sq = sum_sq / static_cast<float>(hidden_size);
    float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

    i = 0;
    __m512 vinv_rms = _mm512_set1_ps(inv_rms);
    for (; i <= hidden_size -16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vw = _mm512_loadu_ps(weight+ i);

        auto tmp = _mm512_mul_ps(vinv_rms, vw);
        auto res = _mm512_mul_ps(tmp, vx);

        _mm512_storeu_ps(out + i, res);

    }

    for (; i < hidden_size; ++i) {
        out[i] = x[i] * inv_rms * weight[i];
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

double benchmark_rmsnorm(void (*fn)(const float*, const float*, float*, int, float),
                         const float* x, const float* weight, float* out,
                         int hidden_size, float eps,
                         int warmup_iters, int bench_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        fn(x, weight, out, hidden_size, eps);
    }

    Timer t;
    for (int i = 0; i < bench_iters; ++i) {
        fn(x, weight, out, hidden_size, eps);
    }
    double elapsed_s = t.elapsed_ms() / 1000.0;

    // Two-pass RMSNorm traffic model (logical):
    //   pass1: read x                  -> 4B
    //   pass2: read x + read w + write out -> 12B
    //   total per element: 16B
    double total_bytes = static_cast<double>(hidden_size) * 16.0 * bench_iters;
    return total_bytes / (elapsed_s * 1e9);
}

int main(int argc, char** argv) {
    int hidden_size = (argc > 1) ? std::atoi(argv[1]) : 2048;
    float eps = 1e-6f;

    hidden_size = std::max(16, hidden_size);
    hidden_size = (hidden_size / 16) * 16;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  K5: RMSNorm                                            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("  hidden_size:    %d\n", hidden_size);
    printf("  eps:            %.1e\n", eps);
    printf("  Data per call:  %.2f KB logical (two-pass model)\n\n",
           static_cast<double>(hidden_size) * 16.0 / 1024.0);

    float* x = static_cast<float*>(aligned_alloc_64(hidden_size * sizeof(float)));
    float* weight = static_cast<float*>(aligned_alloc_64(hidden_size * sizeof(float)));
    float* out_ref = static_cast<float*>(aligned_alloc_64(hidden_size * sizeof(float)));
    float* out_avx = static_cast<float*>(aligned_alloc_64(hidden_size * sizeof(float)));

    std::mt19937 rng(2027);
    std::uniform_real_distribution<float> dist_x(-3.0f, 3.0f);
    std::uniform_real_distribution<float> dist_w(0.8f, 1.2f);
    for (int i = 0; i < hidden_size; ++i) {
        x[i] = dist_x(rng);
        weight[i] = dist_w(rng);
    }

    print_separator();
    printf("  Correctness\n");
    print_separator();

    rmsnorm_scalar(x, weight, out_ref, hidden_size, eps);
    rmsnorm_avx512(x, weight, out_avx, hidden_size, eps);
    verify_fp32(out_ref, out_avx, hidden_size, 1e-5f, 1e-6f, "AVX512 RMSNorm");

    print_separator();
    printf("  Performance\n");
    print_separator();

    {
        double gbps = benchmark_rmsnorm(rmsnorm_scalar, x, weight, out_ref,
                                        hidden_size, eps, 3, 3000);
        print_result("Scalar throughput:", gbps, "GB/s");
    }
    {
        double gbps = benchmark_rmsnorm(rmsnorm_avx512, x, weight, out_avx,
                                        hidden_size, eps, 3, 3000);
        print_result("AVX512 throughput:", gbps, "GB/s");
    }

    printf("\n");
    printf("  YOUR TODO (K5):\n");
    printf("  1. Implement rmsnorm_avx512() two-pass vector path\n");
    printf("  2. Keep accuracy within tolerance vs scalar\n");
    printf("  3. Compare speedup at hidden_size=2048 and 4096\n\n");

    free(x);
    free(weight);
    free(out_ref);
    free(out_avx);
    return 0;
}
