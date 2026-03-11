#include "../common/common.h"
#include <algorithm>
#include <vector>

// ================================================================
// K6: Softmax (Numerically Stable)  ★★★☆☆
// ================================================================
//
// Problem:
//   out[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))
//
// Constraints:
//   - Numerically stable (subtract max)
//   - Start with 3-pass reference path
//   - n is often 512 for MoE gate logits
//
// Scaffold intent:
//   - Keep scalar reference as correctness oracle.
//   - Leave AVX512 TODO path for your implementation.
//   - Keep validation + benchmark harness ready.

void softmax_scalar(const float* x, float* out, int n) {
    float max_val = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        out[i] = std::exp(x[i] - max_val);
        sum += out[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        out[i] *= inv_sum;
    }
}

void softmax_avx512(const float* x, float* out, int n) {
    // TODO(K6): Implement AVX512 softmax.
    //
    // Suggested baseline plan (3-pass):
    //   1) Max reduction:
    //      - __m512 vmax = set1(-INF)
    //      - vmax = max(vmax, load(x+i))
    //      - scalar max tail
    //
    //   2) Exp + sum:
    //      - compute t = x - max
    //      - exp(t) (you can temporarily use scalar expf per lane like K4 scaffold,
    //        then replace with vectorized approximation)
    //      - store tmp/out and accumulate sum
    //
    //   3) Normalize:
    //      - out *= (1.0f / sum)
    //
    // Optional next step:
    //   - online softmax (2-pass) once 3-pass AVX512 is stable.

    //softmax_scalar(x, out, n);

    // Polynomial coefficients for 2^f approximation
    const auto c0 = _mm512_set1_ps(1.0f);
    const auto c1 = _mm512_set1_ps(1.0f);
    const auto c2 = _mm512_set1_ps(0.5f);
    const auto c3 = _mm512_set1_ps(0.16666667f);
    const auto c4 = _mm512_set1_ps(0.04166667f);
    const auto c5 = _mm512_set1_ps(0.008333333f);

    __m512 vmax = _mm512_set1_ps(-INFINITY);
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        vmax = _mm512_max_ps(vmax, vx);
    }
    float max_val = _mm512_reduce_max_ps(vmax);

    for (; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    vmax = _mm512_set1_ps(max_val);
    __m512 vsum = _mm512_setzero_ps();
    i = 0;
    for (; i <= n - 16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        //__m512 vout = _mm512_loadu_ps(out + i);
        __m512 vx_max_sub = _mm512_sub_ps(vx, vmax);

        // === Phase 1: Range Reduction ===
        auto vlinv = _mm512_set1_ps(1.4426950408f);        // 1/ln2
        auto vn = _mm512_roundscale_ps(
            _mm512_mul_ps(vx_max_sub, vlinv),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
        );
        auto vl = _mm512_set1_ps(0.6931471805f);           // ln2
        auto f = _mm512_fnmadd_ps(vn, vl, vx_max_sub);        // f = -(n*ln2) + (-x)

        // === Phase 2: Polynomial approximation of 2^f (Horner's method) ===
        auto p5 = _mm512_fmadd_ps(f, c5, c4);
        auto p4 = _mm512_fmadd_ps(f, p5, c3);
        auto p3 = _mm512_fmadd_ps(f, p4, c2);
        auto p2 = _mm512_fmadd_ps(f, p3, c1);
        auto p1 = _mm512_fmadd_ps(f, p2, c0);


        // alignas(64) float lane_x[16];
        // alignas(64) float lane_exp[16];
        // _mm512_store_ps(lane_x, vx_max_sub);
        // for (int lane = 0; lane < 16; ++lane) {
        //     lane_exp[lane] = std::exp(lane_x[lane]);
        // }
        // __m512 vexp = _mm512_loadu_ps(lane_exp);

        // === Phase 3: Reconstruction ===
        auto vexp = _mm512_scalef_ps(p1, vn);          // exp(x) = 2^f * 2^n
        _mm512_storeu_ps(out + i, vexp);

        vsum = _mm512_add_ps(vsum, vexp);

    }

    float sum = _mm512_reduce_add_ps(vsum);

    for (; i < n; ++i) {
        out[i] = std::exp(x[i] - max_val);
        sum += out[i];
    }

    i=0;
    float inv_sum = 1.0f / sum;
    __m512 vinv_sum = _mm512_set1_ps(inv_sum);
    for (; i <= n - 16; i += 16) {
        __m512 vout = _mm512_loadu_ps(out + i);
        __m512 vout_norm = _mm512_mul_ps(vout, vinv_sum);
        _mm512_storeu_ps(out + i, vout_norm);
    }

    for (; i < n; ++i) {
        out[i] *= inv_sum;
    }

}

bool verify_softmax(const float* ref, const float* test, int n,
                    float rtol, float atol, const char* label) {
    int errors = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    float sum_ref = 0.0f;
    float sum_test = 0.0f;

    for (int i = 0; i < n; ++i) {
        float abs_err = std::fabs(ref[i] - test[i]);
        float rel_err = abs_err / std::max(std::fabs(ref[i]), 1e-12f);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_ref += ref[i];
        sum_test += test[i];

        if (abs_err > (atol + rtol * std::fabs(ref[i]))) {
            errors++;
            if (errors <= 5) {
                printf("    MISMATCH at i=%d: ref=%.8e vs test=%.8e (abs=%.3e rel=%.3e)\n",
                       i, ref[i], test[i], abs_err, rel_err);
            }
        }
    }

    float sum_diff = std::fabs(sum_ref - sum_test);
    bool sum_ok = sum_diff < 1e-5f;

    if (errors == 0 && sum_ok) {
        printf("  ✓ %-20s PASS  (max abs=%.2e, max rel=%.2e, sum diff=%.2e)\n",
               label, max_abs_err, max_rel_err, sum_diff);
        return true;
    }

    printf("  ✗ %-20s FAIL  (%d mismatches, sum diff=%.2e)\n", label, errors, sum_diff);
    return false;
}

double benchmark_softmax(void (*fn)(const float*, float*, int),
                        const float* x, float* out, int n,
                        int warmup_iters, int bench_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        fn(x, out, n);
    }

    Timer t;
    for (int i = 0; i < bench_iters; ++i) {
        fn(x, out, n);
    }
    double elapsed_s = t.elapsed_ms() / 1000.0;

    // Logical traffic model (3-pass style):
    // pass1: read x            -> 4B
    // pass2: read x + write out-> 8B
    // pass3: read/write out    -> 8B
    // total ~20B per element
    double total_bytes = static_cast<double>(n) * 20.0 * bench_iters;
    return total_bytes / (elapsed_s * 1e9);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 512;
    n = std::max(16, n);
    n = (n / 16) * 16;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  K6: Softmax (Numerically Stable)                      ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("  n:              %d\n", n);
    printf("  Data per call:  %.2f KB logical (3-pass model)\n\n",
           static_cast<double>(n) * 20.0 / 1024.0);

    float* x = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
    float* out_ref = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
    float* out_avx = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));

    std::mt19937 rng(2028);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    for (int i = 0; i < n; ++i) {
        x[i] = dist(rng);
    }

    print_separator();
    printf("  Correctness\n");
    print_separator();

    softmax_scalar(x, out_ref, n);
    softmax_avx512(x, out_avx, n);
    verify_softmax(out_ref, out_avx, n, 1e-5f, 1e-6f, "AVX512 Softmax");

    print_separator();
    printf("  Performance\n");
    print_separator();

    {
        double gbps = benchmark_softmax(softmax_scalar, x, out_ref, n, 10, 10000);
        print_result("Scalar throughput:", gbps, "GB/s");
    }
    {
        double gbps = benchmark_softmax(softmax_avx512, x, out_avx, n, 10, 10000);
        print_result("AVX512 throughput:", gbps, "GB/s");
    }

    printf("\n");
    printf("  YOUR TODO (K6):\n");
    printf("  1. Implement softmax_avx512() 3-pass stable path\n");
    printf("  2. Keep softmax sum close to 1 and match scalar tolerance\n");
    printf("  3. Try n=512 and n=4096, then consider online softmax\n\n");

    free(x);
    free(out_ref);
    free(out_avx);
    return 0;
}
