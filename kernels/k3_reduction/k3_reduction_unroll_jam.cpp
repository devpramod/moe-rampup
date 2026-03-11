#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <immintrin.h>
#include "../common/common.h"

// ----------------------------------------------------------------------------
// Scalar Reference
// ----------------------------------------------------------------------------
float reduce_sum_scalar(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

float reduce_max_scalar(const float* x, int n) {
    float max_val = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    return max_val;
}

// ----------------------------------------------------------------------------
// AVX512 Implementation
// ----------------------------------------------------------------------------
// 16 0  17 1  18 2
// n % 16
//
//
float reduce_sum_avx512(const float* x, int n) {
    // TODO: Implement AVX512 sum reduction
    // __m512 vsum = _mm512_setzero_ps();
    // int i = 0;
    // for (; i <= n - (?); i += (?)) {
    //     __m512 vx = _mm512_loadu_ps(x + (?));
    //     vsum = _mm512_add_ps(vsum, (?));
    // }
    // float sum = _mm512_reduce_add_ps((?));
    // 
    // // Handle tail elements
    // for (; i < n; ++i) {
    //     sum += x[i];
    // }
    // return sum;
    // 1. Create 4 independent accumulators
    __m512 vsum0 = _mm512_setzero_ps();
    __m512 vsum1 = _mm512_setzero_ps();
    __m512 vsum2 = _mm512_setzero_ps();
    __m512 vsum3 = _mm512_setzero_ps();

    int i = 0;
    // 2. Main loop: process 64 elements (4 vectors) at a time
    for (; i <= n - 64; i += 64) {
        vsum0 = _mm512_add_ps(vsum0, _mm512_loadu_ps(x + i));
        vsum1 = _mm512_add_ps(vsum1, _mm512_loadu_ps(x + i + 16));
        vsum2 = _mm512_add_ps(vsum2, _mm512_loadu_ps(x + i + 32));
        vsum3 = _mm512_add_ps(vsum3, _mm512_loadu_ps(x + i + 48));
    }

    // 3. Combine the 4 accumulators into 1
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);

    // 4. Secondary SIMD loop: catch remaining 16-element chunks
    for (; i <= n - 16; i += 16) {
        vsum0 = _mm512_add_ps(vsum0, _mm512_loadu_ps(x + i));
    }

    // 5. Horizontal reduction
    float sum = _mm512_reduce_add_ps(vsum0);

    // 6. Scalar tail: catch the final 0-15 elements
    for (; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

float reduce_max_avx512(const float* x, int n) {
    __m512 vmax0 = _mm512_set1_ps(-INFINITY);
    __m512 vmax1 = _mm512_set1_ps(-INFINITY);
    __m512 vmax2 = _mm512_set1_ps(-INFINITY);
    __m512 vmax3 = _mm512_set1_ps(-INFINITY);

    int i = 0;
    for (; i <= n - 64; i += 64) {
        vmax0 = _mm512_max_ps(vmax0, _mm512_loadu_ps(x + i));
        vmax1 = _mm512_max_ps(vmax1, _mm512_loadu_ps(x + i + 16));
        vmax2 = _mm512_max_ps(vmax2, _mm512_loadu_ps(x + i + 32));
        vmax3 = _mm512_max_ps(vmax3, _mm512_loadu_ps(x + i + 48));
    }

    vmax0 = _mm512_max_ps(vmax0, vmax1);
    vmax2 = _mm512_max_ps(vmax2, vmax3);
    vmax0 = _mm512_max_ps(vmax0, vmax2);

    for (; i <= n - 16; i += 16) {
        vmax0 = _mm512_max_ps(vmax0, _mm512_loadu_ps(x + i));
    }

    float max_val = _mm512_reduce_max_ps(vmax0);

    for (; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }


    return max_val;
}

// ----------------------------------------------------------------------------
// Benchmark & Main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    int n = 10000000; // 10M elements
    if (argc > 1) n = std::atoi(argv[1]);

    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  K3: Horizontal Sum / Reduction                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    size_t bytes = n * sizeof(float);
    std::cout << "  Array size:     " << n << " elements (" << bytes / 1e6 << " MB)\n";

    float* x = (float*)aligned_alloc_64(bytes);

    // Initialize with some values
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f; // Simple value to easily verify sum
    }
    x[n/2] = 100.0f; // Plant a max value

    // Correctness Check
    float sum_ref = reduce_sum_scalar(x, n);
    float sum_avx = reduce_sum_avx512(x, n);
    float max_ref = reduce_max_scalar(x, n);
    float max_avx = reduce_max_avx512(x, n);

    bool sum_ok = std::abs(sum_ref - sum_avx) < 1e-3;
    bool max_ok = max_ref == max_avx;

    std::cout << "────────────────────────────────────────────────────────────\n";
    std::cout << "  Correctness\n";
    std::cout << "────────────────────────────────────────────────────────────\n";
    std::cout << "  Sum AVX512           " << (sum_ok ? "PASS" : "FAIL") << " (Ref: " << sum_ref << ", AVX: " << sum_avx << ")\n";
    std::cout << "  Max AVX512           " << (max_ok ? "PASS" : "FAIL") << " (Ref: " << max_ref << ", AVX: " << max_avx << ")\n\n";

    // Performance
    int iters = 100;
    Timer t;

    t.reset();
    for (int i = 0; i < iters; ++i) {
        volatile float s = reduce_sum_scalar(x, n);
    }
    double scalar_time = t.elapsed_ms() / 1000.0; // Convert to seconds

    t.reset();
    for (int i = 0; i < iters; ++i) {
        volatile float s = reduce_sum_avx512(x, n);
    }
    double avx_time = t.elapsed_ms() / 1000.0; // Convert to seconds

    double data_moved_gb = (double)bytes * iters / 1e9;
    
    std::cout << "────────────────────────────────────────────────────────────\n";
    std::cout << "  Performance\n";
    std::cout << "────────────────────────────────────────────────────────────\n";
    std::cout << "  Scalar throughput:                  " << data_moved_gb / scalar_time << " GB/s\n";
    std::cout << "  AVX512 throughput:                  " << data_moved_gb / avx_time << " GB/s\n\n";

    free(x);
    return 0;
}
