#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <immintrin.h>

// ============================================================
// BF16 helpers — understand these before touching intrinsics
// ============================================================
// BF16 is just the top 16 bits of an IEEE 754 float32.
//
//   FP32:  [1 sign][8 exponent][23 mantissa]  = 32 bits
//   BF16:  [1 sign][8 exponent][ 7 mantissa]  = 16 bits
//
// Conversion is a truncation (FP32→BF16) or a left-shift (BF16→FP32).

inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    // Round to nearest even (add 0x7FFF + bit 16 for tie-breaking)
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_fp32(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ============================================================
// Timing helper
// ============================================================
struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;

    Timer() : start(Clock::now()) {}

    double elapsed_ms() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double elapsed_us() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count();
    }

    void reset() { start = Clock::now(); }
};

// ============================================================
// Aligned allocation (64-byte for cache-line / AVX512 alignment)
// ============================================================
inline void* aligned_alloc_64(size_t bytes) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        fprintf(stderr, "ERROR: posix_memalign failed for %zu bytes\n", bytes);
        exit(1);
    }
    return ptr;
}

// ============================================================
// Print helpers
// ============================================================
inline void print_separator() {
    printf("────────────────────────────────────────────────────────────\n");
}

inline void print_result(const char* label, double value, const char* unit) {
    printf("  %-30s %12.3f %s\n", label, value, unit);
}
