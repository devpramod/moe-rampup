cmake_minimum_required(VERSION 3.16)

# Prefer GCC 12.3+ for AVX512-BF16 intrinsic support in K1.
# This is set before project() so CMake can pick the compiler early.
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "/usr/bin/g++-12")
	set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
endif()

project(moe_kernels LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.3")
	message(FATAL_ERROR "GCC >= 12.3 is required for this project (current: ${CMAKE_CXX_COMPILER_VERSION}).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Compiler flags ---
# AVX512 + BF16 + AMX support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -march=native -fopenmp")

# --- K1: Vector Add BF16 ---
add_executable(k1_vector_add_bf16 k1_vector_add_bf16/k1_vector_add_bf16.cpp)
target_include_directories(k1_vector_add_bf16 PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_options(k1_vector_add_bf16 PRIVATE
	-mavx512f
	-mavx512vl
	-mavx512bf16
)

# --- K2: Vector FMA ---
add_executable(k2_vector_fma k2_vector_fma/k2_vector_fma.cpp)
target_include_directories(k2_vector_fma PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_options(k2_vector_fma PRIVATE
	-mavx512f
)

# --- K3: Horizontal Sum / Reduction ---
add_executable(k3_reduction k3_reduction/k3_reduction.cpp)
target_include_directories(k3_reduction PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_options(k3_reduction PRIVATE
	-mavx512f
)

add_executable(k3_reduction_unroll_jam k3_reduction/k3_reduction_unroll_jam.cpp)
target_include_directories(k3_reduction_unroll_jam PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_options(k3_reduction_unroll_jam PRIVATE
	-mavx512f
)

# --- K4: SiLU Activation ---
add_executable(k4_silu k4_silu/k4_silu.cpp)
target_include_directories(k4_silu PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_options(k4_silu PRIVATE
	-mavx512f
)

# --- K5: RMSNorm ---
add_executable(k5_rmsnorm k5_rmsnorm/k5_rmsnorm.cpp)
target_include_directories(k5_rmsnorm PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_options(k5_rmsnorm PRIVATE
	-mavx512f
)

add_executable(k5_rmsnorm_newton k5_rmsnorm/k5_rmsnorm_newton.cpp)
target_include_directories(k5_rmsnorm_newton PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_options(k5_rmsnorm_newton PRIVATE
	-mavx512f
)

# --- K6: Softmax ---
add_executable(k6_softmax k6_softmax/k6_softmax.cpp)
target_include_directories(k6_softmax PRIVATE ${CMAKE_SOURCE_DIR}/common)
target_compile_options(k6_softmax PRIVATE
	-mavx512f
)
