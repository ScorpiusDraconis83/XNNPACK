// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f32-qc4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstdint>
#include <functional>

#include <benchmark/benchmark.h>
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"

namespace {

struct ConstantOrFunction {
  ConstantOrFunction(size_t x) : fn([x]() { return x; }) {}  //NOLINT
  ConstantOrFunction(int x) : fn([x]() { return x; }) {}  //NOLINT
  template <typename Fn>
  ConstantOrFunction(Fn fn) : fn(std::move(fn)) {}  //NOLINT

  std::function<size_t()> fn;

  operator size_t() const { return fn(); }  //NOLINT
};

}  // namespace



#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4v__rvv,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4v__rvv)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4v__rvv,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4v__rvv)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4v__rvv,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4v__rvv)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4v__rvv,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4v__rvv)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x4v__rvv,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x4v__rvv)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x4v__rvv,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x4v__rvv)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x4v__rvv,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x4v__rvv)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x4v__rvv,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x4v__rvv)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld32_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld32_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld32_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld32_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld32_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld32_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld32_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld32_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld64_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld64_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld64_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld64_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld128_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld128_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld128_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld128_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld128_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld128_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld128_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld128_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld32_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld32_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld32_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld32_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld64_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld64_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld64_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld128_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld128_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld128_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128_2)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__neondot)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__neondot)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__neondot)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__neondot)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__neondot)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x8c4__neondot)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__neondot,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x64c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x64c4__avx512amx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x64c4__avx512amx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/16, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/16, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__avx512amx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__avx512amx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x32c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x32c4__avx512amx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x32c4__avx512amx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/16, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/16, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512amx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512amx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512amx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512amx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/16, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/16, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx_prfm)
#endif  // XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm)
#endif  // XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm)
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && XNN_ARCH_X86_64 && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_11x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_11x16c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/11, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_11x16c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/1, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/2, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/3, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/4, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/5, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/6, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/7, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/8, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/9, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/10, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_11x32c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_11x32c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/11, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_11x32c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/1, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/2, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x64c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x64c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/3, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x64c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x64c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x64c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/4, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x64c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x64c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x64c4__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/5, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x64c4__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/1, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x32c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x32c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/2, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x32c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__asm_amd64_avx512vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__asm_amd64_avx512vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512,
      /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__asm_amd64_avx512vnni)
#endif  // XNN_ENABLE_AVX512VNNI && XNN_ARCH_X86_64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni_prfm)
#endif  // XNN_ENABLE_AVX256VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm)
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx256skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx256skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx256skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx256skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx256skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx256skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx)
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__ssse3_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__ssse3_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__ssse3_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__ssse3_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__ssse3_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__ssse3_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__ssse3_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__ssse3_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x4c8__ssse3_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x4c8__ssse3_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x4c8__ssse3_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x4c8__ssse3_madd,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/6, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x4c8__ssse3_madd)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__ssse3_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__ssse3_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__ssse3_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__ssse3_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__ssse3_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__ssse3_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__ssse3_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__ssse3_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_5x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x4c8__ssse3_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_5x4c8__ssse3_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_6x4c8__ssse3_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x4c8__ssse3_madd_prfm,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/6, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_ssse3);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_6x4c8__ssse3_madd_prfm)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld128,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld128)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld64)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld64,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch32_neonmlal_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch32_neonmlal_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch32_neonmlal_ld64_2)

  static void qd8_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch32_neonmlal_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch32_neonmlal_ld64_2,
      xnn_init_f32_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch32_neonmlal_ld64_2)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


static void qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    xnn_pack_qs8_qc4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/1, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    xnn_pack_qs8_qc4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    xnn_pack_qs8_qc4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    xnn_pack_qs8_qc4w_gemm_goi_w,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    xnn_pack_qs8_qc4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    xnn_pack_qs8_qc4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    xnn_pack_qs8_qc4w_gemm_goi_w,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar)

static void qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_qc4w_minmax_scalar_params,
    xnn_pack_qs8_qc4w_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
