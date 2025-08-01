// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qc4w-gemm-minmax.yaml
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



#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/32, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ENABLE_AVX256VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnnigfni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni_prfm)
#endif  // XNN_ENABLE_AVX256VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm)
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx256skx,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx256skx)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx256skx,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx256skx)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx256skx,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx256skx)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx)
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4uw_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane_prfm)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane)

  static void qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane_prfm,
      xnn_init_f16_qc4w_minmax_scalar_params,
      xnn_pack_qs8_qc4w_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_fp16_arith);
  }

  BENCHMARK_GEMM(qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane_prfm)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
