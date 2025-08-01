// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-gemm-minmax-fp32.yaml
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



#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane)

  static void qu8_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neonv8_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane)

  static void qu8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane)

  static void qu8_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane)

  static void qu8_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane)

  static void qu8_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_fp32_neon_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM
  static void qu8_gemm_minmax_fp32_ukernel_1x1c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x1c4__armsimd32,
      xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/1, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_v6);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x1c4__armsimd32)

  static void qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32,
      xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_v6);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32)

  static void qu8_gemm_minmax_fp32_ukernel_2x1c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x1c4__armsimd32,
      xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/1, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_v6);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x1c4__armsimd32)

  static void qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32,
      xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_v6);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32)
#endif  // XNN_ARCH_ARM


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx)

  static void qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx)

  static void qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx)

  static void qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx)

  static void qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm)

  static void qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm)

  static void qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm)

  static void qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qu8_gemm_minmax_fp32_ukernel_1x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx256skx,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x8c8__avx256skx)
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2)

  static void qu8_gemm_minmax_fp32_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x8c8__avx2)

  static void qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2)

  static void qu8_gemm_minmax_fp32_ukernel_4x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x8c8__avx2)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128)

  static void qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_scalar_params,
      xnn_pack_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_2x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_3x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x2__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_3x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_4x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x2__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf)

static void qu8_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

static void qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic)

static void qu8_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_params,
    xnn_pack_qu8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qu8_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
