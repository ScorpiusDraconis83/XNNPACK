// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-qc8w-gemm-minmax-fp32.yaml
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



#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/9, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x32c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x32c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/10, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x32c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/9, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x64c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x64c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/10, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x64c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/9, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x96c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x96c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/10, /*nr=*/96, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x96c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x128c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x128c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x128c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x128c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x128c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x128c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x128c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x128c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/9, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x128c4__hvx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x128c4__hvx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x128c4__hvx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/10, /*nr=*/128, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_hvx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x128c4__hvx)
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4v__rvv,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4v__rvv)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4v__rvv,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4v__rvv)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x4v__rvv(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x4v__rvv,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/4 * xnn_init_hardware_config()->vlenb / sizeof(int32_t), /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_riscv_vector);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x4v__rvv)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c2__wasmsimd_dot16x2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c2__wasmsimd_dot16x2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c2__wasmsimd_dot16x2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c2s2__wasmsimd_dot16x2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c2s2__wasmsimd_dot16x2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/2, /*sr=*/2,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c2s2__wasmsimd_dot16x2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c2__wasmsimd_dot16x2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c2__wasmsimd_dot16x2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c2__wasmsimd_dot16x2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c2s2__wasmsimd_dot16x2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c2s2__wasmsimd_dot16x2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/2,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c2s2__wasmsimd_dot16x2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_i8mm);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld32_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld32_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld32_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld64_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld64_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld64_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld128_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld128_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld128_2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld128_2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld128_2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__aarch64_neondot_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neondot_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neondot_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__aarch64_neondot_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neondot_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neondot_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__neondot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_dot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld1r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld4r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld4r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_dup)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld1r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld2r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neonv8_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neonv8_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neonv8_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neonv8_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neonv8_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neonv8_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_dup)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld1r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld4r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld4r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld1r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neonv8_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neonv8_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neonv8_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neonv8_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neonv8_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neonv8_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon_v8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mull(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mull,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mull)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_dup)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld4r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld4r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_dup)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld1r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld2r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neon_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neon_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_dup)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld1r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld2r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld4r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld4r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld4r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_dup(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_dup)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld1r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld1r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/2,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neon_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neon_mlal)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_neon);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane_prfm)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/1, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_v6);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_v6);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x1c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x1c4__armsimd32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/1, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_v6);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x1c4__armsimd32)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/2, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_arm_v6);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32)
#endif  // XNN_ARCH_ARM


#if XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512amx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512amx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512amx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512amx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_16x16c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x16c4__avx512amx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/16, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_16x16c4__avx512amx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_16x16c4__avx512amx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x16c4__avx512amx_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/16, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_16x16c4__avx512amx_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x32c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x32c4__avx512amx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x32c4__avx512amx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x32c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x32c4__avx512amx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x32c4__avx512amx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__avx512amx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/16, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__avx512amx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__avx512amx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__avx512amx_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/16, /*nr=*/32, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__avx512amx_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x64c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x64c4__avx512amx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x64c4__avx512amx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x64c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x64c4__avx512amx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x64c4__avx512amx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/16, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/16, /*nr=*/64, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512amx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx_prfm)
#endif  // XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c4__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c4__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c4__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c4__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c4__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c4__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c4__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c4__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni_prfm)
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && XNN_ARCH_X86_64 && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_11x16c4__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_11x16c4__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/11, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_11x16c4__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__asm_amd64_avx512vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_11x16c8__asm_amd64_avx512vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_11x16c8__asm_amd64_avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/11, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_11x16c8__asm_amd64_avx512vnni)
#endif  // XNN_ENABLE_AVX512VNNI && XNN_ARCH_X86_64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx512skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx256vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx256vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx256vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx256vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx256vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx256vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x8c8__avx256vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x8c8__avx256vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x8c8__avx256vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x8c8__avx256vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_12x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x8c8__avx256vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_12x8c8__avx256vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_14x8c8__avx256vnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x8c8__avx256vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_14x8c8__avx256vnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx256vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx256vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx256vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx256vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx256vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx256vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_9x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x8c8__avx256vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_9x8c8__avx256vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_10x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x8c8__avx256vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_10x8c8__avx256vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_12x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x8c8__avx256vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_12x8c8__avx256vnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_14x8c8__avx256vnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x8c8__avx256vnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256vnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_14x8c8__avx256vnni_prfm)
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256skx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx256skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx256skx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx256skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx256skx)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx256skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx256skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx256skx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx256skx)
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNIINT8 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnniint8_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnniint8_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnniint8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnniint8_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnniint8_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnniint8_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnniint8);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnniint8_prfm)
#endif  // XNN_ENABLE_AVXVNNIINT8 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni_prfm)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni_prfm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avxvnni);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni_prfm)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx2);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_avx);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_x86_sse4_1);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/0);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMRELAXEDSIMD
  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c16__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c16__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c16__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c16__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c16__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c16__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c16__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c16__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/4, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c16__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c16__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c16__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c16__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/16, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c16__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmusdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmusdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmusdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmusdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmusdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmusdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmusdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmusdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmusdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmusdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmusdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmusdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmsdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmsdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmsdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmsdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmsdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmsdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmsdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmsdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_gemm_goi_w,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmsdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2_acc2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2_acc2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2_acc2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2_acc2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2_acc2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2_acc2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_usdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2_acc2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2_acc2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2_acc2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2_acc2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2_acc2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2_acc2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2_acc2)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot)

  static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
      xnn_pack_qs8_to_qu8_gemm_goi_w,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      /*arch_flags=*/xnn_arch_wasm_sdot);
  }

  BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_imagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_imagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_imagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_imagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_imagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_imagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic)

static void qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
    xnn_pack_qs8_gemm_goi_w,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
    /*arch_flags=*/0);
}

BENCHMARK_GEMM(qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
