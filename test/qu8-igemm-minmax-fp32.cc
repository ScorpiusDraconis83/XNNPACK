// clang-format off
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-igemm-minmax-fp32.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack-lh.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/ppmm.h"
#include "src/xnnpack/requantization.h"
#include "test/gemm-microkernel-tester.h"
#include "test/next_prime.h"

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


namespace {

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
std::vector<GemmTestParams> CreateTests1(
    size_t k_block, size_t adj_k_block,
    ConstantOrFunction mr, ConstantOrFunction nr, size_t kr, size_t sr,
    bool is_igemm,
    bool unsigned_inputs,
    uint8_t planes,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    uint64_t arch_flags = 0) {
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  std::string nrs = std::to_string(nr);

  const GemmMicrokernelTester tester = GemmMicrokernelTester()
      .mr(mr).nr(nr).kr(kr).sr(sr).unsigned_inputs(unsigned_inputs).planes(planes);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
      , test_func, arch_flags));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .a_stride(xnnpack::NextPrime(k_block + 1))
        , test_func, arch_flags));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block)
      , test_func, arch_flags)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block)
      , test_func, arch_flags)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block)
      , test_func, arch_flags)
      .loop_n(1, nr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs,
        tester.clone()
            .m(mr).n(nr)
        , test_func, arch_flags)
        .loop_k(1, adj_k_block - 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(adj_k_block + 1))
          , test_func, arch_flags)
          .loop_k(1, adj_k_block - 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs + "_subtile",
        tester.clone()
        , test_func, arch_flags)
        .loop_k(1, adj_k_block - 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs,
      tester.clone()
          .m(mr).n(nr)
      , test_func, arch_flags)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr)
            .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
      , test_func, arch_flags)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs + "_subtile",
      tester.clone()
      , test_func, arch_flags)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs,
        tester.clone()
            .m(mr).n(nr)
        , test_func, arch_flags)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
          , test_func, arch_flags)
          .loop_k(adj_k_block + k_block, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs + "_subtile",
        tester.clone()
        , test_func, arch_flags)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs,
      tester.clone()
          .m(mr)
      , test_func, arch_flags)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
        , test_func, arch_flags)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_subtile",
      tester.clone()
      , test_func, arch_flags)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs,
      tester.clone()
          .m(mr)
      , test_func, arch_flags)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
        , test_func, arch_flags)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_subtile",
      tester.clone()
      , test_func, arch_flags)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "small_kernel",
        tester.clone()
            .m(mr).n(nr).ks(3)
        , test_func, arch_flags)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "small_kernel_subtile",
        tester.clone()
            .ks(3)
        , test_func, arch_flags)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
        , test_func, arch_flags)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
        , test_func, arch_flags)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "strided_cm_subtile",
      tester.clone()
          .mr(mr).nr(nr).kr(kr).sr(sr)
          .cm_stride(xnnpack::NextPrime(nr + 1))
      , test_func, arch_flags)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "a_offset",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
        , test_func, arch_flags)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "zero",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
        , test_func, arch_flags)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_zi(0, mr - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "qmin",
      tester.clone()
          .m(mr).n(nr).k(k_block).qmin(128)
      , test_func, arch_flags));
  gemm_tests.push_back(GemmTestParams(
      "qmax",
      tester.clone()
          .m(mr).n(nr).k(k_block).qmax(128)
      , test_func, arch_flags));
  gemm_tests.push_back(GemmTestParams(
      "strided_cm",
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .cm_stride(xnnpack::NextPrime(nr + 1))
      , test_func, arch_flags));
  gemm_tests.push_back(GemmTestParams(
      "no_a_zero_point",
      tester.clone()
          .m(mr).n(nr).a_zero_point(0)
      , test_func, arch_flags)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "no_b_zero_point",
      tester.clone()
          .m(mr).n(nr).b_zero_point(0)
      , test_func, arch_flags)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "b_zero_point",
      tester.clone()
          .m(mr).n(nr).k(k_block)
      , test_func, arch_flags)
      .loop_bzp(0, 255));
  gemm_tests.push_back(GemmTestParams(
      "no_zero_point",
      tester.clone()
          .m(mr).n(nr)
          .a_zero_point(0)
          .b_zero_point(0)
      , test_func, arch_flags)
      .loop_k(1, k_block * 3, k_block + 1));

  return gemm_tests;
}

}  // namespace


#if XNN_ARCH_ARM
  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X1C4__ARMSIMD32, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/1, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x1c4__armsimd32,
                        xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_arm_v6)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X1C4__ARMSIMD32, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/2, /*nr=*/1, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x1c4__armsimd32,
                        xnn_init_qu8_conv_minmax_fp32_armsimd32_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_arm_v6)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane,
                        xnn_init_qu8_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x16__neon_mlal_lane,
                        xnn_init_qu8_conv_minmax_fp32_neon_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X16__NEONV8_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane,
                        xnn_init_qu8_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X4C2__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C2__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__sse41_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C2S4__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2S4__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2S4__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2S4__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2S4__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X4C8__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X4C8__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C8__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C8__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X4C8__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C8__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C8__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C2__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__avx_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__avx_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__avx_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__avx_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X4C2S4__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C2S4__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2S4__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2S4__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2S4__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C8__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C8__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__avx_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X4C8__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C8__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx2,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x8c8__avx2,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X16C8__AVX512SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_7X16C8__AVX512SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_7x16c8__avx512skx,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X16C8__AVX512SKX_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_5X16C8__AVX512SKX_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_7X16C8__AVX512SKX_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_8X16C8__AVX512SKX_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X4C2__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2S4__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C2S4__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QU8_IGEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128,
                        xnn_init_qu8_conv_minmax_fp32_scalar_params,
                        xnn_pack_qu8_conv_goki_w,
                        xnn_qu8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_1X2__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_1X4__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_1X4__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_2X2__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_fmagic,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_2X2__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_lrintf,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_2X4__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_2x4__scalar_lrintf,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_3X2__SCALAR_IMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x2__scalar_imagic,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_3X4__SCALAR_IMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/3, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_imagic,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_4X2__SCALAR_IMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x2__scalar_imagic,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QU8_IGEMM_MINMAX_FP32_4X4__SCALAR_IMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qu8_igemm_minmax_fp32_ukernel_4x4__scalar_imagic,
                      xnn_init_qu8_conv_minmax_fp32_scalar_params,
                      xnn_pack_qu8_conv_goki_w,
                      xnn_qu8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });

