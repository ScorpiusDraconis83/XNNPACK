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
//   Specification: test/qd8-f32-qb4w-gemm-minmax.yaml
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
          .b_zero_point(8)
          .bl(32)
      , test_func, arch_flags));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .a_stride(xnnpack::NextPrime(k_block + 1))
            .b_zero_point(8)
            .bl(32)
        , test_func, arch_flags));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block)
          .b_zero_point(8)
          .bl(32)
      , test_func, arch_flags)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block)
          .b_zero_point(8)
          .bl(32)
      , test_func, arch_flags)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block)
          .b_zero_point(8)
          .bl(32)
      , test_func, arch_flags)
      .loop_n(1, nr));
  gemm_tests.push_back(GemmTestParams(
      "bl",
      tester.clone()
          .m(mr).n(nr).k(k_block * 12)
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_k(k_block, k_block * 12, k_block, LoopStepType::Linear)
      .loop_bl(32, k_block * 32, 32));

  return gemm_tests;
}

}  // namespace


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QB4W_GEMM_MINMAX_1X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/32,
        /*adj_k_block=*/32,
        /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x2__scalar,
                      xnn_init_f32_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QB4W_GEMM_MINMAX_1X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/32,
        /*adj_k_block=*/32,
        /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar,
                      xnn_init_f32_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QB4W_GEMM_MINMAX_1X8__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/32,
        /*adj_k_block=*/32,
        /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8__scalar,
                      xnn_init_f32_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QB4W_GEMM_MINMAX_2X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/32,
        /*adj_k_block=*/32,
        /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x2__scalar,
                      xnn_init_f32_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QB4W_GEMM_MINMAX_2X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/32,
        /*adj_k_block=*/32,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4__scalar,
                      xnn_init_f32_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QB4W_GEMM_MINMAX_2X8__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/32,
        /*adj_k_block=*/32,
        /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8__scalar,
                      xnn_init_f32_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QB4W_GEMM_MINMAX_4X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/32,
        /*adj_k_block=*/32,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar,
                      xnn_init_f32_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X4C8__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X4C8__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X4C8__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X4C8__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X4C8__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X4C8__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X4C8__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X4C8__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X4C8__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X4C8__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X4C8__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X4C8__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X4C8__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X4C8__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X4C8__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X4C8__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X4C8__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X4C8__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X4C8__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X4C8__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld128,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X4C8__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X4C8__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X4C8__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X4C8__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld64,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_5X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x8c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_5X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_6X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x8c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_6X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16c4__neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__avx2,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__avx2,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__avx2,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__avx2,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X16__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X16__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X16__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X16__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_6X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_6X16__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_2X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/2, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_3X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_4X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_5X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_5X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_5X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_6X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_6X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_6X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/6, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_7X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_7X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_7X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/7, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_8X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_8X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_8X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/8, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_5X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_7X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_8X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_9X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_10X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_12X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_14X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_5X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_7X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_8X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_9X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_10X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_12X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_14X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X16C8__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_5X16C8__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_7X16C8__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_8X16C8__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_9X16C8__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_10X16C8__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_12X16C8__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_14X16C8__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_1X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_5X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_7X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_8X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_9X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_10X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_12X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QB4W_GEMM_MINMAX_14X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

