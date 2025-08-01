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
//   Specification: test/qs8-qc8w-igemm-minmax-fp32.yaml
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

  return gemm_tests;
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
std::vector<GemmTestParams> CreateTests2(
    size_t k_block, size_t adj_k_block,
    ConstantOrFunction mr, ConstantOrFunction nr, size_t kr, size_t sr,
    ConstantOrFunction mr_packed,
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
      .mr(mr).nr(nr).kr(kr).sr(sr).mr_packed(mr_packed).unsigned_inputs(unsigned_inputs).planes(planes);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
      , test_func, arch_flags));
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

  return gemm_tests;
}

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  // NOLINTNEXTLINE(clang-diagnostic-unused-function)
  std::vector<GemmTestParams> CreateTests3(
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
    nr = nr * xnn_init_hardware_config()->vlenb / sizeof(int32_t);
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
        .loop_n(nr + 1, nr * 2 - 1, 4)
        .loop_k(1, k_block * 3, k_block + 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs + "_strided_a",
          tester.clone()
              .m(mr)
              .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
          , test_func, arch_flags)
          .loop_n(nr + 1, nr * 2 - 1, 4)
          .loop_k(1, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_subtile",
        tester.clone()
        , test_func, arch_flags)
        .loop_n(nr + 1, nr * 2 - 1, 4)
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
          .loop_n(nr + 1, nr * 2 - 1, 4)
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

    return gemm_tests;
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

}  // namespace


#if XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_7X16C4__AVX512AMX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c4__avx512amx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512amx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_7X32C4__AVX512AMX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/7, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x32c4__avx512amx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512amx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X8C4__ASM_AARCH32_NEONDOT_CORTEX_A55, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X8C4__ASM_AARCH32_NEONDOT_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL_CORTEX_A53_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_CORTEX_A53_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_CORTEX_A55, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X16C4__ASM_AARCH64_NEONDOT_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X1C4__ARMSIMD32, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/2, /*nr=*/1, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x1c4__armsimd32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_v6)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X2C4__ARMSIMD32, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/2, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2c4__armsimd32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_v6)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__neoni8mm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16c8__neoni8mm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD1R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD2R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD2R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C4__NEON_MLAL_LD2R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld2r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C4S2__NEONV8_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/2,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4s2__neonv8_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C8__AARCH64_NEONDOT_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__aarch64_neondot_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C8__NEON_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neon_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C8__NEONDOT_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neondot_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X16__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C2__NEON_MLAL_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_dup,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C2__NEON_MLAL_LD4R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld4r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_dup,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C2S4__NEON_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD1R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld1r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C4__NEONV8_MLAL_LD2R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C4S2__NEON_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/2,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4s2__neon_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16__neon_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X16__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X8__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8__neon_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X8__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__neondot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__neon_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_6X8__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_6X16__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_v8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_6X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16c4__neondot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_8X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c4__neondot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  INSTANTIATE_TEST_SUITE_P(
      PQS8_QC8W_IGEMM_MINMAX_FP32_32X32C4__NEONSME2, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  , /*nr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_nr();
        } else {
          return 0;
        }
      }
  , /*kr=*/4, /*sr=*/1,
          /*mr_packed=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  ,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test_PQS8(xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_x8_pack_lh_ukernel__igemm_neonsme2,
                        xnn_x8_pack_lh_size__igemm_neonsme2,
                        xnn_pack_kai_qs8_conv_goki_w_sme2,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_sme2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X4C2__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X4C2__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X4C2__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X4C2__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X4C2__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X4C2__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X4C2__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__avx_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X4C2__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__avx_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X4C2S4__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4C2S4__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4C2S4__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X4C2S4__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X4C2S4__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X4C2S4__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X4C2S4__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4C8__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X4C8__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4C8__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4C8__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X4C8__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X4C8__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4C8__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X8C8__AVX256SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avx256skx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx256skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X8C8__AVX256SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__avx256skx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx256skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_5X16C8__AVX512SKX_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_12X16C4__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_12x16c4__avx512vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_14X16C4__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_14x16c4__avx512vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_14X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_14x16c4__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_8X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c8__avx512vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_10X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_10x16c8__avx512vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_12X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_12x16c8__avx512vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_7X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_9X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_9x16c8__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_10X8C8__AVX256VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_10x8c8__avx256vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avx256vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__avxvnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avxvnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_6X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c8__avxvnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avxvnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_8X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c8__avxvnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avxvnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNIINT8 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C8__AVXVNNIINT8_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnniint8_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avxvnniint8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_5X8C8__AVXVNNIINT8_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avxvnniint8_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_x86_avxvnniint8)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVXVNNIINT8 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4C2__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4C2S4__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X4C2S4__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X4C2__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X4C2S4__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X16C2__WASMSIMD_DOT16X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c2__wasmsimd_dot16x2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4C16__WASMSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/4, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c16__wasmsdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_sdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C16__WASMUSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c16__wasmusdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_usdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X4C16__WASMSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/4, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c16__wasmsdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_sdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C16__WASMSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c16__wasmsdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_sdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X8C16__WASMUSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c16__wasmusdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_usdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X8C16__WASMUSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c16__wasmusdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_usdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C8__WASMUSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__wasmusdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_usdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X8C8__WASMUSDOT_U2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__wasmusdot_u2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_usdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C8__WASMUSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__wasmusdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_usdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C8__WASMUSDOT_U2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__wasmusdot_u2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_usdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X8C8__WASMUSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__wasmusdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_to_qu8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_usdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X8C8__WASMSDOT_U2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__wasmsdot_u2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_sdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X8C8__WASMSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__wasmsdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_sdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X8C8__WASMSDOT_U2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__wasmsdot_u2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_sdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X16C4__WASMSDOT_U2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_sdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X16C4__WASMSDOT_U2_ACC2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2_acc2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_sdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X16C4__WASMSDOT_U2_ACC2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2_acc2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_wasm_sdot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_IGEMM_MINMAX_FP32_1X2__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                      xnn_pack_qs8_conv_goki_w,
                      xnn_qs8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_IGEMM_MINMAX_FP32_1X2__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                      xnn_pack_qs8_conv_goki_w,
                      xnn_qs8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_IGEMM_MINMAX_FP32_2X2__SCALAR_IMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                      xnn_pack_qs8_conv_goki_w,
                      xnn_qs8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_IGEMM_MINMAX_FP32_2X4__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                      xnn_pack_qs8_conv_goki_w,
                      xnn_qs8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_IGEMM_MINMAX_FP32_2X4__SCALAR_IMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                      xnn_pack_qs8_conv_goki_w,
                      xnn_qs8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_IGEMM_MINMAX_FP32_2X4__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                      xnn_pack_qs8_conv_goki_w,
                      xnn_qs8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_IGEMM_MINMAX_FP32_4X2__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                      xnn_pack_qs8_conv_goki_w,
                      xnn_qs8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_IGEMM_MINMAX_FP32_4X4__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                      xnn_pack_qs8_conv_goki_w,
                      xnn_qs8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_IGEMM_MINMAX_FP32_4X4__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                      xnn_pack_qs8_conv_goki_w,
                      xnn_qs8_requantize_fp32);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X32C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/2, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x32c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_3X32C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x32c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_9X32C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/9, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_9x32c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_10X32C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/10, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_10x32c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X64C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/64, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x64c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_2X64C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/2, /*nr=*/64, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x64c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X96C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/96, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x96c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X96C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/96, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x96c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_9X96C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/9, /*nr=*/96, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_9x96c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X128C4__HVX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/128, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x128c4__hvx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_1X4V__RVV, GemmTest,
      testing::ValuesIn(CreateTests3(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4v__rvv,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_riscv_vector)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_IGEMM_MINMAX_FP32_4X4V__RVV, GemmTest,
      testing::ValuesIn(CreateTests3(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4v__rvv,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_qs8_conv_goki_w,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_riscv_vector)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

