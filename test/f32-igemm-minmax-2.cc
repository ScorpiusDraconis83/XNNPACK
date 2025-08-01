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
//   Specification: test/f32-igemm-minmax.yaml
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
      "min",
      tester.clone()
          .m(mr).n(nr).k(k_block).min(0.0f)
      , test_func, arch_flags));
  gemm_tests.push_back(GemmTestParams(
      "max",
      tester.clone()
          .m(mr).n(nr).k(k_block).max(0.0f)
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
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kb2s,
      tester.clone()
        .m(mr).n(nr).k(k_block * 2)
    , test_func, arch_flags));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block * 2)
            .a_stride(xnnpack::NextPrime(k_block * 2 + 1))
        , test_func, arch_flags));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kb2s + "_subtile",
      tester.clone()
          .k(k_block * 2)
      , test_func, arch_flags)
      .loop_n(1, nr)
      .loop_m(1, mr));
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
      "min",
      tester.clone()
          .m(mr).n(nr).k(k_block).min(0.0f)
      , test_func, arch_flags));
  gemm_tests.push_back(GemmTestParams(
      "max",
      tester.clone()
          .m(mr).n(nr).k(k_block).max(0.0f)
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
    nr = nr * xnn_init_hardware_config()->vlenb / sizeof(float);
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
        "min",
        tester.clone()
            .m(mr).n(nr).k(k_block).min(0.0f)
        , test_func, arch_flags));
    gemm_tests.push_back(GemmTestParams(
        "max",
        tester.clone()
            .m(mr).n(nr).k(k_block).max(0.0f)
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


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__ASM_AARCH32_NEON_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__ASM_AARCH32_NEON_CORTEX_A7, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__ASM_AARCH32_NEON_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__ASM_AARCH32_NEON_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_CORTEX_A53_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_CORTEX_A75, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A53_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A55, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_CORTEX_A75_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X8__ASM_AARCH64_NEONFMA_CORTEX_A75, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_CORTEX_A73, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X16__NEON_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x16__neon_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X16__NEON_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x16__neon_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X16__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x16__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__NEON_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__NEON_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__NEON_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8S4__NEON, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8s4__neon,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X4__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x4__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__NEON_DUP_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__NEON_DUP_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__NEON_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__NEONFMA_DUP_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__NEONFMA_DUP_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8S4__NEON, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8s4__neon,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8S4__NEONFMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8s4__neonfma,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X2__NEON_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x2__neon_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__NEONFMA_DUP_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_arm_neon_fma)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__SSE_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__sse_dup,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__SSE_LOAD1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__sse_load1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8S4__SSE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8s4__sse,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X2C4__SSE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x2c4__sse,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__SSE_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__sse_dup,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8S4__SSE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8s4__sse,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X8S4__SSE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x8s4__sse,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__SSE_LOAD1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__sse_load1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x16__avx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__avx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x16__avx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_7X8__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_7x8__avx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X16S4__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x16__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X16S4__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x16s4__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X16S4__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X8__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x8__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X16S4__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x16s4__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X16S4__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x16s4__fma3_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_fma3)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X16__AVX512F_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx512f)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X16__AVX512F_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x16__avx512f_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx512f)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_7X16__AVX512F_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx512f)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_8X16__AVX512F_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_8x16__avx512f_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx512f)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X32__AVX512F_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x32__avx512f_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx512f)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X32__AVX512F_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x32__avx512f_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx512f)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_7X32__AVX512F_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_7x32__avx512f_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx512f)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_8X32__AVX512F_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_8x32__avx512f_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_x86_avx512f)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__WASMSIMD_ARM_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__WASMSIMD_X86_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__WASMSIMD_X86_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X8__WASMSIMD_X86_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X8S4__WASMSIMD_ARM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_arm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X8S4__WASMSIMD_X86, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_x86,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X2C4__WASMSIMD_ARM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_arm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X2C4__WASMSIMD_X86, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmsimd_x86,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__WASMSIMD_X86_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X8__WASMSIMD_ARM_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__WASMSIMD_ARM_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__WASMSIMD_X86_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__WASMSIMD_X86_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8S4__WASMSIMD_ARM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_arm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8S4__WASMSIMD_X86, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_x86,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8__WASMRELAXEDSIMD_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X8S4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X2C4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X2C4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_4X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_4x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X8__WASMRELAXEDSIMD_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_5X8S4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8__WASMRELAXEDSIMD_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X8S4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    F32_IGEMM_MINMAX_1X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_f32_igemm_minmax_ukernel_1x4__scalar,
                      xnn_init_f32_minmax_scalar_params,
                      xnn_pack_f32_conv_goki_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    F32_IGEMM_MINMAX_4X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_f32_igemm_minmax_ukernel_4x2__scalar,
                      xnn_init_f32_minmax_scalar_params,
                      xnn_pack_f32_conv_goki_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    F32_IGEMM_MINMAX_4X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/true,
        /*unsigned_inputs=*/false,
        /*planes=*/1,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_f32_igemm_minmax_ukernel_4x4__scalar,
                      xnn_init_f32_minmax_scalar_params,
                      xnn_pack_f32_conv_goki_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X4V__RVV, GemmTest,
      testing::ValuesIn(CreateTests3(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x4v__rvv,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_riscv_vector)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_7X4V__RVV, GemmTest,
      testing::ValuesIn(CreateTests3(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_7x4v__rvv,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_riscv_vector)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_2X32__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_2x32__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X32__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x32__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X32__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x32__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_7X32__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_7x32__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_8X32__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_8x32__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_9X32__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/9, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_9x32__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_2X64__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_2x64__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X64__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x64__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_6X64__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_6x64__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_7X64__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_7x64__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_8X64__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_8x64__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_9X64__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/9, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_9x64__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_1X128__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/128, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_1x128__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_2X128__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/128, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_2x128__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_3X128__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/128, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_3x128__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_IGEMM_MINMAX_8X128__HVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/128, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_igemm_minmax_ukernel_8x128__hvx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_conv_goki_w);
          },
          xnn_arch_hvx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON

