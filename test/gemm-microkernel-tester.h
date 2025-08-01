// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_GEMM_MICROKERNEL_TESTER_H_
#define XNNPACK_TEST_GEMM_MICROKERNEL_TESTER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <ostream>
#include <string>

#include <gtest/gtest.h>
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/requantization.h"
#include "test/next_prime.h"

class GemmMicrokernelTester {
 public:
  GemmMicrokernelTester clone() const { return *this; }

  GemmMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  size_t mr() const { return this->mr_; }

  GemmMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  size_t nr() const { return this->nr_; }

  GemmMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  size_t kr() const { return this->kr_; }

  GemmMicrokernelTester& sr(size_t sr) {
    this->sr_ = sr;
    return *this;
  }

  size_t sr() const { return this->sr_; }

  GemmMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  size_t m() const { return this->m_; }

  GemmMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  size_t n() const { return this->n_; }

  GemmMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  size_t k() const { return this->k_; }

  GemmMicrokernelTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  size_t ks() const { return this->ks_; }

  inline GemmMicrokernelTester& bl(size_t bl) {
    this->bl_ = bl;
    return *this;
  }

  inline size_t bl() const { return this->bl_; }

  size_t packed_k() const { return round_up_po2(k(), kr() * sr()); }

  size_t packed_n() const { return round_up(n(), nr()); }

  bool unsigned_inputs() const { return this->unsigned_inputs_; }

  GemmMicrokernelTester& unsigned_inputs(bool unsigned_inputs) {
    this->unsigned_inputs_ = unsigned_inputs;
    return *this;
  }

  uint8_t planes() const { return this->planes_; }

  GemmMicrokernelTester& planes(uint8_t planes) {
    this->planes_ = planes;
    return *this;
  }

  GemmMicrokernelTester& a_stride(size_t a_stride) {
    this->a_stride_ = a_stride;
    return *this;
  }

  size_t a_stride() const {
    return this->a_stride_ == 0 ? k() : this->a_stride_;
  }

  GemmMicrokernelTester& cm_stride(size_t cm_stride) {
    this->cm_stride_ = cm_stride;
    return *this;
  }

  size_t cm_stride() const {
    return this->cm_stride_ == 0 ? n() : this->cm_stride_;
  }

  GemmMicrokernelTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  uint8_t a_zero_point() const { return this->a_zero_point_; }

  GemmMicrokernelTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  uint8_t b_zero_point() const { return this->b_zero_point_; }

  GemmMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const { return this->qmin_; }

  GemmMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const { return this->qmax_; }

  GemmMicrokernelTester& min(float min) {
    this->min_ = min;
    return *this;
  }

  float min() const { return this->min_; }

  GemmMicrokernelTester& max(float max) {
    this->max_ = max;
    return *this;
  }

  float max() const { return this->max_; }

  GemmMicrokernelTester& a_offset(size_t a_offset) {
    this->a_offset_ = a_offset;
    return *this;
  }

  size_t a_offset() const { return this->a_offset_; }

  GemmMicrokernelTester& zero_index(size_t zero_index) {
    this->zero_index_ = zero_index;
    return *this;
  }

  size_t zero_index() const { return this->zero_index_; }

  GemmMicrokernelTester& known_nc_mod_nr(bool known_nc_mod_nr) {
    this->known_nc_mod_nr_ = known_nc_mod_nr;
    return *this;
  }

  bool known_nc_mod_nr() const { return known_nc_mod_nr_; }

  GemmMicrokernelTester& relu(bool relu) {
    this->relu_ = relu;
    return *this;
  }

  bool relu() const { return relu_; }

  GemmMicrokernelTester& mr_packed(size_t mr_packed) {
    this->mr_packed_ = mr_packed;
    return *this;
  }

  size_t mr_packed() const {
    if (this->mr_packed_ == 0) {
      return this->mr_;
    }
    return this->mr_packed_;
  }

  size_t nc_mod_nr() const { return known_nc_mod_nr() ? n() % nr() : SIZE_MAX; }

  void Test(xnn_qd8_f16_qc8w_igemm_ukernel_fn igemm,
            xnn_init_f16_minmax_params_fn init_params,
            xnn_pack_qs8_igemm_fn pack) const;

  void Test(xnn_qd8_f32_qc8w_igemm_ukernel_fn gemm,
            xnn_init_f32_minmax_params_fn init_params,
            xnn_pack_qs8_igemm_fn pack) const;

  void Test(xnn_qu8_gemm_minmax_ukernel_fn gemm,
            xnn_init_qu8_conv_minmax_params_fn init_params,
            xnn_pack_qu8_gemm_fn pack, xnn_qu8_requantize_fn requantize) const;

  void Test(xnn_qu8_igemm_minmax_ukernel_fn igemm,
            xnn_init_qu8_conv_minmax_params_fn init_params,
            xnn_pack_qu8_igemm_fn pack, xnn_qu8_requantize_fn requantize);

  void Test(xnn_qs8_qc4w_gemm_minmax_ukernel_fn gemm,
            xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
            xnn_pack_qs8_qc4w_gemm_fn pack, xnn_qs8_requantize_fn requantize);

  void Test(xnn_qs8_qc8w_gemm_minmax_ukernel_fn gemm,
            xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
            xnn_pack_qs8_gemm_fn pack, xnn_qs8_requantize_fn requantize) const;

  void Test(xnn_qs8_qc8w_igemm_minmax_ukernel_fn igemm,
            xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
            xnn_pack_qs8_igemm_fn pack, xnn_qs8_requantize_fn requantize) const;

  void Test(xnn_qs8_gemm_minmax_ukernel_fn gemm,
            xnn_init_qs8_conv_minmax_params_fn init_params,
            xnn_pack_qs8_gemm_fn pack, xnn_qs8_requantize_fn requantize) const;

  void Test(xnn_qd8_f16_qc8w_gemm_ukernel_fn gemm,
            xnn_init_f16_minmax_params_fn init_params,
            xnn_pack_qs8_gemm_fn pack) const;

  void Test(xnn_qd8_f32_qc8w_gemm_ukernel_fn gemm,
            xnn_init_f32_minmax_params_fn init_params,
            xnn_pack_qs8_gemm_fn pack) const;

  void Test(xnn_qd8_f16_qc4w_gemm_ukernel_fn gemm,
            xnn_init_f16_qc4w_minmax_params_fn init_params,
            xnn_pack_qs8_qc4w_gemm_fn pack) const;

  void Test(xnn_qd8_f16_qb4w_gemm_ukernel_fn gemm,
            xnn_init_f16_qb4w_minmax_params_fn init_params,
            xnn_pack_qs8_qb4w_gemm_fn pack) const;

  void Test(xnn_qd8_f32_qc4w_gemm_ukernel_fn gemm,
            xnn_init_f32_qc4w_minmax_params_fn init_params,
            xnn_pack_qs8_qc4w_gemm_fn pack) const;

  void Test(xnn_qd8_f32_qb4w_gemm_ukernel_fn gemm,
            xnn_init_f32_qb4w_minmax_params_fn init_params,
            xnn_pack_qs8_qb4w_gemm_fn pack) const;

  void Test(xnn_qs8_igemm_minmax_ukernel_fn igemm,
            xnn_init_qs8_conv_minmax_params_fn init_params,
            xnn_pack_qs8_igemm_fn pack, xnn_qs8_requantize_fn requantize) const;

  void Test(xnn_bf16_f32_gemm_minmax_ukernel_fn gemm_minmax,
            xnn_init_f32_minmax_params_fn init_params,
            xnn_pack_bf16_f32_gemm_fn pack) const;

  void Test(xnn_bf16_gemm_minmax_ukernel_fn gemm_minmax,
            xnn_init_bf16_minmax_params_fn init_params,
            xnn_pack_f16_gemm_fn pack) const;

  void Test(xnn_f16_gemm_minmax_ukernel_fn gemm_minmax,
            xnn_init_f16_minmax_params_fn init_params,
            xnn_pack_f16_gemm_fn pack) const;

  void Test(xnn_f16_igemm_minmax_ukernel_fn igemm_minmax,
            xnn_init_f16_minmax_params_fn init_params,
            xnn_pack_f16_igemm_fn pack) const;

  void Test(xnn_f32_ppmm_minmax_ukernel_fn ppmm_minmax,
            xnn_init_f32_minmax_params_fn init_params,
            xnn_pack_f32_gemm_fn pack) const;

  void Test(xnn_f32_gemm_ukernel_fn gemm, xnn_pack_f32_gemm_fn pack) const;

  void Test(xnn_f32_gemm_relu_ukernel_fn gemm_relu,
            xnn_pack_f32_gemm_fn pack) const;

  void Test(xnn_f32_gemm_minmax_ukernel_fn gemm_minmax,
            xnn_init_f32_minmax_params_fn init_params,
            xnn_pack_f32_gemm_fn pack) const;

  void Test(xnn_f32_qc4w_gemm_minmax_ukernel_fn gemm_minmax,
            xnn_init_f32_qc4w_minmax_params_fn init_params,
            xnn_pack_f32_qc4w_gemm_fn pack) const;

  void Test(xnn_f32_qc8w_gemm_ukernel_fn gemm,
            xnn_pack_f32_qs8w_gemm_fn pack) const;

  void Test(xnn_f32_qc8w_gemm_relu_ukernel_fn gemm_relu,
            xnn_pack_f32_qs8w_gemm_fn pack) const;

  void Test(xnn_f32_qc8w_gemm_minmax_ukernel_fn gemm_minmax,
            xnn_init_f32_minmax_params_fn init_params,
            xnn_pack_f32_qs8w_gemm_fn pack) const;

  void Test(xnn_f32_igemm_ukernel_fn igemm, xnn_pack_f32_igemm_fn pack) const;

  void Test(xnn_f32_igemm_relu_ukernel_fn igemm_relu,
            xnn_pack_f32_igemm_fn pack) const;

  void Test(xnn_f32_igemm_minmax_ukernel_fn igemm_minmax,
            xnn_init_f32_minmax_params_fn init_params,
            xnn_pack_f32_igemm_fn pack) const;

  void Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_fn gemm,
            xnn_init_f32_minmax_params_fn init_minmax_params,
            xnn_pack_weights_and_biases_fn pack,
            xnn_packed_stride_weights_and_biases_fn packed_stride);

  void Test_QP8F32QC8W(xnn_qp8_f32_qc8w_gemm_minmax_ukernel_fn gemm,
                       xnn_init_f32_minmax_params_fn init_minmax_params,
                       xnn_pack_weights_and_biases_fn pack,
                       xnn_packed_stride_weights_and_biases_fn packed_stride);

  void Test(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_fn gemm,
            xnn_init_f32_qb4w_minmax_params_fn init_minmax_params,
            xnn_pack_weights_and_biases_fn pack,
            xnn_packed_stride_weights_and_biases_fn packed_stride);

  void Test_PF32(xnn_pf32_gemm_minmax_ukernel_fn gemm,
                 xnn_init_f32_minmax_params_fn init_minmax_params,
                 xnn_pack_weights_and_biases_fn pack,
                 xnn_packed_stride_weights_and_biases_fn packed_stride);

  void Test_PF16(xnn_pf16_gemm_minmax_ukernel_fn gemm,
                 xnn_init_f16_minmax_params_fn init_minmax_params,
                 xnn_pack_weights_and_biases_fn pack,
                 xnn_packed_stride_weights_and_biases_fn packed_stride);

  void Test_PQS8(xnn_pqs8_qc8w_gemm_minmax_ukernel_fn gemm,
                 xnn_init_qs8_qc8w_conv_minmax_params_fn init_minmax_params,
                 xnn_pack_weights_and_biases_fn pack,
                 xnn_packed_stride_weights_and_biases_fn packed_stride) const;

  void Test_PQS8(xnn_packed_lhs_igemm_ukernel_fn packed_igemm,
                 xnn_init_qs8_qc8w_conv_minmax_params_fn init_minmax_params,
                 xnn_pack_lh_igemm_ukernel_fn pack_lh_for_igemm_fn,
                 xnn_pack_lh_igemm_size_fn size_for_igemm_fn,
                 xnn_pack_qs8_igemm_fn pack_rhs,
                 xnn_qs8_requantize_fn requantize) const;

 private:
  size_t mr_{1};
  size_t nr_{1};
  size_t kr_{1};
  size_t sr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t ks_{1};
  size_t bl_{0};
  bool unsigned_inputs_{false};
  uint8_t planes_{1};
  size_t a_stride_{0};
  size_t cm_stride_{0};
  uint8_t a_zero_point_{127};
  uint8_t b_zero_point_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  float min_ = -std::numeric_limits<float>::infinity();
  float max_ = std::numeric_limits<float>::infinity();
  size_t a_offset_{0};
  size_t zero_index_{0};
  bool known_nc_mod_nr_{true};
  bool relu_{false};
  size_t mr_packed_{0};
};

enum class LoopStepType { Linear, NextPrime };

struct LoopParams {
  LoopParams() = default;
  explicit LoopParams(size_t from, size_t to, size_t step,
                      LoopStepType step_type)
      : is_set(true),
        from(from),
        to(std::max(from, to)),
        step(step),
        step_type(step_type) {}
  bool is_set = false;
  size_t from = 1;
  size_t to = 1;
  size_t step = 1;
  LoopStepType step_type = LoopStepType::Linear;

  size_t next(size_t n) const {
    switch (step_type) {
      case LoopStepType::Linear:
        return n + step;
      case LoopStepType::NextPrime:
        return xnnpack::NextPrime(n + step);
      default:
        std::cerr << "Unknown loop step type " << static_cast<int>(step_type)
                  << std::endl;
        std::abort();
    }
  }
};

inline std::ostream& operator<<(std::ostream& outs,
                                const LoopParams& loop_params) {
  return outs << "LoopParams(from=" << loop_params.from
              << ", to=" << loop_params.to << ", step=" << loop_params.step
              << ", type="
              << (loop_params.step_type == LoopStepType::Linear ? "linear"
                                                                : "next-prime")
              << ")";
}

struct GemmTestParams {
  GemmTestParams(std::string test_name, GemmMicrokernelTester tester,
                 std::function<void(GemmMicrokernelTester& tester)> test_func,
                 uint64_t arch_flags = 0)
      : test_name(test_name),
        tester(tester),
        test_func(test_func),
        arch_flags(arch_flags) {}

  // Setters for the loops over `k`, `m`, and `n`.
  GemmTestParams& loop_k(size_t from, size_t to, size_t step = 1,
                         LoopStepType step_type = LoopStepType::NextPrime) {
    loop_k_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_m(size_t from, size_t to, size_t step = 1,
                         LoopStepType step_type = LoopStepType::Linear) {
    loop_m_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_n(size_t from, size_t to, size_t step = 1,
                         LoopStepType step_type = LoopStepType::NextPrime) {
    loop_n_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_zi(size_t from, size_t to, size_t step = 1,
                          LoopStepType step_type = LoopStepType::Linear) {
    loop_zi_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_bzp(size_t from, size_t to, size_t step = 1,
                           LoopStepType step_type = LoopStepType::Linear) {
    loop_bzp_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_bl(size_t from, size_t to, size_t step = 1,
                          LoopStepType step_type = LoopStepType::Linear) {
    loop_bl_ = LoopParams(from, to, step, step_type);
    return *this;
  }

  std::string test_name;
  GemmMicrokernelTester tester;
  std::function<void(GemmMicrokernelTester& tester)> test_func;
  uint64_t arch_flags;
  LoopParams loop_k_;
  LoopParams loop_m_;
  LoopParams loop_n_;
  LoopParams loop_zi_;
  LoopParams loop_bzp_;
  LoopParams loop_bl_;
};

inline std::ostream& operator<<(std::ostream& outs,
                                const GemmTestParams& params) {
  outs << "GemmTestParams(name=" << params.test_name;
  if (params.loop_k_.is_set) {
    outs << ", loop_k=" << params.loop_k_;
  } else {
    outs << ", k=" << params.tester.k();
  }
  if (params.loop_m_.is_set) {
    outs << ", loop_m=" << params.loop_m_;
  } else {
    outs << ", m=" << params.tester.m();
  }
  if (params.loop_n_.is_set) {
    outs << ", loop_n=" << params.loop_n_;
  } else {
    outs << ", n=" << params.tester.n();
  }
  if (params.loop_zi_.is_set) {
    outs << ", loop_zi=" << params.loop_zi_;
  } else {
    outs << ", zi=" << params.tester.zero_index();
  }
  if (params.loop_bzp_.is_set) {
    outs << ", loop_bzp=" << params.loop_bzp_;
  } else {
    outs << ", bzp=" << static_cast<int>(params.tester.b_zero_point());
  }
  if (params.loop_bl_.is_set) {
    outs << ", loop_bl=" << params.loop_bl_;
  } else {
    outs << ", bl=" << params.tester.bl();
  }
  return outs << ")";
}

using GemmTest = testing::TestWithParam<GemmTestParams>;

#endif  // XNNPACK_TEST_GEMM_MICROKERNEL_TESTER_H_
