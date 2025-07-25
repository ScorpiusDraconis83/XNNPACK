// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-f32acc-rsum/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"


void xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u24_acc3(
    size_t batch,
    const xnn_float16* input,
    float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  float32x4_t vacc2 = vmovq_n_f32(0.0f);
  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    const float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    const float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    const float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    const float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    const float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc2 = vaddq_f32(vacc2, vt2);
    vacc0 = vaddq_f32(vacc0, vt3);
    vacc1 = vaddq_f32(vacc1, vt4);
    vacc2 = vaddq_f32(vacc2, vt5);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    const float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    const float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    batch -= 8 * sizeof(uint16_t);
    vacc2 = vaddq_f32(vacc2, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
  }
  vacc0 = vaddq_f32(vacc0, vacc2);
  vacc0 = vaddq_f32(vacc0, vacc1);
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i)); i += 4;
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i)); i += 2;
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}
