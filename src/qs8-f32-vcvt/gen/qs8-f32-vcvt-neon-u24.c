// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-f32-vcvt/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"


void xnn_qs8_f32_vcvt_ukernel__neon_u24(
    size_t batch,
    const int8_t* input,
    float* output,
    const struct xnn_qs8_f32_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16x8_t vminus_zero_point = vdupq_n_s16(-params->scalar.zero_point);
  const float32x4_t vscale = vdupq_n_f32(params->scalar.scale);
  for (; batch >= 24 * sizeof(int8_t); batch -= 24 * sizeof(int8_t)) {
    const int8x8_t vx01234567 = vld1_s8(input); input += 8;
    const int8x8_t vx89ABCDEF = vld1_s8(input); input += 8;
    const int8x8_t vxGHIJKLMN = vld1_s8(input); input += 8;

    const int16x8_t vhx01234567 = vaddw_s8(vminus_zero_point, vx01234567);
    const int16x8_t vhx89ABCDEF = vaddw_s8(vminus_zero_point, vx89ABCDEF);
    const int16x8_t vhxGHIJKLMN = vaddw_s8(vminus_zero_point, vxGHIJKLMN);

    const int32x4_t vwx0123 = vmovl_s16(vget_low_s16(vhx01234567));
    const int32x4_t vwx4567 = vmovl_s16(vget_high_s16(vhx01234567));
    const int32x4_t vwx89AB = vmovl_s16(vget_low_s16(vhx89ABCDEF));
    const int32x4_t vwxCDEF = vmovl_s16(vget_high_s16(vhx89ABCDEF));
    const int32x4_t vwxGHIJ = vmovl_s16(vget_low_s16(vhxGHIJKLMN));
    const int32x4_t vwxKLMN = vmovl_s16(vget_high_s16(vhxGHIJKLMN));

    float32x4_t vy0123 = vcvtq_f32_s32(vwx0123);
    float32x4_t vy4567 = vcvtq_f32_s32(vwx4567);
    float32x4_t vy89AB = vcvtq_f32_s32(vwx89AB);
    float32x4_t vyCDEF = vcvtq_f32_s32(vwxCDEF);
    float32x4_t vyGHIJ = vcvtq_f32_s32(vwxGHIJ);
    float32x4_t vyKLMN = vcvtq_f32_s32(vwxKLMN);

    vy0123 = vmulq_f32(vy0123, vscale);
    vy4567 = vmulq_f32(vy4567, vscale);
    vy89AB = vmulq_f32(vy89AB, vscale);
    vyCDEF = vmulq_f32(vyCDEF, vscale);
    vyGHIJ = vmulq_f32(vyGHIJ, vscale);
    vyKLMN = vmulq_f32(vyKLMN, vscale);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
    vst1q_f32(output, vyGHIJ); output += 4;
    vst1q_f32(output, vyKLMN); output += 4;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const int8x8_t vx = vld1_s8(input); input += 8;

    const int16x8_t vhx = vaddw_s8(vminus_zero_point, vx);

    const int32x4_t vwx_lo = vmovl_s16(vget_low_s16(vhx));
    const int32x4_t vwx_hi = vmovl_s16(vget_high_s16(vhx));

    float32x4_t vy_lo = vcvtq_f32_s32(vwx_lo);
    float32x4_t vy_hi = vcvtq_f32_s32(vwx_hi);

    vy_lo = vmulq_f32(vy_lo, vscale);
    vy_hi = vmulq_f32(vy_hi, vscale);

    vst1q_f32(output, vy_lo); output += 4;
    vst1q_f32(output, vy_hi); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const int8x8_t vx = vld1_s8(input);

    const int16x8_t vhx = vaddw_s8(vminus_zero_point, vx);

    const int32x4_t vwx_lo = vmovl_s16(vget_low_s16(vhx));
    const int32x4_t vwx_hi = vmovl_s16(vget_high_s16(vhx));

    float32x4_t vy = vcvtq_f32_s32(vwx_lo);
    vy = vmulq_f32(vy, vscale);

    if (batch & (4 * sizeof(int8_t))) {
      vst1q_f32(output, vy); output += 4;
      vy = vcvtq_f32_s32(vwx_hi);
      vy = vmulq_f32(vy, vscale);
    }
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(int8_t))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(int8_t))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}
