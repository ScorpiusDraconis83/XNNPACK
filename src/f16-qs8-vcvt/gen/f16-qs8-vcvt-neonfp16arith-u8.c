// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-qs8-vcvt/neonfp16arith.c.in
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
#include "src/xnnpack/vcvt.h"


void xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8(
    size_t batch,
    const xnn_float16* input,
    int8_t* output,
    const struct xnn_f16_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;

  const float16x8_t vscale = vreinterpretq_f16_u16(vdupq_n_u16(*(const uint16_t*) &params->scalar.scale));
  const int16x8_t voutput_zero_point = vdupq_n_s16(params->scalar.output_zero_point);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vx = vmulq_f16(vx, vscale);

    int16x8_t vacc = vcvtnq_s16_f16(vx);

    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));

    vx = vmulq_f16(vx, vscale);

    int16x8_t vacc = vcvtnq_s16_f16(vx);
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);

    if (batch & (4 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_s8(output, vy, 0);
    }
  }
}
