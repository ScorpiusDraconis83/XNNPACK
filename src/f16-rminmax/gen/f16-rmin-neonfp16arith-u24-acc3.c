// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/neonfp16arith.c.in
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


void xnn_f16_rmin_ukernel__neonfp16arith_u24_acc3(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  float16x8_t vmin0 = vreinterpretq_f16_u16(vld1q_dup_u16(o));
  float16x8_t vmin1 = vmin0;
  float16x8_t vmin2 = vmin0;
  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    const float16x8_t vt0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vmin0 = vminq_f16(vmin0, vt0);
    vmin1 = vminq_f16(vmin1, vt1);
    vmin2 = vminq_f16(vmin2, vt2);
  }
  vmin0 = vminq_f16(vmin0, vmin1);
  vmin0 = vminq_f16(vmin0, vmin2);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vt = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vmin0 = vminq_f16(vmin0, vt);
  }
  float16x4_t vmin_lo = vmin_f16(vget_low_f16(vmin0), vget_high_f16(vmin0));

  if (XNN_UNLIKELY(batch != 0)) {
    const float16x8_t vt = vreinterpretq_f16_u16(vld1q_u16(i));
    float16x4_t vt_lo = vget_low_f16(vt);
    if (batch & (4 * sizeof(uint16_t))) {
      vmin_lo = vmin_f16(vmin_lo, vt_lo);
      vt_lo = vget_high_f16(vt);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vmin_lo = vmin_f16(vmin_lo, vext_f16(vmin_lo, vt_lo, 2));
      vt_lo = vext_f16(vt_lo, vt_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vmin_lo = vmin_f16(vmin_lo, vext_f16(vmin_lo, vt_lo, 1));
    }
  }
  #if XNN_ARCH_ARM64 && defined(__GNUC__)
    *((__fp16*) o) = vminv_f16(vmin_lo);
  #else
    vmin_lo = vpmin_f16(vmin_lo, vmin_lo);
    vmin_lo = vpmin_f16(vmin_lo, vmin_lo);
    vst1_lane_u16(o, vreinterpret_u16_f16(vmin_lo), 0);
  #endif
}
