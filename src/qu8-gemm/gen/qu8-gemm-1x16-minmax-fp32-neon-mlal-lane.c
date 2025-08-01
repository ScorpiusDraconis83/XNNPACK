// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/neon-mlal-lane.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"


void xnn_qu8_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params* restrict params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  const uint8x8_t vb_zero_point = vdup_n_u8(params->fp32_neon.kernel_zero_point);
  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

    size_t k = kc;
    while (k >= 8 * sizeof(uint8_t)) {
      const uint8x8_t va0 = vld1_u8(a0); a0 += 8;
      const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));

      const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb89ABCDEFc0 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb89ABCDEFc1 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb89ABCDEFc2 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb89ABCDEFc3 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb89ABCDEFc4 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb89ABCDEFc5 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb89ABCDEFc6 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb89ABCDEFc7 = vld1_u8(w); w = (const uint8_t*) w + 8;

      const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
      const int16x8_t vxb89ABCDEFc0 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc0, vb_zero_point));
      const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
      const int16x8_t vxb89ABCDEFc1 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc1, vb_zero_point));
      const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
      const int16x8_t vxb89ABCDEFc2 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc2, vb_zero_point));
      const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
      const int16x8_t vxb89ABCDEFc3 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc3, vb_zero_point));
      const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
      const int16x8_t vxb89ABCDEFc4 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc4, vb_zero_point));
      const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
      const int16x8_t vxb89ABCDEFc5 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc5, vb_zero_point));
      const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
      const int16x8_t vxb89ABCDEFc6 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc6, vb_zero_point));
      const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));
      const int16x8_t vxb89ABCDEFc7 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc7, vb_zero_point));


      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);

      k -= 8 * sizeof(uint8_t);
    }
    if XNN_UNLIKELY(k != 0) {
      const uint8x8_t va0 = vld1_u8(a0); a0 = (const uint8_t*) ((uintptr_t) a0 + k);
      const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));

      const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const uint8_t*) w + 8;
      const uint8x8_t vb89ABCDEFc0 = vld1_u8(w); w = (const uint8_t*) w + 8;

      const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
      const int16x8_t vxb89ABCDEFc0 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc0, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);

      if (k >= 2 * sizeof(uint8_t)) {
        const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint8x8_t vb89ABCDEFc1 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
        const int16x8_t vxb89ABCDEFc1 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc1, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);

        if (k > 2 * sizeof(uint8_t)) {
          const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const uint8_t*) w + 8;
          const uint8x8_t vb89ABCDEFc2 = vld1_u8(w); w = (const uint8_t*) w + 8;
          const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
          const int16x8_t vxb89ABCDEFc2 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc2, vb_zero_point));

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);

          if (k >= 4 * sizeof(uint8_t)) {
            const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const uint8_t*) w + 8;
            const uint8x8_t vb89ABCDEFc3 = vld1_u8(w); w = (const uint8_t*) w + 8;
            const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
            const int16x8_t vxb89ABCDEFc3 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc3, vb_zero_point));

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

            if (k > 4 * sizeof(uint8_t)) {
              const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const uint8_t*) w + 8;
              const uint8x8_t vb89ABCDEFc4 = vld1_u8(w); w = (const uint8_t*) w + 8;
              const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
              const int16x8_t vxb89ABCDEFc4 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc4, vb_zero_point));

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);

              if (k >= 6 * sizeof(uint8_t)) {
                const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const uint8_t*) w + 8;
                const uint8x8_t vb89ABCDEFc5 = vld1_u8(w); w = (const uint8_t*) w + 8;
                const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
                const int16x8_t vxb89ABCDEFc5 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc5, vb_zero_point));

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);

                if (k > 6 * sizeof(uint8_t)) {
                  const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const uint8_t*) w + 8;
                  const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
                  const uint8x8_t vb89ABCDEFc6 = vld1_u8(w); w = (const uint8_t*) w + 8;
                  const int16x8_t vxb89ABCDEFc6 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc6, vb_zero_point));

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                  vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                  vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                }
              }
            }
          }
        }
      }
    }

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    const float32x4_t vscale = vdupq_n_f32(params->fp32_neon.scale);
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale);
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale);
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale);
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscale);

    const float32x4_t vmagic_bias = vdupq_n_f32(params->fp32_neon.magic_bias);
    vacc0x0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0x0123, vmagic_bias));
    vacc0x4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0x4567, vmagic_bias));
    vacc0x89AB = vreinterpretq_s32_f32(vaddq_f32(vfpacc0x89AB, vmagic_bias));
    vacc0xCDEF = vreinterpretq_s32_f32(vaddq_f32(vfpacc0xCDEF, vmagic_bias));

    const int32x4_t vmagic_bias_less_output_zero_point = vdupq_n_s32(params->fp32_neon.magic_bias_less_output_zero_point);
    vacc0x0123 = vqsubq_s32(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc0x4567 = vqsubq_s32(vacc0x4567, vmagic_bias_less_output_zero_point);
    vacc0x89AB = vqsubq_s32(vacc0x89AB, vmagic_bias_less_output_zero_point);
    vacc0xCDEF = vqsubq_s32(vacc0xCDEF, vmagic_bias_less_output_zero_point);

    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
      int16x8_t vacc0x89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF);


      uint8x16_t vout0x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc0x89ABCDEF);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
      int16x8_t vacc0x89ABCDEF = vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF));


      uint8x16_t vout0x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc0x89ABCDEF));
    #endif

    const uint8x16_t voutput_min = vdupq_n_u8(params->fp32_neon.output_min);
    vout0x0123456789ABCDEF = vmaxq_u8(vout0x0123456789ABCDEF, voutput_min);

    const uint8x16_t voutput_max = vdupq_n_u8(params->fp32_neon.output_max);
    vout0x0123456789ABCDEF = vminq_u8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_u8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      uint8x8_t vout0x01234567 = vget_low_u8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_u8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_u8(vout0x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_u8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_u8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_u8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_u8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_u8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
