// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-neon-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_qu8_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neonv8_mul16(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const uint8x8_t vkernel_zero_point = vdup_n_u8(params->fp32_neonv8.kernel_zero_point);
  const float32x4_t vscale = vdupq_n_f32(params->fp32_neonv8.scale);
  const int16x8_t voutput_zero_point = vdupq_n_s16(params->fp32_neonv8.output_zero_point);
  const uint8x16_t voutput_min = vdupq_n_u8(params->fp32_neonv8.output_min);
  const uint8x16_t voutput_max = vdupq_n_u8(params->fp32_neonv8.output_max);

  do {
    const void* w = weights;

    // First pass to process 5 inputs.
    {
      int32_t* b = buffer;
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 16; c -= 16) {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vaccCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi0x89ABCDEF), vget_low_s16(vk0x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi0x89ABCDEF), vget_high_s16(vk0x89ABCDEF));

        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi1x89ABCDEF), vget_low_s16(vk1x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi1x89ABCDEF), vget_high_s16(vk1x89ABCDEF));

        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi2x89ABCDEF), vget_low_s16(vk2x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi2x89ABCDEF), vget_high_s16(vk2x89ABCDEF));

        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi3x89ABCDEF), vget_low_s16(vk3x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi3x89ABCDEF), vget_high_s16(vk3x89ABCDEF));

        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi4x89ABCDEF), vget_low_s16(vk4x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi4x89ABCDEF), vget_high_s16(vk4x89ABCDEF));

        vst1q_s32(b, vacc0123); b += 4;
        vst1q_s32(b, vacc4567); b += 4;
        vst1q_s32(b, vacc89AB); b += 4;
        vst1q_s32(b, vaccCDEF); b += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
          int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

          const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
          const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));

          const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
          const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));

          const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
          const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));

          const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
          const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));

          const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
          const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));

          vst1q_s32(b, vacc0123); b += 4;
          vst1q_s32(b, vacc4567); b += 4;
          c -= 8;
        } while (c != 0);
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      int32_t* b = buffer;

      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = round_up_po2(channels, 8);

      for (; c >= 16; c -= 16) {
        int32x4_t vacc0123 = vld1q_s32(b);
        int32x4_t vacc4567 = vld1q_s32(b + 4);
        int32x4_t vacc89AB = vld1q_s32(b + 8);
        int32x4_t vaccCDEF = vld1q_s32(b + 12);

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi0x89ABCDEF), vget_low_s16(vk0x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi0x89ABCDEF), vget_high_s16(vk0x89ABCDEF));

        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi1x89ABCDEF), vget_low_s16(vk1x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi1x89ABCDEF), vget_high_s16(vk1x89ABCDEF));

        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi2x89ABCDEF), vget_low_s16(vk2x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi2x89ABCDEF), vget_high_s16(vk2x89ABCDEF));

        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi3x89ABCDEF), vget_low_s16(vk3x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi3x89ABCDEF), vget_high_s16(vk3x89ABCDEF));

        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi4x89ABCDEF), vget_low_s16(vk4x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi4x89ABCDEF), vget_high_s16(vk4x89ABCDEF));

        vst1q_s32(b, vacc0123); b += 4;
        vst1q_s32(b, vacc4567); b += 4;
        vst1q_s32(b, vacc89AB); b += 4;
        vst1q_s32(b, vaccCDEF); b += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          int32x4_t vacc0123 = vld1q_s32(b);
          int32x4_t vacc4567 = vld1q_s32(b + 4);

          const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
          const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));

          const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
          const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));

          const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
          const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));

          const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
          const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));

          const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
          const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));

          vst1q_s32(b, vacc0123); b += 4;
          vst1q_s32(b, vacc4567); b += 4;
          c -= 8;
        } while (c != 0);
      }
    }

    // Last pass to process up to 5 inputs.
    {
      const int32_t* b = buffer;

      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }

      size_t c = channels;
      for (; c >= 16; c -= 16) {
        int32x4_t vacc0123 = vld1q_s32(b); b += 4;
        int32x4_t vacc4567 = vld1q_s32(b); b += 4;
        int32x4_t vacc89AB = vld1q_s32(b); b += 4;
        int32x4_t vaccCDEF = vld1q_s32(b); b += 4;

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi0x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
        const int16x8_t vk0x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi0x89ABCDEF), vget_low_s16(vk0x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi0x89ABCDEF), vget_high_s16(vk0x89ABCDEF));

        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi1x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
        const int16x8_t vk1x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi1x89ABCDEF), vget_low_s16(vk1x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi1x89ABCDEF), vget_high_s16(vk1x89ABCDEF));

        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi2x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
        const int16x8_t vk2x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi2x89ABCDEF), vget_low_s16(vk2x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi2x89ABCDEF), vget_high_s16(vk2x89ABCDEF));

        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi3x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
        const int16x8_t vk3x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi3x89ABCDEF), vget_low_s16(vk3x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi3x89ABCDEF), vget_high_s16(vk3x89ABCDEF));

        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;
        const int16x8_t vi4x89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
        const int16x8_t vk4x89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
        vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi4x89ABCDEF), vget_low_s16(vk4x89ABCDEF));
        vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi4x89ABCDEF), vget_high_s16(vk4x89ABCDEF));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
        float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
        float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);

        vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale);
        vfpacc89AB = vmulq_f32(vfpacc89AB, vscale);
        vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscale);

        vacc0123 = vcvtnq_s32_f32(vfpacc0123);
        vacc4567 = vcvtnq_s32_f32(vfpacc4567);
        vacc89AB = vcvtnq_s32_f32(vfpacc89AB);
        vaccCDEF = vcvtnq_s32_f32(vfpaccCDEF);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
        int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);

        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
        vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

        uint8x16_t vout0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc01234567), vacc89ABCDEF);
#else  // !XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
        int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));

        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
        vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

        uint8x16_t vout0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
#endif  // !XNN_ARCH_ARM64

        vout0123456789ABCDEF = vmaxq_u8(vout0123456789ABCDEF, voutput_min);

        vout0123456789ABCDEF = vminq_u8(vout0123456789ABCDEF, voutput_max);

        vst1q_u8(output, vout0123456789ABCDEF); output += 16;
      }

      if XNN_UNLIKELY(c != 0) {
        do {
          int32x4_t vacc0123 = vld1q_s32(b); b += 4;
          int32x4_t vacc4567 = vld1q_s32(b); b += 4;

          const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
          const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));

          const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
          const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));

          const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
          const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));

          const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
          const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));

          const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
          const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const int8_t*) w + 8;

          vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
          vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));

          float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
          float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

          vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
          vfpacc4567 = vmulq_f32(vfpacc4567, vscale);

          vacc0123 = vcvtnq_s32_f32(vfpacc0123);
          vacc4567 = vcvtnq_s32_f32(vfpacc4567);

  #if XNN_ARCH_ARM64
          int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
  #else
          int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
  #endif
          vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

          uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
          vout01234567 = vmax_u8(vout01234567, vget_low_u8(voutput_min));
          vout01234567 = vmin_u8(vout01234567, vget_low_u8(voutput_max));

          if XNN_LIKELY(c >= 8) {
            vst1_u8(output, vout01234567); output += 8;
            c -= 8;
          } else {
            if (c & 4) {
              vst1_lane_u32((void*) output, vreinterpret_u32_u8(vout01234567), 0); output += 4;
              vout01234567 = vext_u8(vout01234567, vout01234567, 4);
            }
            if (c & 2) {
              vst1_lane_u16((void*) output, vreinterpret_u16_u8(vout01234567), 0); output += 2;
              vout01234567 = vext_u8(vout01234567, vout01234567, 2);
            }
            if (c & 1) {
              vst1_lane_u8(output, vout01234567, 0); output += 1;
            }
            c = 0;
          }
        } while (c != 0);
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
