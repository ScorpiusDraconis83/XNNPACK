// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/microparams.h"


void xnn_f32_dwconv_minmax_ukernel_3p4c__neonfma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
  const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }

    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 4; c -= 4) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      vst1q_f32(output, vacc0123); output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0);
      const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1);
      const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2);
      const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      float32x2_t vacc01 = vget_low_f32(vacc0123);
      if (c & 2) {
        vst1_f32(output, vacc01); output += 2;
        vacc01 = vget_high_f32(vacc0123);
      }
      if (c & 1) {
        vst1_lane_f32(output, vacc01, 0); output += 1;
      }
    }

    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
