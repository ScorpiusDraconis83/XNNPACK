// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rdminmax/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/simd/f32-scalar.h"


void xnn_f32_rdmax_ukernel_2p2x__scalar_c2(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const void* params)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 2 * input_stride;
  for (; channels >= 2; channels -= 2) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);

    xnn_simd_f32_t vmax0 = xnn_loadu_f32(output);
    xnn_simd_f32_t vmax1 = xnn_loadu_f32((float*)((uintptr_t) output + 1 * sizeof(float)));

    for (int r = rows; r > 0; r -= 2) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = i0;
      }
      xnn_simd_f32_t vin_0_0 = xnn_loadu_f32(&i0[0]);
      xnn_simd_f32_t vin_0_1 = xnn_loadu_f32(&i0[1]);
      xnn_simd_f32_t vin_1_0 = xnn_loadu_f32(&i1[0]);
      xnn_simd_f32_t vin_1_1 = xnn_loadu_f32(&i1[1]);
      vmax0 = xnn_max_f32(vmax0, vin_0_0);
      vmax1 = xnn_max_f32(vmax1, vin_0_1);
      vmax0 = xnn_max_f32(vmax0, vin_1_0);
      vmax1 = xnn_max_f32(vmax1, vin_1_1);

      i0 = (float*) ((uintptr_t) i0 + input_increment);
      i1 = (float*) ((uintptr_t) i1 + input_increment);
    }

    xnn_storeu_f32(output, vmax0);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
    xnn_storeu_f32(output, vmax1);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);

    input = (float*) ((uintptr_t) input + 2 * sizeof(float));
  }
  if (channels != 0) {
    input_increment = 2 * input_stride;
    do {
      const float* i0 = input;
      const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);

      xnn_simd_f32_t vmax;

      if (channels >= xnn_simd_size_f32) {
        vmax = xnn_loadu_f32(output);
      } else {
        vmax = xnn_load_tail_safe_f32(output, channels);
      }

      for (int r = rows; r > 0; r -= 2) {
        if XNN_UNPREDICTABLE(r < 2) {
          i1 = i0;
        }
        xnn_simd_f32_t vin0;
        xnn_simd_f32_t vin1;
        if (channels >= xnn_simd_size_f32) {
          vin0 = xnn_loadu_f32(&i0[0]);
        } else {
          vin0 = xnn_load_tail_safe_f32(&i0[0], channels);
        }
        if (channels >= xnn_simd_size_f32) {
          vin1 = xnn_loadu_f32(&i1[0]);
        } else {
          vin1 = xnn_load_tail_safe_f32(&i1[0], channels);
        }
        vmax = xnn_max_f32(vmax, vin0);
        vmax = xnn_max_f32(vmax, vin1);
        i0 = (float*) ((uintptr_t) i0 + input_increment);
        i1 = (float*) ((uintptr_t) i1 + input_increment);
      }

      if (channels >= xnn_simd_size_f32) {
        xnn_storeu_f32(output, vmax);
        output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
        input = (float*) ((uintptr_t) input + xnn_simd_bytes_f32);
        channels -= xnn_simd_size_f32;
      } else {
        xnn_store_tail_f32(output, vmax, channels);

        channels = 0;
      }
    } while (channels != 0);
  }
}
