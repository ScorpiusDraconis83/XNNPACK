// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/s8-rdminmax/simd.c.in
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

#include "src/xnnpack/simd/s8-scalar.h"


void xnn_s8_rdmin_ukernel_2p2x__scalar_c2(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const void* params)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 2 * input_stride;
  for (; channels >= 2; channels -= 2) {
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);

    xnn_simd_s8_t vmin0 = xnn_loadu_s8(output);
    xnn_simd_s8_t vmin1 = xnn_loadu_s8(output + 1 * sizeof(int8_t));;

    for (int r = rows; r > 0; r -= 2) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = i0;
      }
      xnn_simd_s8_t vin_0_0 = xnn_loadu_s8(&i0[0]);
      xnn_simd_s8_t vin_0_1 = xnn_loadu_s8(&i0[1]);
      xnn_simd_s8_t vin_1_0 = xnn_loadu_s8(&i1[0]);
      xnn_simd_s8_t vin_1_1 = xnn_loadu_s8(&i1[1]);
      vmin0 = xnn_min_s8(vmin0, vin_0_0);
      vmin1 = xnn_min_s8(vmin1, vin_0_1);
      vmin0 = xnn_min_s8(vmin0, vin_1_0);
      vmin1 = xnn_min_s8(vmin1, vin_1_1);

      i0 += input_increment;
      i1 += input_increment;
    }

    xnn_storeu_s8(output, vmin0);
    output += xnn_simd_bytes_s8;
    xnn_storeu_s8(output, vmin1);
    output += xnn_simd_bytes_s8;

    input += 2 * sizeof(int8_t);
  }
  if (channels != 0) {
    input_increment = 2 * input_stride;
    do {
      const int8_t* i0 = input;
      const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);

      xnn_simd_s8_t vmin;

      if (channels >= xnn_simd_size_s8) {
        vmin = xnn_loadu_s8(output);
      } else {
        vmin = xnn_load_tail_safe_s8(output, channels);
      }

      for (int r = rows; r > 0; r -= 2) {
        if XNN_UNPREDICTABLE(r < 2) {
          i1 = i0;
        }
        xnn_simd_s8_t vin0 = xnn_loadu_s8(&i0[0]);
        xnn_simd_s8_t vin1 = xnn_loadu_s8(&i1[0]);
        vmin = xnn_min_s8(vmin, vin0);
        vmin = xnn_min_s8(vmin, vin1);
        i0 += input_increment;
        i1 += input_increment;
      }

      if (channels >= xnn_simd_size_s8) {
        xnn_storeu_s8(output, vmin);
        output += xnn_simd_bytes_s8;
        input += xnn_simd_bytes_s8;
        channels -= xnn_simd_size_s8;
      } else {
        xnn_store_tail_s8(output, vmin, channels);

        channels = 0;
      }
    } while (channels != 0);
  }
}
