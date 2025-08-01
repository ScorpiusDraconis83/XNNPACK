// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vunary/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-scalar.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vsqr_ukernel__scalar_u1(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);


  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vx);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx =
        xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vx);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsqr_ukernel__scalar_u2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 2 * xnn_simd_size_f32;

    const xnn_simd_f32_t vy0 = xnn_mul_f32(vx0, vx0);
    const xnn_simd_f32_t vy1 = xnn_mul_f32(vx1, vx1);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    output += 2 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vx);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx =
        xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vx);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vsqr_ukernel__scalar_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 1);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input + 0 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx2 = xnn_loadu_f32(input + 2 * xnn_simd_size_f32);
    const xnn_simd_f32_t vx3 = xnn_loadu_f32(input + 3 * xnn_simd_size_f32);
    input += 4 * xnn_simd_size_f32;

    const xnn_simd_f32_t vy0 = xnn_mul_f32(vx0, vx0);
    const xnn_simd_f32_t vy1 = xnn_mul_f32(vx1, vx1);
    const xnn_simd_f32_t vy2 = xnn_mul_f32(vx2, vx2);
    const xnn_simd_f32_t vy3 = xnn_mul_f32(vx3, vx3);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy3);
    output += 4 * xnn_simd_size_f32;
  }

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vx);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }

  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx =
        xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    const xnn_simd_f32_t vy = xnn_mul_f32(vx, vx);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
