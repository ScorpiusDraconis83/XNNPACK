// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-velu/scalar-rr2-lut16-p3.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_elu_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsat_cutoff = -0x1.154246p+4f;
  const float vmagic_bias = 0x1.800000p19f;
  const float vlog2e = 0x1.715476p+0f;
  const float vminus_ln2_hi = -0x1.62E400p-1f;
  const float vminus_ln2_lo = -0x1.7F7D1Cp-20f;
  const uint32_t vindex_mask = UINT32_C(0xF);
  const float vc3 = 0x1.55561Cp-3f;
  const float vc2 = 0x1.0001ECp-1f;
  const float vone = 1.0f;

  const float vprescale = params->scalar.prescale;
  const float valpha = params->scalar.alpha;
  const float vbeta = params->scalar.beta;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    input += 4;

    const float vz0 = vx0 * vprescale;
    const float vz1 = vx1 * vprescale;
    const float vz2 = vx2 * vprescale;
    const float vz3 = vx3 * vprescale;

    float vn0 = vz0 * vlog2e + vmagic_bias;
    float vn1 = vz1 * vlog2e + vmagic_bias;
    float vn2 = vz2 * vlog2e + vmagic_bias;
    float vn3 = vz3 * vlog2e + vmagic_bias;

    const uint32_t ven0 = float_as_uint32(vn0) << 19;
    const uint32_t vidx0 = float_as_uint32(vn0) & vindex_mask;
    vn0 -= vmagic_bias;
    const uint32_t ven1 = float_as_uint32(vn1) << 19;
    const uint32_t vidx1 = float_as_uint32(vn1) & vindex_mask;
    vn1 -= vmagic_bias;
    const uint32_t ven2 = float_as_uint32(vn2) << 19;
    const uint32_t vidx2 = float_as_uint32(vn2) & vindex_mask;
    vn2 -= vmagic_bias;
    const uint32_t ven3 = float_as_uint32(vn3) << 19;
    const uint32_t vidx3 = float_as_uint32(vn3) & vindex_mask;
    vn3 -= vmagic_bias;

    float vt0 = vn0 * vminus_ln2_hi + vz0;
    float vs0 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx0] + ven0);
    float vt1 = vn1 * vminus_ln2_hi + vz1;
    float vs1 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx1] + ven1);
    float vt2 = vn2 * vminus_ln2_hi + vz2;
    float vs2 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx2] + ven2);
    float vt3 = vn3 * vminus_ln2_hi + vz3;
    float vs3 = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx3] + ven3);

    vt0 = vn0 * vminus_ln2_lo + vt0;
    if XNN_UNPREDICTABLE(vz0 <= vsat_cutoff) {
      vs0 = 0.0f;
      vt0 = 0.0f;
    }
    vt1 = vn1 * vminus_ln2_lo + vt1;
    if XNN_UNPREDICTABLE(vz1 <= vsat_cutoff) {
      vs1 = 0.0f;
      vt1 = 0.0f;
    }
    vt2 = vn2 * vminus_ln2_lo + vt2;
    if XNN_UNPREDICTABLE(vz2 <= vsat_cutoff) {
      vs2 = 0.0f;
      vt2 = 0.0f;
    }
    vt3 = vn3 * vminus_ln2_lo + vt3;
    if XNN_UNPREDICTABLE(vz3 <= vsat_cutoff) {
      vs3 = 0.0f;
      vt3 = 0.0f;
    }

    float vp0 = vc3 * vt0 + vc2;
    float vp1 = vc3 * vt1 + vc2;
    float vp2 = vc3 * vt2 + vc2;
    float vp3 = vc3 * vt3 + vc2;

    vp0 *= vt0;
    vp1 *= vt1;
    vp2 *= vt2;
    vp3 *= vt3;

    vt0 *= vs0;
    vs0 -= vone;
    vt1 *= vs1;
    vs1 -= vone;
    vt2 *= vs2;
    vs2 -= vone;
    vt3 *= vs3;
    vs3 -= vone;

    vp0 = vp0 * vt0 + vt0;
    vp1 = vp1 * vt1 + vt1;
    vp2 = vp2 * vt2 + vt2;
    vp3 = vp3 * vt3 + vt3;

    const float ve0 = (vp0 + vs0) * valpha;
    float vy0 = vx0 * vbeta;
    const float ve1 = (vp1 + vs1) * valpha;
    float vy1 = vx1 * vbeta;
    const float ve2 = (vp2 + vs2) * valpha;
    float vy2 = vx2 * vbeta;
    const float ve3 = (vp3 + vs3) * valpha;
    float vy3 = vx3 * vbeta;

    if XNN_UNPREDICTABLE(vx0 < 0.0f) {
      vy0 = ve0;
    }
    if XNN_UNPREDICTABLE(vx1 < 0.0f) {
      vy1 = ve1;
    }
    if XNN_UNPREDICTABLE(vx2 < 0.0f) {
      vy2 = ve2;
    }
    if XNN_UNPREDICTABLE(vx3 < 0.0f) {
      vy3 = ve3;
    }

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *input++;

      const float vz = vx * vprescale;

      float vn = vz * vlog2e + vmagic_bias;
      const uint32_t ven = float_as_uint32(vn) << 19;
      const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
      vn -= vmagic_bias;

      float vt = vn * vminus_ln2_hi + vz;
      float vs = uint32_as_float(xnn_table_exp2minus_k_over_16[vidx] + ven);

      vt = vn * vminus_ln2_lo + vt;
      if XNN_UNPREDICTABLE(vz <= vsat_cutoff) {
        vs = 0.0f;
        vt = 0.0f;
      }

      float vp = vc3 * vt + vc2;
      vp *= vt;

      vt *= vs;
      vs -= vone;
      vp = vp * vt + vt;
      const float ve = (vp + vs) * valpha;

      float vy = vx * vbeta;
      if XNN_UNPREDICTABLE(vx < 0.0f) {
        vy = ve;
      }

      *output++ = vy;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}
