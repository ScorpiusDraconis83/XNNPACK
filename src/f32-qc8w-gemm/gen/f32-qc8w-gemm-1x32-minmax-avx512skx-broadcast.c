// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/avx512-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>
#include <smmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/microparams.h"


void xnn_f32_qc8w_gemm_minmax_ukernel_1x32__avx512skx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  do {
    __m512 vacc0x0 = _mm512_loadu_ps(w);
    __m512 vacc0x1 = _mm512_loadu_ps((const float*) w + 16);
    w = (const float*) w + 32;

    size_t k = kc;
    do {
      const __m512i vbi0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) w));
      const __m512i vbi1  = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16)));
      const __m512 vb0  = _mm512_cvtepi32_ps(vbi0);
      const __m512 vb1  = _mm512_cvtepi32_ps(vbi1);
      w = (const int8_t*) w + 32;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0 = _mm512_fmadd_ps(va0, vb0, vacc0x0);
      vacc0x1 = _mm512_fmadd_ps(va0, vb1, vacc0x1);

      a0 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vscale0 = _mm512_loadu_ps((const float*) w + 0);
    vacc0x0 = _mm512_mul_ps(vacc0x0, vscale0);
    const __m512 vscale1 = _mm512_loadu_ps((const float*) w + 16);
    vacc0x1 = _mm512_mul_ps(vacc0x1, vscale1);
    w = (const float*) w + 32;
    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0 = _mm512_max_ps(vmin, vacc0x0);
    vacc0x1 = _mm512_max_ps(vmin, vacc0x1);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0 = _mm512_min_ps(vmax, vacc0x0);
    vacc0x1 = _mm512_min_ps(vmax, vacc0x1);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c0, vacc0x0);
      _mm512_storeu_ps(c0 + 16, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 32;
    } else {
      // NC remainder (1..31)
      assert(nc >= 1);
      assert(nc <= 31);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 0));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 16));

      _mm512_mask_storeu_ps(c0 + 0, vmask0, vacc0x0);
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vacc0x1);
      nc = 0;
    }
  } while (nc != 0);
}
