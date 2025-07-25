// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"


void xnn_f16_rmax_ukernel__avx512skx_u48_acc3(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  __m512 vmax0 = _mm512_cvtph_ps(_mm256_set1_epi16(o[0]));
  __m512 vmax1 = vmax0;
  __m512 vmax2 = vmax0;
  for (; batch >= 48 * sizeof(uint16_t); batch -= 48 * sizeof(uint16_t)) {
    const __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    const __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    const __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    i += 48;

    vmax0 = _mm512_max_ps(vmax0, vt0);
    vmax1 = _mm512_max_ps(vmax1, vt1);
    vmax2 = _mm512_max_ps(vmax2, vt2);
  }
  vmax0 = _mm512_max_ps(vmax0, vmax1);
  vmax0 = _mm512_max_ps(vmax0, vmax2);
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;

    vmax0 = _mm512_max_ps(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vmax0 = _mm512_mask_max_ps(vmax0, vmask, vmax0, vt);
  }
  __m256 vmax256 = _mm256_max_ps(_mm512_castps512_ps256(vmax0), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vmax0), 1)));
  __m128 vmax = _mm_max_ps(_mm256_castps256_ps128(vmax256), _mm256_extractf128_ps(vmax256, 1));
  vmax = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
  vmax = _mm_max_ss(vmax, _mm_movehdup_ps(vmax));
  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(vmax, _MM_FROUND_TO_NEAREST_INT), 0);
}
