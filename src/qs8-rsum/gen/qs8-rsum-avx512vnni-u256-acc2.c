// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avx512vnni.c.in
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
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"

void xnn_qs8_rsum_ukernel__avx512vnni_u256_acc2(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params)
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  __m512i vacc0 = _mm512_setzero_si512();
  __m512i vacc1 = _mm512_setzero_si512();
  const __m512i vone = _mm512_set1_epi8(1);
  for (; batch >= 256; batch -= 256) {
    vacc0 = _mm512_dpbusd_epi32(vacc0, vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
    vacc1 = _mm512_dpbusd_epi32(vacc1, vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
    vacc0 = _mm512_dpbusd_epi32(vacc0, vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
    vacc1 = _mm512_dpbusd_epi32(vacc1, vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
  }
  if (XNN_UNLIKELY(batch != 0)) {
    for (; batch >= 64; batch -= 64) {
      vacc0 = _mm512_dpbusd_epi32(vacc0, vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
    }
    if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 63);
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << (batch & 63)) - UINT64_C(1)));
      vacc0 = _mm512_dpbusd_epi32(vacc0, vone, _mm512_maskz_loadu_epi8(vmask, (const __m512i*) input)); input += 64;
    }
  }

  vacc0 = _mm512_add_epi32(vacc0, vacc1);

  int32_t res = _mm512_reduce_add_epi32(vacc0);

  *output += res;
}
