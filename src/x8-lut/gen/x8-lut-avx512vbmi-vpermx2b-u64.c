// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-lut/avx512vbmi-vpermx2b.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/lut.h"
#include "src/xnnpack/common.h"


void xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u64(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t* restrict table)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vtable0 = _mm512_loadu_si512(table);
  const __m512i vtable1 = _mm512_loadu_si512(table + 64);
  const __m512i vtable2 = _mm512_loadu_si512(table + 128);
  const __m512i vtable3 = _mm512_loadu_si512(table + 192);

  for (; batch >= 64 * sizeof(uint8_t); batch -= 64 * sizeof(uint8_t)) {
    __m512i vx = _mm512_loadu_si512(input);
    input += 64;

    __m512i vy = _mm512_permutex2var_epi8(vtable0, vx, vtable1);
    const __mmask64 vm = _mm512_movepi8_mask(vx);
    const __m512i vt = _mm512_permutex2var_epi8(vtable2, vx, vtable3);
    vy = _mm512_mask_mov_epi8(vy, vm, vt);

    _mm512_storeu_si512(output, vy);
    output += 64;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch < 64);
    const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << batch) - UINT64_C(1)));

    __m512i vx = _mm512_maskz_loadu_epi8(vmask, input);

    __m512i vy = _mm512_maskz_permutex2var_epi8(vmask, vtable0, vx, vtable1);
    const __mmask64 vm = _mm512_movepi8_mask(vx);
    const __m512i vt = _mm512_maskz_permutex2var_epi8(vmask, vtable2, vx, vtable3);
    vy = _mm512_mask_mov_epi8(vy, vm, vt);

    _mm512_mask_storeu_epi8(output, vmask, vy);
  }
}
