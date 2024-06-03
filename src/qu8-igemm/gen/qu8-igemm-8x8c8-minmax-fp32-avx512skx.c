// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx8c8-avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>


void xnn_qu8_igemm_minmax_fp32_ukernel_8x8c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (8 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  uint8_t* c4 = (uint8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  uint8_t* c5 = (uint8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  uint8_t* c6 = (uint8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  uint8_t* c7 = (uint8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    c7 = c6;
  }

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    __m256i vacc1x01 = vacc0x01;
    __m256i vacc1x23 = vacc0x23;
    __m256i vacc1x45 = vacc0x45;
    __m256i vacc1x67 = vacc0x67;
    __m256i vacc2x01 = vacc0x01;
    __m256i vacc2x23 = vacc0x23;
    __m256i vacc2x45 = vacc0x45;
    __m256i vacc2x67 = vacc0x67;
    __m256i vacc3x01 = vacc0x01;
    __m256i vacc3x23 = vacc0x23;
    __m256i vacc3x45 = vacc0x45;
    __m256i vacc3x67 = vacc0x67;
    __m256i vacc4x01 = vacc0x01;
    __m256i vacc4x23 = vacc0x23;
    __m256i vacc4x45 = vacc0x45;
    __m256i vacc4x67 = vacc0x67;
    __m256i vacc5x01 = vacc0x01;
    __m256i vacc5x23 = vacc0x23;
    __m256i vacc5x45 = vacc0x45;
    __m256i vacc5x67 = vacc0x67;
    __m256i vacc6x01 = vacc0x01;
    __m256i vacc6x23 = vacc0x23;
    __m256i vacc6x45 = vacc0x45;
    __m256i vacc6x67 = vacc0x67;
    __m256i vacc7x01 = vacc0x01;
    __m256i vacc7x23 = vacc0x23;
    __m256i vacc7x45 = vacc0x45;
    __m256i vacc7x67 = vacc0x67;
    w = (const int32_t*) w + 8;

    size_t p = ks;
    const __m256i vb_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.kernel_zero_point);
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint8_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint8_t*) ((uintptr_t) a3 + a_offset);
      }
      const uint8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const uint8_t*) ((uintptr_t) a4 + a_offset);
      }
      const uint8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const uint8_t*) ((uintptr_t) a5 + a_offset);
      }
      const uint8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const uint8_t*) ((uintptr_t) a6 + a_offset);
      }
      const uint8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const uint8_t*) ((uintptr_t) a7 + a_offset);
      }
      a += 8;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        const __m256i vxa0 = _mm256_cvtepu8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
        const __m256i vxa1 = _mm256_cvtepu8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
        const __m256i vxa2 = _mm256_cvtepu8_epi16(va2);
        a2 += 8;
        const __m128i va3 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a3));
        const __m256i vxa3 = _mm256_cvtepu8_epi16(va3);
        a3 += 8;
        const __m128i va4 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a4));
        const __m256i vxa4 = _mm256_cvtepu8_epi16(va4);
        a4 += 8;
        const __m128i va5 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a5));
        const __m256i vxa5 = _mm256_cvtepu8_epi16(va5);
        a5 += 8;
        const __m128i va6 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a6));
        const __m256i vxa6 = _mm256_cvtepu8_epi16(va6);
        a6 += 8;
        const __m128i va7 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a7));
        const __m256i vxa7 = _mm256_cvtepu8_epi16(va7);
        a7 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m256i vxb01 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb01), vb_zero_point);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
        vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
        vacc3x01 = _mm256_add_epi32(vacc3x01, _mm256_madd_epi16(vxa3, vxb01));
        vacc4x01 = _mm256_add_epi32(vacc4x01, _mm256_madd_epi16(vxa4, vxb01));
        vacc5x01 = _mm256_add_epi32(vacc5x01, _mm256_madd_epi16(vxa5, vxb01));
        vacc6x01 = _mm256_add_epi32(vacc6x01, _mm256_madd_epi16(vxa6, vxb01));
        vacc7x01 = _mm256_add_epi32(vacc7x01, _mm256_madd_epi16(vxa7, vxb01));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 16));
        const __m256i vxb23 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb23), vb_zero_point);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
        vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
        vacc3x23 = _mm256_add_epi32(vacc3x23, _mm256_madd_epi16(vxa3, vxb23));
        vacc4x23 = _mm256_add_epi32(vacc4x23, _mm256_madd_epi16(vxa4, vxb23));
        vacc5x23 = _mm256_add_epi32(vacc5x23, _mm256_madd_epi16(vxa5, vxb23));
        vacc6x23 = _mm256_add_epi32(vacc6x23, _mm256_madd_epi16(vxa6, vxb23));
        vacc7x23 = _mm256_add_epi32(vacc7x23, _mm256_madd_epi16(vxa7, vxb23));
        const __m128i vb45 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 32));
        const __m256i vxb45 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb45), vb_zero_point);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
        vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
        vacc3x45 = _mm256_add_epi32(vacc3x45, _mm256_madd_epi16(vxa3, vxb45));
        vacc4x45 = _mm256_add_epi32(vacc4x45, _mm256_madd_epi16(vxa4, vxb45));
        vacc5x45 = _mm256_add_epi32(vacc5x45, _mm256_madd_epi16(vxa5, vxb45));
        vacc6x45 = _mm256_add_epi32(vacc6x45, _mm256_madd_epi16(vxa6, vxb45));
        vacc7x45 = _mm256_add_epi32(vacc7x45, _mm256_madd_epi16(vxa7, vxb45));
        const __m128i vb67 = _mm_load_si128((const __m128i*) ((const uint8_t*) w + 48));
        const __m256i vxb67 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(vb67), vb_zero_point);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
        vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
        vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));
        vacc3x67 = _mm256_add_epi32(vacc3x67, _mm256_madd_epi16(vxa3, vxb67));
        vacc4x67 = _mm256_add_epi32(vacc4x67, _mm256_madd_epi16(vxa4, vxb67));
        vacc5x67 = _mm256_add_epi32(vacc5x67, _mm256_madd_epi16(vxa5, vxb67));
        vacc6x67 = _mm256_add_epi32(vacc6x67, _mm256_madd_epi16(vxa6, vxb67));
        vacc7x67 = _mm256_add_epi32(vacc7x67, _mm256_madd_epi16(vxa7, vxb67));

        w = (const void*) ((const uint8_t*) w + 64);
        k += 8 * sizeof(uint8_t);
      }
      p -= 8 * sizeof(void*);
    } while (p != 0);

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);
    const __m256i vacc2x0213 = _mm256_hadd_epi32(vacc2x01, vacc2x23);
    const __m256i vacc2x4657 = _mm256_hadd_epi32(vacc2x45, vacc2x67);
    const __m256i vacc3x0213 = _mm256_hadd_epi32(vacc3x01, vacc3x23);
    const __m256i vacc3x4657 = _mm256_hadd_epi32(vacc3x45, vacc3x67);
    const __m256i vacc4x0213 = _mm256_hadd_epi32(vacc4x01, vacc4x23);
    const __m256i vacc4x4657 = _mm256_hadd_epi32(vacc4x45, vacc4x67);
    const __m256i vacc5x0213 = _mm256_hadd_epi32(vacc5x01, vacc5x23);
    const __m256i vacc5x4657 = _mm256_hadd_epi32(vacc5x45, vacc5x67);
    const __m256i vacc6x0213 = _mm256_hadd_epi32(vacc6x01, vacc6x23);
    const __m256i vacc6x4657 = _mm256_hadd_epi32(vacc6x45, vacc6x67);
    const __m256i vacc7x0213 = _mm256_hadd_epi32(vacc7x01, vacc7x23);
    const __m256i vacc7x4657 = _mm256_hadd_epi32(vacc7x45, vacc7x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);
    const __m256i vacc2x02461357 = _mm256_hadd_epi32(vacc2x0213, vacc2x4657);
    const __m256i vacc3x02461357 = _mm256_hadd_epi32(vacc3x0213, vacc3x4657);
    const __m256i vacc4x02461357 = _mm256_hadd_epi32(vacc4x0213, vacc4x4657);
    const __m256i vacc5x02461357 = _mm256_hadd_epi32(vacc5x0213, vacc5x4657);
    const __m256i vacc6x02461357 = _mm256_hadd_epi32(vacc6x0213, vacc6x4657);
    const __m256i vacc7x02461357 = _mm256_hadd_epi32(vacc7x0213, vacc7x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);
    __m256i vacc2x01234567 = _mm256_permutevar8x32_epi32(vacc2x02461357, vpermute_mask);
    __m256i vacc3x01234567 = _mm256_permutevar8x32_epi32(vacc3x02461357, vpermute_mask);
    __m256i vacc4x01234567 = _mm256_permutevar8x32_epi32(vacc4x02461357, vpermute_mask);
    __m256i vacc5x01234567 = _mm256_permutevar8x32_epi32(vacc5x02461357, vpermute_mask);
    __m256i vacc6x01234567 = _mm256_permutevar8x32_epi32(vacc6x02461357, vpermute_mask);
    __m256i vacc7x01234567 = _mm256_permutevar8x32_epi32(vacc7x02461357, vpermute_mask);

    __m256 vscaled0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vscaled1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vscaled2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);
    __m256 vscaled3x01234567 = _mm256_cvtepi32_ps(vacc3x01234567);
    __m256 vscaled4x01234567 = _mm256_cvtepi32_ps(vacc4x01234567);
    __m256 vscaled5x01234567 = _mm256_cvtepi32_ps(vacc5x01234567);
    __m256 vscaled6x01234567 = _mm256_cvtepi32_ps(vacc6x01234567);
    __m256 vscaled7x01234567 = _mm256_cvtepi32_ps(vacc7x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vscaled0x01234567 = _mm256_mul_ps(vscaled0x01234567, vscale);
    vscaled1x01234567 = _mm256_mul_ps(vscaled1x01234567, vscale);
    vscaled2x01234567 = _mm256_mul_ps(vscaled2x01234567, vscale);
    vscaled3x01234567 = _mm256_mul_ps(vscaled3x01234567, vscale);
    vscaled4x01234567 = _mm256_mul_ps(vscaled4x01234567, vscale);
    vscaled5x01234567 = _mm256_mul_ps(vscaled5x01234567, vscale);
    vscaled6x01234567 = _mm256_mul_ps(vscaled6x01234567, vscale);
    vscaled7x01234567 = _mm256_mul_ps(vscaled7x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vscaled0x01234567 = _mm256_min_ps(vscaled0x01234567, voutput_max_less_zero_point);
    vscaled1x01234567 = _mm256_min_ps(vscaled1x01234567, voutput_max_less_zero_point);
    vscaled2x01234567 = _mm256_min_ps(vscaled2x01234567, voutput_max_less_zero_point);
    vscaled3x01234567 = _mm256_min_ps(vscaled3x01234567, voutput_max_less_zero_point);
    vscaled4x01234567 = _mm256_min_ps(vscaled4x01234567, voutput_max_less_zero_point);
    vscaled5x01234567 = _mm256_min_ps(vscaled5x01234567, voutput_max_less_zero_point);
    vscaled6x01234567 = _mm256_min_ps(vscaled6x01234567, voutput_max_less_zero_point);
    vscaled7x01234567 = _mm256_min_ps(vscaled7x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vscaled0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vscaled1x01234567);
    vacc2x01234567 = _mm256_cvtps_epi32(vscaled2x01234567);
    vacc3x01234567 = _mm256_cvtps_epi32(vscaled3x01234567);
    vacc4x01234567 = _mm256_cvtps_epi32(vscaled4x01234567);
    vacc5x01234567 = _mm256_cvtps_epi32(vscaled5x01234567);
    vacc6x01234567 = _mm256_cvtps_epi32(vscaled6x01234567);
    vacc7x01234567 = _mm256_cvtps_epi32(vscaled7x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc01x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc1x01234567), voutput_zero_point);
    __m256i vacc23x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc2x01234567, vacc3x01234567), voutput_zero_point);
    __m256i vacc45x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc4x01234567, vacc5x01234567), voutput_zero_point);
    __m256i vacc67x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc6x01234567, vacc7x01234567), voutput_zero_point);

    vacc01x01234567 = _mm256_permute4x64_epi64(vacc01x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc23x01234567 = _mm256_permute4x64_epi64(vacc23x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc45x01234567 = _mm256_permute4x64_epi64(vacc45x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc67x01234567 = _mm256_permute4x64_epi64(vacc67x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packus_epi16(vacc01x01234567, vacc23x01234567);

    vout = _mm256_max_epu8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storeh_pi((__m64*) c3, _mm_castsi128_ps(vout_hi));
      _mm_storeh_pi((__m64*) c2, _mm_castsi128_ps(vout_lo));
      _mm_storel_epi64((__m128i*) c1, vout_hi);
      _mm_storel_epi64((__m128i*) c0, vout_lo);

      c7 = (uint8_t*) ((uintptr_t) c7 + cn_stride);
      c6 = (uint8_t*) ((uintptr_t) c6 + cn_stride);
      c5 = (uint8_t*) ((uintptr_t) c5 + cn_stride);
      c4 = (uint8_t*) ((uintptr_t) c4 + cn_stride);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        unaligned_store_u32(c3, (uint32_t) _mm_extract_epi32(vout_hi, 2));
        unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout_lo, 2));
        _mm_storeu_si32(c1, vout_hi);
        _mm_storeu_si32(c0, vout_lo);

        c7 += 4;
        c6 += 4;
        c5 += 4;
        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c3, (uint16_t) _mm_extract_epi16(vout_hi, 4));
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout_lo, 4));
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout_hi, 0));
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));

        c7 += 2;
        c6 += 2;
        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c3 = (uint8_t) _mm_extract_epi8(vout_hi, 8);
        *c2 = (uint8_t) _mm_extract_epi8(vout_lo, 8);
        *c1 = (uint8_t) _mm_extract_epi8(vout_hi, 0);
        *c0 = (uint8_t) _mm_extract_epi8(vout_lo, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
