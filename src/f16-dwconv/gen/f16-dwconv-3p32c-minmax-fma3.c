// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass-fma3.c.in
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

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f16_dwconv_minmax_ukernel_3p32c__fma3(
    size_t channels,
    size_t output_width,
    const xnn_float16** input,
    const xnn_float16* weights,
    xnn_float16* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const xnn_float16* zero,
    const struct xnn_f16_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmin = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m256 vmax = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.max));
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    input = (const xnn_float16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = (const uint16_t*)weights;
    for (; c >= 32; c -= 32) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
      __m256 vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      __m256 vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      const __m256 vi0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m256 vi0xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 24)));
      i0 += 32;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      const __m256 vk0xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      const __m256 vk0xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xGHIJKLMN, vk0xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0xOPQRSTUV, vk0xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      const __m256 vi1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m256 vi1xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 24)));
      i1 += 32;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 64)));
      const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 72)));
      const __m256 vk1xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 80)));
      const __m256 vk1xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 88)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xGHIJKLMN, vk1xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1xOPQRSTUV, vk1xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      const __m256 vi2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m256 vi2xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 24)));
      i2 += 32;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 96)));
      const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 104)));
      const __m256 vk2xGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 112)));
      const __m256 vk2xOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 120)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));
      vaccGHIJKLMNp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xGHIJKLMN, vk2xGHIJKLMN, vaccGHIJKLMNp0), _MM_FROUND_TO_NEAREST_INT));
      vaccOPQRSTUVp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2xOPQRSTUV, vk2xOPQRSTUV, vaccOPQRSTUVp0), _MM_FROUND_TO_NEAREST_INT));

      w += 128;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      __m256 vaccGHIJKLMN = _mm256_max_ps(vaccGHIJKLMNp0, vmin);
      __m256 vaccOPQRSTUV = _mm256_max_ps(vaccOPQRSTUVp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);
      vaccGHIJKLMN = _mm256_min_ps(vaccGHIJKLMN, vmax);
      vaccOPQRSTUV = _mm256_min_ps(vaccOPQRSTUV, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vaccGHIJKLMN, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vaccOPQRSTUV, _MM_FROUND_TO_NEAREST_INT));
      o += 32;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        _mm_storeu_si32(o, vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    input_offset += input_pixel_stride;
    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}
