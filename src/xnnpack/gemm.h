// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_GEMM_H_
#define XNNPACK_SRC_XNNPACK_GEMM_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)             \
  void fn_name(size_t mr, size_t nc, size_t kc, const uint16_t* a,         \
               size_t a_stride, const void* w, float* c, size_t cm_stride, \
               size_t cn_stride, const struct xnn_f32_minmax_params* params);

DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_1x4c2__scalar)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_1x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_2x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_3x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_4x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_5x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_6x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_7x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_8x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_9x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_10x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_11x16c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_1x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_2x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_3x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_4x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_5x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_6x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_7x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_8x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_9x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_10x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_11x32c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_1x64c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_2x64c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_3x64c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_4x64c2__asm_amd64_avx512bf16_broadcast)
DECLARE_BF16_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_f32_gemm_minmax_ukernel_5x64c2__asm_amd64_avx512bf16_broadcast)

#define DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)              \
  void fn_name(size_t mr, size_t nc, size_t kc, const xnn_bfloat16* a,  \
               size_t a_stride, const xnn_bfloat16* w, xnn_bfloat16* c, \
               size_t cm_stride, size_t cn_stride,                      \
               const struct xnn_bf16_minmax_params* params);

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland)

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip)

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128)

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot)

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal)

#define DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)             \
  void fn_name(size_t mr, size_t nc, size_t kc, const xnn_float16* a, \
               size_t a_stride, const xnn_float16* w, xnn_float16* c, \
               size_t cm_stride, size_t cn_stride,                    \
               const struct xnn_f16_minmax_params* params);

DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64)

DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_5x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_7x8__avx2_broadcast)

DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_f32acc_gemm_minmax_ukernel_1x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_f32acc_gemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_f32acc_gemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_f32acc_gemm_minmax_ukernel_4x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_f32acc_gemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_f32acc_gemm_minmax_ukernel_5x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_f32acc_gemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_f32acc_gemm_minmax_ukernel_6x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_f32acc_gemm_minmax_ukernel_7x8__avx2_broadcast)

DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x32__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x32__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_5x32__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x32__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_7x32__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_8x32__avx512fp16_broadcast)

DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x64__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x64__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_5x64__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x64__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_8x64__avx512fp16_broadcast)

DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_8x8__wasmrelaxedsimd_splat)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_1x16__wasmrelaxedsimd_splat)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_4x16__wasmrelaxedsimd_splat)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_6x16__wasmrelaxedsimd_splat)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f16_gemm_minmax_ukernel_8x16__wasmrelaxedsimd_splat)

#define DECLARE_F32_GEMM_UKERNEL_FUNCTION(fn_name)                           \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const float* a, \
                            size_t a_stride, const float* w, float* c,       \
                            size_t cm_stride, size_t cn_stride,              \
                            const struct xnn_f32_default_params* params);

#define DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(fn_name)                      \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const float* a, \
                            size_t a_stride, const float* w, float* c,       \
                            size_t cm_stride, size_t cn_stride,              \
                            const struct xnn_f32_relu_params* params);

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_mr();
size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_nr();
size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_mr();
size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_nr();

#define DECLARE_PF16_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                  \
                                                                            \
  XNN_INTERNAL size_t fn_name##_get_mr();                                   \
  XNN_INTERNAL size_t fn_name##_get_nr();                                   \
                                                                            \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const void* a, \
                            const void* w, void* c, size_t cm_stride,       \
                            size_t cn_stride,                               \
                            const struct xnn_f16_minmax_params* params);

DECLARE_PF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2)
DECLARE_PF16_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2)

size_t xnn_pf32_gemm_minmax_ukernel_1x32c2__neonsme2_get_mr();
size_t xnn_pf32_gemm_minmax_ukernel_1x32c2__neonsme2_get_nr();
size_t xnn_pf32_gemm_minmax_ukernel_32x32c2__neonsme2_get_mr();
size_t xnn_pf32_gemm_minmax_ukernel_32x32c2__neonsme2_get_nr();

#define DECLARE_PF32_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                  \
                                                                            \
  XNN_INTERNAL size_t fn_name##_get_mr();                                   \
  XNN_INTERNAL size_t fn_name##_get_nr();                                   \
                                                                            \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const void* a, \
                            const float* w, float* c, size_t cm_stride,     \
                            size_t cn_stride,                               \
                            const struct xnn_f32_minmax_params* params);

DECLARE_PF32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2)
DECLARE_PF32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2)

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2_get_mr();
size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2_get_nr();
size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2_get_mr();
size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2_get_nr();

#define DECLARE_PQS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)            \
                                                                           \
  XNN_INTERNAL size_t fn_name##_get_mr();                                  \
  XNN_INTERNAL size_t fn_name##_get_nr();                                  \
                                                                           \
  XNN_INTERNAL void fn_name(                                               \
      size_t m, size_t n, size_t k, const void* lhs_packed, const void* w, \
      void* dst, size_t dst_stride_row, size_t dst_stride_col,             \
      const union xnn_qs8_qc8w_conv_minmax_params* params);

DECLARE_PQS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2)
DECLARE_PQS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2)

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr();
size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_nr();

#define DECLARE_PQS8_QC8W_PACKED_IGEMM_MINMAX_UKERNEL_FUNCTION(fn_name)       \
                                                                              \
  XNN_INTERNAL size_t fn_name##_get_mr();                                     \
  XNN_INTERNAL size_t fn_name##_get_nr();                                     \
                                                                              \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, size_t ks,       \
                            const void* packed_lhs, const void* w, int8_t* c, \
                            size_t cm_stride, const void* params);

DECLARE_PQS8_QC8W_PACKED_IGEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme2)

#define DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                    \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const float* a, \
                            size_t a_stride, const float* w, float* c,       \
                            size_t cm_stride, size_t cn_stride,              \
                            const struct xnn_f32_minmax_params* params);

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x2__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x16__neon_lane_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x16__aarch64_neonfma_lane_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__neon_dup_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__neon_dup_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8s4__neon)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8s4__neon)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8s4__neon)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x8s4__neon)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x8s4__neonfma)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_4x4__asm_aarch32_vfp_ld64)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x4__asm_aarch32_vfp_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__sse_load1)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__sse_load1)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__sse_load1)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__sse_load1)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__sse_load1)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__sse_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__sse_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__sse_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__sse_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__sse_dup)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8s4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8s4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8s4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8s4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8s4__sse)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2c4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x2c4__sse)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x8__avx_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x8__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x8__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x8__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x8__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x8__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x8__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_2)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x16__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__asm_aarch64_neonfma_ld32_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x16__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__asm_aarch64_neonfma_ld64_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x16__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__asm_aarch64_neonfma_ld128_2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__asm_aarch64_neonfma_ld128_2)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_9x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_10x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_11x16c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x32c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x32c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x32c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x32c2__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x32c2__asm_amd64_avx512f_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x64__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x64__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x64__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x64__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x64__asm_amd64_avx512f_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x8__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x8__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x8__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_9x8__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x16__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16__asm_amd64_fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_fma3_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_9x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_10x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_11x16__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_9x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_10x32__asm_amd64_avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_11x32__asm_amd64_avx512f_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_10x8__fma3_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x16s4__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16s4__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x16s4__fma3_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_9x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_10x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_11x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_12x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_13x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_14x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_15x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_16x16__avx512f_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_9x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_10x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_11x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_12x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_13x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_14x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_15x32__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_16x32__avx512f_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_7x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_8x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_9x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_10x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_11x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_12x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_13x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_14x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_15x64__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_16x64__avx512f_broadcast)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8s4__wasmsimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x8s4__wasmsimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8s4__wasmsimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8s4__wasmsimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8s4__wasmsimd)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_3x8s4__wasmsimd)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_4x8s4__wasmsimd)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_5x8s4__wasmsimd)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_6x8s4__wasmsimd)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(
    xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2c4__wasmsimd)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_gemm_relu_ukernel_4x2c4__wasmsimd)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_x86)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x4__scalar)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_2x4__scalar)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2__scalar)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x4__scalar)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x4__scalar)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_2x4__scalar)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x2__scalar)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x4__scalar)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_1x4__scalar)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_2x4__scalar)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x2__scalar)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_gemm_minmax_ukernel_4x4__scalar)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x4v__rvv)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_7x4v__rvv)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x4v__rvv)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_7x4v__rvv)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x4v__rvv)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_7x4v__rvv)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_2x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_7x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_8x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_9x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_10x32__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_2x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_7x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_8x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_9x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_10x64__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x128__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_2x128__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x128__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x128__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x128__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x128__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_7x128__hvx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_8x128__hvx_broadcast)

#define DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)              \
  XNN_INTERNAL void fn_name(size_t mr, size_t nr, size_t k, const float* a, \
                            size_t a_stride, const void* w, float* c,       \
                            size_t cm_stride, size_t cn_stride,             \
                            const struct xnn_f32_qc4w_minmax_params* params);

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_2x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_5x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_7x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_8x16__avx_broadcast)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_2x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_3x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_5x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_7x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_8x16__fma3_broadcast)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_2x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_7x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_8x16__avx2_broadcast)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_2x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_3x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_5x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_7x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_8x32__avx512skx_broadcast)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x8__sse41_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_3x8__sse41_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x8__sse41_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_5x8__sse41_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_6x8__sse41_dup)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x2__scalar)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc4w_gemm_minmax_ukernel_4x4__scalar)

#define DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)       \
  XNN_INTERNAL void fn_name(                                             \
      size_t mr, size_t nr, size_t k, const int8_t* a, size_t a_stride,  \
      const void* w, xnn_float16* c, size_t cm_stride, size_t cn_stride, \
      const struct xnn_f16_qb4w_minmax_params* params,                   \
      const struct xnn_qd8_quantization_params* quantization_params);

DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar)

DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith)

DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith)

DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane_prfm)

DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm)

#define DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(fn_name)                      \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const float* a, \
                            size_t a_stride, const void* w, float* c,        \
                            size_t cm_stride, size_t cn_stride,              \
                            const struct xnn_f32_default_params* params);

#define DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(fn_name)                 \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const float* a, \
                            size_t a_stride, const void* w, float* c,        \
                            size_t cm_stride, size_t cn_stride,              \
                            const struct xnn_f32_relu_params* params);

#define DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)               \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const float* a, \
                            size_t a_stride, const void* w, float* c,        \
                            size_t cm_stride, size_t cn_stride,              \
                            const struct xnn_f32_minmax_params* params);

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse41_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse41_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse41_load1)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse41_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse41_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse41_dup)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__sse41)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x2c4__sse41)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_2x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_7x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_8x16__fma3_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_7x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_8x8__avx2_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx2_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_2x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x16s4__avx2_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx512skx_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_2x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_7x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_8x32__avx512skx_broadcast)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_1x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_3x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_4x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_5x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_6x8s4__wasmsimd)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_1x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_3x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_5x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_6x8s4__wasmsimd)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmsimd_arm)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmsimd_x86)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmsimd_x86)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmsimd_x86)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmsimd_x86)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmsimd_x86)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_ukernel_4x2c4__wasmsimd)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x2c4__wasmsimd)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmsimd_x86)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_1x4__scalar)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_2x4__scalar)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x2__scalar)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x4__scalar)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_1x4__scalar)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_2x4__scalar)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x2__scalar)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_relu_ukernel_4x4__scalar)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar)

#define DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                      \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const uint8_t* a, \
                            size_t a_stride, const void* w, uint8_t* c,        \
                            size_t cm_stride, size_t cn_stride,                \
                            const union xnn_qu8_conv_minmax_params* params);

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu16_ukernel_1x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu16_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu16_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x8c8__avx2)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__avx2)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx256skx)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x1c4__armsimd32)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x1c4__armsimd32)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_1x2__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_1x4__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_2x2__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_2x4__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_3x2__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_3x4__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x2__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_rndnu_ukernel_4x4__scalar)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf)

#define DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)       \
  XNN_INTERNAL void fn_name(                                             \
      size_t mr, size_t nr, size_t k, const int8_t* a, size_t a_stride,  \
      const void* w, xnn_float16* c, size_t cm_stride, size_t cn_stride, \
      const struct xnn_f16_qc4w_minmax_params* params,                   \
      const struct xnn_qd8_quantization_params* quantization_params);

#define DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                            \
      size_t mr, size_t nr, size_t k, const int8_t* a, size_t a_stride, \
      const void* w, float* c, size_t cm_stride, size_t cn_stride,      \
      const struct xnn_f32_qc4w_minmax_params* params,                  \
      const struct xnn_qd8_quantization_params* quantization_params);

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_11x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_11x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x64c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x64c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x64c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x32c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__asm_amd64_avx512vnni)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd_prfm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd_prfm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni_prfm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx2)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx256skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx256skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx256skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld128_2)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128_2)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c4__neondot)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__neondot)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)

    DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
      xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch32_neonmlal_ld64_2)
  DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch32_neonmlal_ld64_2)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512amx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512amx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__avx512amx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x32c4__avx512amx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x64c4__avx512amx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512skx_madd)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512skx_madd_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512skx_madd)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512skx_madd_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x16c8__avx512vnnigfni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x16c8__avx512vnnigfni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2_madd_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256skx_madd_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx2)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx256skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx256skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx256skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld128)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld128)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld128)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld64)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__ssse3_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__ssse3_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__ssse3_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__ssse3_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x4c8__ssse3_madd)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x4c8__ssse3_madd)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__ssse3_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__ssse3_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__ssse3_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__ssse3_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x4c8__ssse3_madd_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x4c8__ssse3_madd_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld64)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld64)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4v__rvv)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4v__rvv)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4v__rvv)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4v__rvv)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x4v__rvv)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x4v__rvv)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x4v__rvv)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x4v__rvv)

#define DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                            \
      size_t mr, size_t nr, size_t k, const int8_t* a, size_t a_stride, \
      const void* w, float* c, size_t cm_stride, size_t cn_stride,      \
      const struct xnn_f32_qb4w_minmax_params* params,                  \
      const struct xnn_qd8_quantization_params* quantization_params);

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x2__scalar)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8__scalar)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x2__scalar)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8__scalar)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4__scalar)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld128)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld128)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld128)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld128)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__avx_ld64)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__avx_ld64)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__avx_ld64)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__avx_ld64)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld128)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld128)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld128)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld128)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse2_ld64)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse2_ld64)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse2_ld64)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse2_ld64)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x8c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x8c4__neondot)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c4__neondot)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16c4__neondot)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_10x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld128)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld128)

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld64)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_2x4c8__sse41_ld64)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld64)
DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld64)

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr();
size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr();
size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr();
size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr();

#define DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                       \
      size_t m, size_t n, size_t k, const void* lhs_packed,        \
      const void* rhs_packed, float* dst, size_t dst_stride_row,   \
      size_t dst_stride_col, struct xnn_f32_minmax_params* minmax_params);

DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x4c16s2__aarch64_neondot)
DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot)
DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x4c8s2__aarch64_neondot)
DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x4c8s2__aarch64_neondot_mstep4)
DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x4c16s2__neoni8mm)
DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x8c16s2__neoni8mm)
DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x4c16s2__neoni8mm_mstep2)
DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__neoni8mm_mstep2)
DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2)
DECLARE_QP8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2)

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr();
size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr();
size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr();
size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr();

#define DECLARE_QP8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                       \
      size_t m, size_t n, size_t k, const void* lhs_packed,        \
      const void* rhs_packed, float* dst, size_t dst_stride_row,   \
      size_t dst_stride_col, struct xnn_f32_minmax_params* minmax_params);

DECLARE_QP8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x4c4__aarch64_neondot_mstep4)
DECLARE_QP8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x4c8__neoni8mm_mstep4)
DECLARE_QP8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x4c4__aarch64_neondot)
DECLARE_QP8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x4c8__aarch64_neondot)
DECLARE_QP8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2)
DECLARE_QP8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2)

#define DECLARE_QP8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                       \
      size_t m, size_t n, size_t k, const void* lhs_packed,        \
      const void* rhs_packed, float* dst, size_t dst_stride_row,   \
      size_t dst_stride_col,                                       \
      const struct xnn_f32_qb4w_minmax_params* minmax_params);

DECLARE_QP8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x4c8s2__aarch64_neondot)
DECLARE_QP8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qb4w_gemm_minmax_ukernel_4x4c8s2__aarch64_neondot)
DECLARE_QP8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x4c16s2__aarch64_neondot)
DECLARE_QP8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot)
DECLARE_QP8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qb4w_gemm_minmax_ukernel_4x8c16s2__neoni8mm)
DECLARE_QP8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qb4w_gemm_minmax_ukernel_8x4c16s2__neoni8mm_mstep2)
DECLARE_QP8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qp8_f32_qb4w_gemm_minmax_ukernel_16x4c16s2__neoni8mm_mstep4)

#define DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)       \
  XNN_INTERNAL void fn_name(                                             \
      size_t mr, size_t nr, size_t k, const int8_t* a, size_t a_stride,  \
      const void* w, xnn_float16* c, size_t cm_stride, size_t cn_stride, \
      const struct xnn_f16_minmax_params* params,                        \
      const struct xnn_qd8_quantization_params* quantization_params);

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c4__neondotfp16arith)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x16c4__neondotfp16arith)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8__asm_aarch32_neonfp16arith_ld64_2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8__asm_aarch32_neonfp16arith_ld64_2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8__asm_aarch32_neonfp16arith_ld64_2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8__asm_aarch32_neonfp16arith_ld64_2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx256skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx256skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx256skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx256skx)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x64c4__avx512amx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x64c4__avx512amx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f16_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm)

#define DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                            \
      size_t mr, size_t nr, size_t k, const int8_t* a, size_t a_stride, \
      const void* w, float* c, size_t cm_stride, size_t cn_stride,      \
      const struct xnn_f32_minmax_params* params,                       \
      const struct xnn_qd8_quantization_params* quantization_params);

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_11x16c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x32c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x32c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x32c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x32c8__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x32c8__asm_amd64_avx512vnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_11x16c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x32c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_11x32c4__asm_amd64_avx512vnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x64c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x64c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x64c4__asm_amd64_avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x64c4__asm_amd64_avx512vnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch32_neonmlal_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__asm_aarch32_neonmlal_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8__asm_aarch32_neonmlal_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch32_neonmlal_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8__asm_aarch32_neonmlal_ld64_2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch64_neondot_ld128_2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128_2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__neondot)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neondot_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neondot_ld64)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neonv8_mlal)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx256skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx256skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx256skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx256skx)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neonv8_mlal)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld128)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512amx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512amx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x16c4__avx512amx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x16c4__avx512amx_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x32c4__avx512amx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x32c4__avx512amx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x32c4__avx512amx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x32c4__avx512amx_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x64c4__avx512amx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x64c4__avx512amx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_12x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_14x16c4__avx512vnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_12x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_14x16c4__avx512vnni_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_12x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_14x16c8__avx512vnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_12x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_14x16c8__avx512vnni_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u2_acc2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u4_acc4)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c16__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c16__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c16__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c16__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c16__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c16__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c16__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c16__wasmusdot)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__wasmusdot)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__wasmusdot_u2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__wasmusdot_u2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__wasmusdot_u2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__wasmusdot_u2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__wasmsdot)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__wasmsdot_u2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__wasmsdot_u2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__wasmsdot_u2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__wasmsdot_u2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__wasmusdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__wasmusdot)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__wasmusdot_u2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__wasmusdot_u2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__scalar)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4v__rvv)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4v__rvv)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4v__rvv)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4v__rvv)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x4v__rvv)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x4v__rvv)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x4v__rvv)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x4v__rvv)

#define DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)           \
  XNN_INTERNAL void fn_name(                                             \
      size_t mr, size_t nc, size_t kc, const int8_t* a, size_t a_stride, \
      const void* w, int8_t* c, size_t cm_stride, size_t cn_stride,      \
      const union xnn_qs8_qc8w_conv_minmax_params* params);

DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld128_2)

DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_7x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_8x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_9x16c8__asm_amd64_avx512vnni)

DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x8c8__avx2_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x8c8__avx2_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x8c8__avx2_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x8c8__avx2_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x8c8__avx2_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x8c8__avx2_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_7x8c8__avx2_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_8x8c8__avx2_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x8c8__avx2_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x8c8__avx2_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x8c8__avx2_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x8c8__avx2_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x8c8__avx2_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x8c8__avx2_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_7x8c8__avx2_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_8x8c8__avx2_madd_prfm)

DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__avx_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__avx_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__avx_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__avx_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__avx_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__avx_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__avx_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__avx_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__avx_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__avx_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__avx_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__avx_madd_prfm)

DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__ssse3_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__ssse3_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__ssse3_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__ssse3_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__ssse3_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__ssse3_madd)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_1x4c8__ssse3_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_2x4c8__ssse3_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_3x4c8__ssse3_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_4x4c8__ssse3_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_5x4c8__ssse3_madd_prfm)
DECLARE_QS8_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_6x4c8__ssse3_madd_prfm)

#define DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)           \
  XNN_INTERNAL void fn_name(                                             \
      size_t mr, size_t nc, size_t kc, const int8_t* a, size_t a_stride, \
      const void* w, int8_t* c, size_t cm_stride, size_t cn_stride,      \
      const union xnn_qs8_qc8w_conv_minmax_params* params);

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld1r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld1r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld1r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld1r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld2r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld2r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neon_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neonv8_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neonv8_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_dup)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_dup)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld1r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld1r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld1r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld2r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld4r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld4r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld4r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld4r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neon_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neonv8_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neonv8_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__neondot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__aarch64_neondot_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neondot_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__aarch64_neondot_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neondot_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mull)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld32_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld64_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_aarch64_neondot_ld128_2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx256skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx256skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx256skx)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512amx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512amx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x16c4__avx512amx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x16c4__avx512amx_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x32c4__avx512amx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x32c4__avx512amx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__avx512amx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__avx512amx_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x64c4__avx512amx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x64c4__avx512amx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c4__avx512vnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c4__avx512vnni_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_11x16c4__asm_amd64_avx512vnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__asm_amd64_avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_11x16c8__asm_amd64_avx512vnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx256vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx256vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx256vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x8c8__avx256vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x8c8__avx256vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x8c8__avx256vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x8c8__avx256vnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnniint8_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnniint8_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x8c8__avx256vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x8c8__avx256vnni_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c16__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c16__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c16__wasmsdot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmusdot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c16__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c16__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c16__wasmusdot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmusdot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmusdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmusdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmusdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmusdot_u2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmsdot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__wasmsdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__wasmsdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__wasmsdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__wasmsdot_u2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmusdot_u2_acc2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmusdot_u2_acc2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmusdot_u2_acc2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__wasmsdot_u2_acc2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__wasmsdot_u2_acc2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__wasmsdot_u2_acc2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c2__wasmsimd_dot16x2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c2__wasmsimd_dot16x2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c2s2__wasmsimd_dot16x2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c2s2__wasmsimd_dot16x2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x1c4__armsimd32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_1x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_2x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_3x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_4x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_6x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_8x16c8__neoni8mm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_1x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_2x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_3x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_4x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_6x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_8x8c8__neoni8mm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4v__rvv)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4v__rvv)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x4v__rvv)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x32c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x32c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x32c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x32c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x32c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x32c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x32c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x32c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x32c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x32c4__hvx)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x64c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x64c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x64c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x64c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x64c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x64c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x64c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x64c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x64c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x64c4__hvx)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x96c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x96c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x96c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x96c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x96c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x96c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x96c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x96c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x96c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x96c4__hvx)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x128c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x128c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x128c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x128c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x128c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x128c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x128c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x128c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x128c4__hvx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x128c4__hvx)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_GEMM_H_
