// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24, 24, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24, 24, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24, 24, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24, 24, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24, 24, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32, 32, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64, 64, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64, 64, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64, 64, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64, 64, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64, 64, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64, 64, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_wasm_blendvps, xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_wasm_blendvps, xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_wasm_blendvps, xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_wasm_blendvps, xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_wasm_blendvps, xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_wasm_blendvps, xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_wasm_blendvps, xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_wasm_blendvps, xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u1, 1, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2, 2, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u1, 1, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2, 2, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u1, 1, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2, 2, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
