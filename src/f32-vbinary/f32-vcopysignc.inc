// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__scalar_u1, 1, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__scalar_u2, 2, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__scalar_u4, 4, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__scalar_u8, 8, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__sse2_u4, 4, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__sse2_u8, 8, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__sse2_u12, 12, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__sse2_u16, 16, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vcopysignc_ukernel__avx_u8, 8, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vcopysignc_ukernel__avx_u16, 16, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vcopysignc_ukernel__avx_u24, 24, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vcopysignc_ukernel__avx_u32, 32, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vcopysignc_ukernel__avx512f_u16, 16, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vcopysignc_ukernel__avx512f_u32, 32, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vcopysignc_ukernel__avx512f_u48, 48, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vcopysignc_ukernel__avx512f_u64, 64, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__wasmsimd_u4, 4, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__wasmsimd_u8, 8, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__wasmsimd_u12, 12, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__wasmsimd_u16, 16, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__hvx_u32, 32, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__hvx_u64, 64, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__hvx_u96, 96, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f32_vcopysignc_ukernel__hvx_u128, 128, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
#endif  // XNN_ARCH_HEXAGON && XNN_ENABLE_HVX

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vcopysignc_ukernel__neon_u4, 4, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vcopysignc_ukernel__neon_u8, 8, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vcopysignc_ukernel__neon_u12, 12, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vcopysignc_ukernel__neon_u16, 16, false, float, struct xnn_f32_default_params, ((xnn_init_f32_default_params_fn) NULL))
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

