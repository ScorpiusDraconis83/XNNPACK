// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_PAD_H_
#define XNNPACK_SRC_XNNPACK_PAD_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_PAD_UKERNEL(arch_flags, fn_name, tile_size)                      \
  XNN_INTERNAL void fn_name(                                                 \
      size_t rows, size_t channels, size_t pre_padding, size_t post_padding, \
      const void* input, size_t input_stride, void* output,                  \
      size_t output_stride, const uint32_t fill_pattern);
#include "src/xx-pad/xx-pad.inc"
#undef XNN_PAD_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_PAD_H_
