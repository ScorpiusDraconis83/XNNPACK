// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/indirection.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microkernel-type.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/packq.h"
#include "src/xnnpack/quantization.h"
#include <pthreadpool.h>

#if XNN_MAX_UARCH_TYPES > 1
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/microparams-init.h"
#endif  // XNN_MAX_UARCH_TYPES > 1

void xnn_compute_transposec_2d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t tile_i,
                               size_t tile_j) {
  const size_t ld_input = context->input_stride[1];
  const size_t ld_output = context->output_stride[0];
  context->const_size_ukernel(
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1]),
      (void*)((uintptr_t)context->y + j * context->output_stride[1] +
              i * context->output_stride[0]),
      ld_input, ld_output, tile_i, tile_j);
}

void xnn_compute_transposec_3d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t k, size_t tile_j,
                               size_t tile_k) {
  const size_t ld_input = context->input_stride[2];
  const size_t ld_output = context->output_stride[1];
  const void* x =
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1] +
                    k * context->input_stride[2]);
  void* y =
      (void*)((uintptr_t)context->y + i * context->output_stride[0] +
              j * context->output_stride[1] + k * context->output_stride[2]);

  context->const_size_ukernel(x, y, ld_input, ld_output, tile_j, tile_k);
}

void xnn_compute_transposec_4d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t k, size_t l,
                               size_t tile_k, size_t tile_l) {
  const size_t ld_input = context->input_stride[3];
  const size_t ld_output = context->output_stride[2];
  const void* x =
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1] +
                    k * context->input_stride[2] +
                    l * context->input_stride[3]);
  void* y =
      (void*)((uintptr_t)context->y + i * context->output_stride[0] +
              j * context->output_stride[1] + k * context->output_stride[2] +
              l * context->output_stride[3]);

  context->const_size_ukernel(x, y, ld_input, ld_output, tile_k, tile_l);
}

void xnn_compute_transposec_5d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t k, size_t l, size_t m,
                               size_t tile_l, size_t tile_m) {
  const size_t ld_input = context->input_stride[4];
  const size_t ld_output = context->output_stride[3];
  const void* x =
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1] +
                    k * context->input_stride[2] +
                    l * context->input_stride[3] +
                    m * context->input_stride[4]);
  void* y =
      (void*)((uintptr_t)context->y + i * context->output_stride[0] +
              j * context->output_stride[1] + k * context->output_stride[2] +
              l * context->output_stride[3] + m * context->output_stride[4]);

  context->const_size_ukernel(x, y, ld_input, ld_output, tile_l, tile_m);
}

void xnn_compute_transposec_6d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t k, size_t l, size_t m,
                               size_t n, size_t tile_m, size_t tile_n) {
  const size_t ld_input = context->input_stride[5];
  const size_t ld_output = context->output_stride[4];
  const void* x =
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1] +
                    k * context->input_stride[2] +
                    l * context->input_stride[3] +
                    m * context->input_stride[4] +
                    n * context->input_stride[5]);
  void* y =
      (void*)((uintptr_t)context->y + i * context->output_stride[0] +
              j * context->output_stride[1] + k * context->output_stride[2] +
              l * context->output_stride[3] + m * context->output_stride[4] +
              n * context->output_stride[5]);

  context->const_size_ukernel(x, y, ld_input, ld_output, tile_m, tile_n);
}

void xnn_compute_transposev_2d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t tile_i,
                               size_t tile_j) {
  const size_t element_size = context->output_stride[1];
  const size_t ld_input = context->input_stride[1];
  const size_t ld_output = context->output_stride[0];
  const void* x =
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1]);
  void* y = (void*)((uintptr_t)context->y + context->output_stride[1] * j +
                    i * context->output_stride[0]);

  context->variable_size_ukernel(
      x, y, ld_input, ld_output, context->input_stride[0],
      context->output_stride[1], element_size, tile_i, tile_j);
}

void xnn_compute_transposev_3d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t k, size_t tile_j,
                               size_t tile_k) {
  const size_t element_size = context->output_stride[2];
  const size_t ld_input = context->input_stride[2];
  const size_t ld_output = context->output_stride[1];
  const void* x =
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1] +
                    k * context->input_stride[2]);
  void* y =
      (void*)((uintptr_t)context->y + i * context->output_stride[0] +
              j * context->output_stride[1] + k * context->output_stride[2]);

  context->variable_size_ukernel(
      x, y, ld_input, ld_output, context->input_stride[1],
      context->output_stride[2], element_size, tile_j, tile_k);
}

void xnn_compute_transposev_4d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t k, size_t l,
                               size_t tile_k, size_t tile_l) {
  const size_t element_size = context->output_stride[3];
  const size_t ld_input = context->input_stride[3];
  const size_t ld_output = context->output_stride[2];
  const void* x =
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1] +
                    k * context->input_stride[2] +
                    l * context->input_stride[3]);
  void* y =
      (void*)((uintptr_t)context->y + context->output_stride[3] * l +
              i * context->output_stride[0] + j * context->output_stride[1] +
              k * context->output_stride[2]);

  context->variable_size_ukernel(
      x, y, ld_input, ld_output, context->input_stride[2],
      context->output_stride[3], element_size, tile_k, tile_l);
}

void xnn_compute_transposev_5d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t k, size_t l, size_t m,
                               size_t tile_l, size_t tile_m) {
  const size_t element_size = context->output_stride[4];
  const size_t ld_input = context->input_stride[4];
  const size_t ld_output = context->output_stride[3];
  const void* x =
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1] +
                    k * context->input_stride[2] +
                    l * context->input_stride[3] +
                    m * context->input_stride[4]);
  void* y =
      (void*)((uintptr_t)context->y + context->output_stride[4] * m +
              i * context->output_stride[0] + j * context->output_stride[1] +
              k * context->output_stride[2] + l * context->output_stride[3]);

  context->variable_size_ukernel(
      x, y, ld_input, ld_output, context->input_stride[3],
      context->output_stride[4], element_size, tile_l, tile_m);
}

void xnn_compute_transposev_6d(struct transpose_context* restrict context,
                               size_t i, size_t j, size_t k, size_t l, size_t m,
                               size_t n, size_t tile_m, size_t tile_n) {
  const size_t element_size = context->output_stride[5];
  const size_t ld_input = context->input_stride[5];
  const size_t ld_output = context->output_stride[4];
  const void* x =
      (const void*)((uintptr_t)context->x + i * context->input_stride[0] +
                    j * context->input_stride[1] +
                    k * context->input_stride[2] +
                    l * context->input_stride[3] +
                    m * context->input_stride[4] +
                    n * context->input_stride[5]);
  void* y =
      (void*)((uintptr_t)context->y + context->output_stride[5] * n +
              i * context->output_stride[0] + j * context->output_stride[1] +
              k * context->output_stride[2] + l * context->output_stride[3] +
              m * context->output_stride[4]);

  context->variable_size_ukernel(
      x, y, ld_input, ld_output, context->input_stride[4],
      context->output_stride[5], element_size, tile_m, tile_n);
}

void xnn_compute_batched_packw_gemm_gio(
    struct packw_gemm_gio_context* restrict context, size_t batch_index,
    size_t n_block_start, size_t n_block_size) {
  const void* kernel = (const void*)((uintptr_t)context->kernel +
                                     n_block_start * context->n_stride +
                                     batch_index * context->gk_stride);
  const void* bias = context->bias;
  if (bias != NULL) {
    bias = (const void*)((uintptr_t)bias + n_block_start * context->b_stride +
                         batch_index * context->gb_stride);
  }
  void* packed_weights = (void*)((uintptr_t)context->packed_weights +
                                 n_block_start * context->w_stride +
                                 batch_index * context->gc_stride);

  if (context->pack_weights_and_biases) {
    context->pack_weights_and_biases(
        /*flags=*/XNN_FLAG_TRANSPOSE_WEIGHTS, context->gemm_config, context->kc,
        n_block_size, /*groups=*/1, /*block_size=*/0,
        /*k_stride=*/context->k_stride_elements, /*accumulator_init=*/bias,
        kernel, /*init_extra_data0_fn=*/context->init_scale_b,
        /*extra_data0=*/context->scale_b,
        /*extra_data0_element_size=*/context->scale_b_size,
        /*init_extra_data1_fn=*/NULL, /*extra_data1=*/NULL,
        /*extra_data1_element_size=*/0, packed_weights,
        /*params=*/context->params);
  } else {
    context->packw_gemm_gio(
        /*groups=*/1, n_block_size, context->kc, context->nr, context->kr,
        context->sr, context->k_stride_elements, kernel, bias, /*scale=*/NULL,
        packed_weights, /*extra_bytes=*/context->nr * context->scale_b_size,
        /*params=*/context->params);

    if (context->scale_b != NULL) {
      assert(context->init_scale_b != NULL);
      void* weights =
          (void*)((uintptr_t)packed_weights + context->nr * (
              context->w_stride - context->scale_b_size));
      context->init_scale_b(n_block_size, context->nr,
                            context->nr * context->w_stride,
                            context->scale_b, weights);
    }
  }
}

void xnn_compute_packw_gemm_gio(
    struct packw_gemm_gio_context* restrict context, size_t n_block_start,
    size_t n_block_size) {
  xnn_compute_batched_packw_gemm_gio(context, /*batch_index=*/0, n_block_start,
                                     n_block_size);
}

void xnn_compute_batched_packw_gemm_goi(
    struct packw_gemm_goi_context* restrict context, size_t batch_index,
    size_t n_block_start, size_t n_block_size) {
  const void* kernel = (const void*)((uintptr_t)context->kernel +
                                     context->k_stride * n_block_start +
                                     batch_index * context->gk_stride);
  const void* bias = context->bias;
  if (bias != NULL) {
    bias = (const void*)((uintptr_t)bias + n_block_start * context->b_stride +
                         batch_index * context->gb_stride);
  }
  void* packed_weights = (void*)((uintptr_t)context->packed_weights +
                                 context->w_stride * n_block_start +
                                 batch_index * context->gc_stride);

  if (context->pack_weights_and_biases) {
    context->pack_weights_and_biases(
        /*flags=*/0, context->gemm_config, context->kc, n_block_size,
        /*groups=*/1, /*block_size=*/0, /*k_stride=*/context->kc,
        /*accumulator_init=*/bias, kernel,
        /*init_extra_data0_fn=*/context->init_scale_b,
        /*extra_data0=*/context->scale_b,
        /*extra_data0_element_size=*/context->scale_b_size,
        /*init_extra_data1_fn=*/NULL, /*extra_data1=*/NULL,
        /*extra_data1_element_size=*/0, packed_weights,
        /*params=*/context->params);
  } else {
    context->packw_gemm_goi(
        /*groups=*/1, n_block_size, context->kc, context->nr, context->kr,
        context->sr, kernel, bias, /*scale=*/NULL, packed_weights,
        /*extra_bytes=*/context->nr * context->scale_b_size,
        /*params=*/context->params);

    if (context->scale_b != NULL) {
      assert(context->init_scale_b != NULL);
      void* weights =
          (void*)((uintptr_t)packed_weights + context->nr *
                  (context->w_stride - context->scale_b_size));
      context->init_scale_b(n_block_size, context->nr,
                            context->nr * context->w_stride, context->scale_b,
                            weights);
    }
  }
}

void xnn_compute_packw_gemm_goi(struct packw_gemm_goi_context* restrict context,
                                size_t n_block_start, size_t n_block_size) {
  xnn_compute_batched_packw_gemm_goi(context, /*batch_index=*/0, n_block_start,
                                     n_block_size);
}

static void compute_group_indices(struct gemm_context* context,
                                  size_t group_index, size_t* group_index_a,
                                  size_t* group_index_b) {
  const size_t num_batch_dims = context->num_batch_dims;
  *group_index_a = 0;
  *group_index_b = 0;
  for (int k = 0; k < num_batch_dims; k++) {
    // Extract the kth batch index from the group_index.
    const size_t index = group_index / context->batch_strides_c[k];
    group_index %= context->batch_strides_c[k];

    // Compute the corresponding kth group index offsets into A and B.
    *group_index_a = (index % context->batch_dims_a[k]) +
                     context->batch_dims_a[k] * *group_index_a;
    *group_index_b = (index % context->batch_dims_b[k]) +
                     context->batch_dims_b[k] * *group_index_b;
  }
}

void xnn_compute_hmp_grouped_gemm(struct gemm_context* restrict context,
                                  uint32_t uarch_index, size_t group_index,
                                  size_t nr_block_start, size_t mr_block_start,
                                  size_t nr_block_size, size_t mr_block_size) {
  const size_t k_scaled = context->k_scaled;
  const size_t a_stride = context->a_stride;
  const size_t cm_stride = context->cm_stride;
  const size_t group_index_c = group_index;

  // Compute the group index offsets into A and B.
  size_t group_index_a = 0;
  size_t group_index_b = 0;
  compute_group_indices(context, group_index, &group_index_a, &group_index_b);

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    if (context->quantization_params != NULL) {
      // If the effective `mr_block_size` is smaller than the kernel's `mr`,
      // create a padded copy of the dynamic quantization params.
      const struct xnn_qd8_quantization_params* quantization_params =
          &context->quantization_params[group_index_a * context->gq_stride +
                                        mr_block_start];
      struct xnn_qd8_quantization_params padded_quantization_params[XNN_MAX_MR];
      if (mr_step < context->mr) {
        for (size_t i = 0; i < mr_step; i++) {
          padded_quantization_params[i] = quantization_params[i];
        }
        for (size_t i = mr_step; i < context->mr; i++) {
          padded_quantization_params[i] =
              padded_quantization_params[mr_step - 1];
        }
        quantization_params = padded_quantization_params;
      };

      context->dq_ukernel.function[uarch_index](
          mr_step, nr_block_size, k_scaled,
          (const void*)((uintptr_t)context->a + mr_block_start * a_stride +
                        group_index_a * context->ga_stride),
          a_stride,
          (const void*)((uintptr_t)context->packed_w +
                        nr_block_start * context->w_stride +
                        group_index_b * context->gw_stride),
          (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                  (nr_block_start << context->log2_csize) +
                  group_index_c * context->gc_stride),
          cm_stride, context->cn_stride, &context->params, quantization_params);
    } else {
      context->ukernel.function[uarch_index](
          mr_step, nr_block_size, k_scaled,
          (const void*)((uintptr_t)context->a + mr_block_start * a_stride +
                        group_index_a * context->ga_stride),
          a_stride,
          (const void*)((uintptr_t)context->packed_w +
                        nr_block_start * context->w_stride +
                        group_index_b * context->gw_stride),
          (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                  (nr_block_start << context->log2_csize) +
                  group_index_c * context->gc_stride),
          cm_stride, context->cn_stride, &context->params);
    }
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_grouped_gemm(struct gemm_context* restrict context,
                              size_t group_index, size_t nr_block_start,
                              size_t mr_block_start, size_t nr_block_size,
                              size_t mr_block_size) {
  xnn_compute_hmp_grouped_gemm(context, XNN_UARCH_DEFAULT, group_index,
                               nr_block_start, mr_block_start, nr_block_size,
                               mr_block_size);
}

void xnn_compute_gemm(struct gemm_context* restrict context,
                      size_t nr_block_start, size_t mr_block_start,
                      size_t nr_block_size, size_t mr_block_size) {
  const size_t a_stride = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);

    context->ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->k_scaled,
        (const void*)((uintptr_t)context->a + mr_block_start * a_stride),
        a_stride,
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride, context->fused_params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_dqgemm(struct gemm_context* restrict context,
                        size_t nr_block_start, size_t mr_block_start,
                        size_t nr_block_size, size_t mr_block_size) {
  const size_t a_stride = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);

    context->dq_ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->k_scaled,
        (const void*)((uintptr_t)context->a + mr_block_start * a_stride),
        a_stride,
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride, context->fused_params,
        &context->quantization_params[mr_block_start]);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_grouped_qp8gemm(struct gemm_context* restrict context,
                                     uint32_t uarch_index, size_t group_index,
                                     size_t nr_block_start,
                                     size_t mr_block_start,
                                     size_t nr_block_size,
                                     size_t mr_block_size) {
  const size_t cm_stride = context->cm_stride;
  const size_t cn_stride = context->cn_stride;

  // Compute the group index offsets into A and B.
  const size_t group_index_c = group_index;
  size_t group_index_a = 0;
  size_t group_index_b = 0;
  compute_group_indices(context, group_index, &group_index_a, &group_index_b);

  const size_t mr = context->mr;
  const size_t mr_packed = context->mr_packed;
  const size_t kr = context->kr;
  const size_t sr = context->sr;
  const size_t kc = context->kc;
  const size_t k_scaled =
      kc << context->packed_lh_config->log2_packed_element_size;
  const uintptr_t a =
      (uintptr_t)context->a + group_index_a * context->ga_stride;
  const uintptr_t c = (uintptr_t)context->c +
                      group_index_c * context->gc_stride +
                      (nr_block_start << context->log2_csize);
  const void* packed_w = (const void*)((uintptr_t)context->packed_w +
                                       group_index_b * context->gw_stride +
                                       nr_block_start * context->w_stride);
  const uintptr_t packed_input_stride = round_up(kc, kr * sr) * sizeof(int8_t);

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, mr);
    const size_t a_offset = context->packed_lh_config->offset_fn(
        mr_block_start, kc, mr_packed, kr, sr);

    if (context->dynamic_quantization) {
      const void* workspace = (const void*)((uintptr_t)a + a_offset);
      const struct xnn_qd8_quantization_params* quantization_params = workspace;
      const void* packed_inputs =
          (const void*)((uintptr_t)workspace +
                        mr * sizeof(struct xnn_qd8_quantization_params));

      context->dq_ukernel.function[uarch_index](
          mr_step, nr_block_size, k_scaled, packed_inputs, packed_input_stride,
          packed_w, (void*)(c + mr_block_start * cm_stride), cm_stride,
          cn_stride, context->fused_params, quantization_params);
    } else {
      context->qp8_ukernel.function[uarch_index](
          mr_step, nr_block_size, k_scaled, (const void*)(a + a_offset),
          packed_w, (void*)(c + mr_block_start * cm_stride), cm_stride,
          /*dst_stride_col=*/sizeof(float), context->fused_params);
    }
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_grouped_qp8gemm(struct gemm_context* restrict context,
                                 size_t group_index, size_t nr_block_start,
                                 size_t mr_block_start, size_t nr_block_size,
                                 size_t mr_block_size) {
  xnn_compute_hmp_grouped_qp8gemm(context, XNN_UARCH_DEFAULT, group_index,
                                  nr_block_start, mr_block_start, nr_block_size,
                                  mr_block_size);
}

XNN_INLINE static void compute_hmp_qp8gemm(
    struct gemm_context* restrict context, uint32_t uarch_index,
    size_t nr_block_start, size_t mr_block_start, size_t nr_block_size,
    size_t mr_block_size) {
  const size_t cm_stride = context->cm_stride;
  const size_t cn_stride = context->cn_stride;
  const size_t mr_packed = context->mr_packed;
  const size_t mr = context->mr;
  const size_t kr = context->kr;
  const size_t sr = context->sr;
  const size_t kc = context->kc;
  const size_t k_scaled =
      kc << context->packed_lh_config->log2_packed_element_size;
  const uintptr_t a = (uintptr_t)context->a;
  const uintptr_t c =
      (uintptr_t)context->c + (nr_block_start << context->log2_csize);
  const uintptr_t packed_input_stride = round_up(kc, kr * sr) * sizeof(int8_t);
  const void* packed_w = (const void*)((uintptr_t)context->packed_w +
                                       nr_block_start * context->w_stride);

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, mr);
    const size_t a_offset = context->packed_lh_config->offset_fn(
        mr_block_start, context->kc, mr_packed, context->kr, context->sr);

    if (context->dynamic_quantization) {
      const void* workspace = (const void*)((uintptr_t)a + a_offset);
      const struct xnn_qd8_quantization_params* quantization_params = workspace;
      const void* packed_inputs =
          (const void*)((uintptr_t)workspace +
                        mr * sizeof(struct xnn_qd8_quantization_params));

      context->dq_ukernel.function[uarch_index](
          mr_step, nr_block_size, k_scaled, packed_inputs, packed_input_stride,
          packed_w, (void*)(c + mr_block_start * cm_stride), cm_stride,
          cn_stride, context->fused_params, quantization_params);
    } else {
      context->qp8_ukernel.function[uarch_index](
          mr_step, nr_block_size, k_scaled,
          (const void*)((uintptr_t)a + a_offset), packed_w,
          (void*)((uintptr_t)c + mr_block_start * cm_stride), cm_stride,
          /*dst_stride_col=*/sizeof(float), context->fused_params);
    }

    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_qp8gemm(struct gemm_context* restrict context,
                             uint32_t uarch_index, size_t nr_block_start,
                             size_t mr_block_start, size_t nr_block_size,
                             size_t mr_block_size) {
  compute_hmp_qp8gemm(context, uarch_index, nr_block_start, mr_block_start,
                      nr_block_size, mr_block_size);
}

void xnn_compute_qp8gemm(struct gemm_context* restrict context,
                         size_t nr_block_start, size_t mr_block_start,
                         size_t nr_block_size, size_t mr_block_size) {
  compute_hmp_qp8gemm(context, XNN_UARCH_DEFAULT, nr_block_start,
                      mr_block_start, nr_block_size, mr_block_size);
}

void xnn_compute_spmm(struct spmm_context* restrict context, size_t batch_index,
                      size_t mr_block_start, size_t mr_block_size) {
  context->ukernel(
      mr_block_size, context->n,
      (const void*)((uintptr_t)context->input +
                    batch_index * context->batched_input_stride +
                    mr_block_start),
      context->nonzero_weights, context->input_increments,
      context->output_channel_nonzeros,
      (void*)((uintptr_t)context->output +
              batch_index * context->batched_output_stride + mr_block_start),
      context->scaled_m, &context->params);
}

XNN_INLINE static void compute_inline_packed_qp8gemm(
    struct gemm_context* context, uint32_t uarch_index, size_t thread_id,
    size_t group_index_a, size_t group_index_b, size_t group_index_c,
    size_t mr_block_start, size_t mr_block_size) {
  assert(context->packed_lh_config != NULL);
  assert(context->packed_lh_config->offset_fn != NULL);
  assert(context->packed_lh_config->pack_lh_fn != NULL);

  const size_t cm_stride = context->cm_stride;
  const size_t cn_stride = context->cn_stride;
  const size_t mr = context->mr;
  const size_t mr_packed = context->mr_packed;
  const size_t kr = context->kr;
  const size_t sr = context->sr;
  const size_t kc = context->kc;
  const size_t nc = context->nc;
  const uintptr_t a =
      (uintptr_t)context->a + group_index_a * context->ga_stride;
  const size_t a_stride = context->a_stride;
  const void* packed_w = (const void*)((uintptr_t)context->packed_w +
                                       group_index_b * context->gw_stride);
  const uintptr_t c =
      (uintptr_t)context->c + group_index_c * context->gc_stride;
  const size_t k_scaled =
      context->kc << context->packed_lh_config->log2_packed_element_size;
  const uintptr_t packed_input_stride = round_up(kc, kr * sr) * sizeof(int8_t);

  const bool skip_lhs_packing = context->packed_lh_config->gemv_noop && mr == 1;
  void* workspace =
      skip_lhs_packing
          ? NULL
          : (void*)((uintptr_t)context->workspace + context->workspace_offset +
                    context->packed_lh_config->offset_fn(thread_id * mr, kc,
                                                         mr_packed, kr, sr));
  const void* packed_lhs = workspace;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, mr);

    // Pack the `mr_step` rows of the left-hand operand into the workspace.
    if (skip_lhs_packing) {
      packed_lhs = (const void*)(a + mr_block_start * a_stride);
    } else {
      context->packed_lh_config->pack_lh_fn(
          /*m=*/mr_step, kc, mr_packed, kr, sr,
          /*m_idx_start=*/0, (const void*)(a + mr_block_start * a_stride),
          a_stride, workspace);
    }

    // Call the appropriate GEMM kernel.
    if (context->dynamic_quantization) {
      const struct xnn_qd8_quantization_params* quantization_params = packed_lhs;
      const void* packed_inputs =
          (const void*)((uintptr_t)packed_lhs +
                        mr * sizeof(struct xnn_qd8_quantization_params));
      context->dq_ukernel.function[uarch_index](
          mr_step, nc, k_scaled, packed_inputs, packed_input_stride, packed_w,
          (void*)(c + mr_block_start * cm_stride), cm_stride, cn_stride,
          context->fused_params, quantization_params);
    } else {
      context->qp8_ukernel.function[uarch_index](
          mr_step, nc, k_scaled, packed_lhs, packed_w,
          (void*)(c + mr_block_start * cm_stride), cm_stride,
          /*dst_stride_col=*/1 << context->log2_csize, context->fused_params);
    }

    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_inline_packed_qp8gemm(struct gemm_context* context,
                                           uint32_t uarch_index,
                                           size_t thread_id,
                                           size_t mr_block_start,
                                           size_t mr_block_size) {
  compute_inline_packed_qp8gemm(context, uarch_index, thread_id,
                                /*group_index_a=*/0, /*group_index_b=*/0,
                                /*group_index_c=*/0, mr_block_start,
                                mr_block_size);
}

void xnn_compute_inline_packed_qp8gemm(struct gemm_context* context,
                                       uint32_t thread_id,
                                       size_t mr_block_start,
                                       size_t mr_block_size) {
  compute_inline_packed_qp8gemm(context, XNN_UARCH_DEFAULT, thread_id,
                                /*group_index_a=*/0, /*group_index_b=*/0,
                                /*group_index_c=*/0, mr_block_start,
                                mr_block_size);
}

void xnn_compute_hmp_grouped_inline_packed_qp8gemm(
    struct gemm_context* context, uint32_t uarch_index, uint32_t thread_id,
    size_t group_index, size_t mr_block_start, size_t mr_block_size) {
  // Compute the group index offsets into A and B.
  size_t group_index_a = 0;
  size_t group_index_b = 0;
  compute_group_indices(context, group_index, &group_index_a, &group_index_b);

  compute_inline_packed_qp8gemm(context, uarch_index, thread_id, group_index_a,
                                group_index_b, group_index, mr_block_start,
                                mr_block_size);
}

void xnn_compute_grouped_inline_packed_qp8gemm(struct gemm_context* context,
                                               uint32_t thread_id,
                                               size_t group_index,
                                               size_t mr_block_start,
                                               size_t mr_block_size) {
  // Compute the group index offsets into A and B.
  size_t group_index_a = 0;
  size_t group_index_b = 0;
  compute_group_indices(context, group_index, &group_index_a, &group_index_b);

  compute_inline_packed_qp8gemm(context, XNN_UARCH_DEFAULT, thread_id,
                                group_index_a, group_index_b, group_index,
                                mr_block_start, mr_block_size);
}

void xnn_compute_grouped_batch_igemm(struct igemm_context* restrict context,
                                     size_t batch_index, size_t group_index,
                                     size_t nr_block_start,
                                     size_t mr_block_start,
                                     size_t nr_block_size,
                                     size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index * context->gw_stride),
        (void*)((uintptr_t)context->c + group_index * context->gc_stride +
                batch_index * context->bc_stride + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + group_index * context->ga_stride +
            batch_index * context->ba_stride,
        context->zero, &context->params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_dq_zero_buffer_igemm(struct igemm_context* restrict context,
                                      size_t batch_index) {
  memset(context->zero_buffers[batch_index],
         context->quantization_params[batch_index].zero_point,
         context->zero_size);
}

void xnn_compute_dq_zero_buffer_subconv(
    struct subconv_context* restrict context, size_t batch_index) {
  memset(context->zero_buffers[batch_index],
         context->quantization_params[batch_index].zero_point,
         context->zero_size);
}

void xnn_compute_grouped_batch_dqigemm(struct igemm_context* restrict context,
                                       size_t batch_index, size_t group_index,
                                       size_t nr_block_start,
                                       size_t mr_block_start,
                                       size_t nr_block_size,
                                       size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->dq_ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index * context->gw_stride),
        (void*)((uintptr_t)context->c + group_index * context->gc_stride +
                batch_index * context->bc_stride + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + group_index * context->ga_stride +
            batch_index * context->ba_stride,
        context->zero, context->zero_buffers[batch_index], &context->params,
        &context->quantization_params[batch_index]);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_grouped_igemm(struct igemm_context* restrict context,
                               size_t group_index, size_t nr_block_start,
                               size_t mr_block_start, size_t nr_block_size,
                               size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index * context->gw_stride),
        (void*)((uintptr_t)context->c + group_index * context->gc_stride +
                mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + group_index * context->ga_stride, context->zero,
        &context->params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_grouped_dqigemm(struct igemm_context* restrict context,
                                 size_t group_index, size_t nr_block_start,
                                 size_t mr_block_start, size_t nr_block_size,
                                 size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->dq_ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index * context->gw_stride),
        (void*)((uintptr_t)context->c + group_index * context->gc_stride +
                mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + group_index * context->ga_stride, context->zero,
        context->zero_buffers[0], &context->params,
        context->quantization_params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

static void compute_batch_inline_packed_igemm(
    struct igemm_context* restrict context, uint32_t uarch_index,
    uint32_t thread_id, size_t batch_index, size_t group_index,
    size_t mr_block_start, size_t mr_block_size) {
  const size_t mr = context->mr;
  const size_t mr_packed = context->mr_packed;
  const size_t kc = context->kc;
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;
  const size_t a_offset = context->a_offset + batch_index * context->ba_stride +
                          group_index * context->ga_stride;
  const void* packed_w = (const void*)((uintptr_t)context->packed_w +
                                       group_index * context->gw_stride);
  const uintptr_t c = (uintptr_t)context->c + batch_index * context->bc_stride +
                      group_index * context->gc_stride;
  void* workspace =
      (void*)((uintptr_t)context->workspace + context->workspace_offset +
              thread_id * context->per_thread_workspace_size);

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, mr);

    // Pack the LHS data into the workspace.
    context->packed_lh_config->pack_lh_for_igemm_fn(
        mr_step, kc, ks, mr_packed, context->kr, context->sr,
        /*a=*/
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        a_offset, context->zero, workspace);

    // Compute the iGEMM on the packed LHS data.
    context->ukernel.packed_lhs_function[uarch_index](
        mr_step, context->nc, kc, ks, /*packed_lhs=*/workspace, packed_w,
        (void*)(c + mr_block_start * cm_stride), cm_stride, &context->params);

    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_batch_inline_packed_igemm(
    struct igemm_context* restrict context, uint32_t thread_id,
    size_t batch_index, size_t mr_block_start, size_t mr_block_size) {
  compute_batch_inline_packed_igemm(context, XNN_UARCH_DEFAULT, thread_id,
                                    batch_index, /*group_index=*/0,
                                    mr_block_start, mr_block_size);
}

void xnn_compute_batch_hmp_inline_packed_igemm(
    struct igemm_context* restrict context, uint32_t uarch_index,
    size_t thread_id, size_t batch_index, size_t mr_block_start,
    size_t mr_block_size) {
  compute_batch_inline_packed_igemm(context, uarch_index, thread_id,
                                    batch_index, /*group_index=*/0,
                                    mr_block_start, mr_block_size);
}

void xnn_compute_grouped_batch_inline_packed_igemm(
    struct igemm_context* restrict context, uint32_t thread_id,
    size_t batch_index, size_t group_index, size_t mr_block_start,
    size_t mr_block_size) {
  compute_batch_inline_packed_igemm(context, XNN_UARCH_DEFAULT, thread_id,
                                    batch_index, group_index, mr_block_start,
                                    mr_block_size);
}

void xnn_compute_grouped_batch_hmp_inline_packed_igemm(
    struct igemm_context* restrict context, uint32_t uarch_index,
    size_t thread_id, size_t batch_index, size_t group_index,
    size_t mr_block_start, size_t mr_block_size) {
  compute_batch_inline_packed_igemm(context, uarch_index, thread_id,
                                    batch_index, group_index, mr_block_start,
                                    mr_block_size);
}

void xnn_compute_batch_igemm(struct igemm_context* restrict context,
                             size_t batch_index, size_t nr_block_start,
                             size_t mr_block_start, size_t nr_block_size,
                             size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + batch_index * context->bc_stride +
                mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + batch_index * context->ba_stride, context->zero,
        &context->params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_batch_dqigemm(struct igemm_context* restrict context,
                               size_t batch_index, size_t nr_block_start,
                               size_t mr_block_start, size_t nr_block_size,
                               size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->dq_ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + batch_index * context->bc_stride +
                mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + batch_index * context->ba_stride, context->zero,
        context->zero_buffers[batch_index], &context->params,
        &context->quantization_params[batch_index]);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_igemm(struct igemm_context* restrict context,
                       size_t nr_block_start, size_t mr_block_start,
                       size_t nr_block_size, size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride, context->a_offset, context->zero,
        &context->params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_dqigemm(struct igemm_context* restrict context,
                         size_t nr_block_start, size_t mr_block_start,
                         size_t nr_block_size, size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->dq_ukernel.function[XNN_UARCH_DEFAULT](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride, context->a_offset, context->zero,
        context->zero_buffers[0], &context->params,
        &context->quantization_params[/*mr_block_start=*/0]);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

// `output_tile_start` should be a multiple of igemm.mr (tile size).
void xnn_compute_conv2d_igemm_indirection(
    struct conv2d_igemm_indirection_init_context* restrict context,
    size_t output_tile_start, size_t output_tile_size) {
  xnn_indirection_init_conv2d(
      output_tile_size, output_tile_start, output_tile_start + output_tile_size,
      context->indirection_buffer, context->input, context->zero_buffer,
      context->input_pixel_stride, context->input_height, context->input_width,
      context->output_height, context->output_width, context->kernel_height,
      context->kernel_width, context->stride_height, context->stride_width,
      context->dilation_height, context->dilation_width,
      context->input_padding_top, context->input_padding_left);
}

void xnn_compute_grouped_subgemm2d(struct subgemm_context* restrict context,
                                   size_t batch_index, size_t group_index,
                                   size_t subkernel_index, size_t slice_y,
                                   size_t slice_x_start, size_t nc_block_start,
                                   size_t slice_x_max, size_t nc_block_size) {
  const struct subconvolution_params* subconvolution_params =
      &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY (slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY (slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t ax_stride = context->ax_stride;
  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size, nc_block_size, context->kc,
      (const void*)((uintptr_t)context->a + group_index * context->ga_stride +
                    slice_y * context->ay_stride + slice_x_start * ax_stride +
                    batch_index * context->ba_stride),
      ax_stride,
      (const void*)((uintptr_t)subconvolution_params->weights +
                    nc_block_start * subconvolution_params->w_stride +
                    group_index * context->gw_stride),
      (void*)((uintptr_t)subconvolution_params->output +
              group_index * context->gc_stride + slice_y * context->cy_stride +
              slice_x_start * cx_stride + batch_index * context->bc_stride +
              (nc_block_start << context->log2_csize)),
      cx_stride, context->cn_stride, &context->params);
}

void xnn_compute_grouped_subconv2d(struct subconv_context* restrict context,
                                   size_t batch_index, size_t group_index,
                                   size_t subkernel_index, size_t slice_y,
                                   size_t slice_x_start, size_t nc_block_start,
                                   size_t slice_x_max, size_t nc_block_size) {
  const struct subconvolution_params* subconvolution_params =
      &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY (slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY (slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size, nc_block_size, context->kc,
      subconvolution_params->scaled_kernel_size,
      (const void**)((uintptr_t)subconvolution_params->indirection_buffer +
                     slice_y * subconvolution_params->indirection_y_stride +
                     slice_x_start *
                         subconvolution_params->indirection_x_stride),
      (const void*)((uintptr_t)subconvolution_params->weights +
                    nc_block_start * subconvolution_params->w_stride +
                    group_index * context->gw_stride),
      (void*)((uintptr_t)subconvolution_params->output +
              group_index * context->gc_stride + slice_y * context->cy_stride +
              slice_x_start * cx_stride + batch_index * context->bc_stride +
              (nc_block_start << context->log2_csize)),
      cx_stride, context->cn_stride,
      context->a_offset + group_index * context->ga_stride +
          batch_index * context->ba_stride,
      context->zero, &context->params);
}

void xnn_compute_grouped_dqsubconv2d(struct subconv_context* restrict context,
                                     size_t batch_index, size_t group_index,
                                     size_t subkernel_index, size_t slice_y,
                                     size_t slice_x_start,
                                     size_t nc_block_start, size_t slice_x_max,
                                     size_t nc_block_size) {
  const struct subconvolution_params* subconvolution_params =
      &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY (slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY (slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t cx_stride = context->cx_stride;
  context->dq_ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size, nc_block_size, context->kc,
      subconvolution_params->scaled_kernel_size,
      (const void**)((uintptr_t)subconvolution_params->indirection_buffer +
                     slice_y * subconvolution_params->indirection_y_stride +
                     slice_x_start *
                         subconvolution_params->indirection_x_stride),
      (const void*)((uintptr_t)subconvolution_params->weights +
                    nc_block_start * subconvolution_params->w_stride +
                    group_index * context->gw_stride),
      (void*)((uintptr_t)subconvolution_params->output +
              group_index * context->gc_stride + slice_y * context->cy_stride +
              slice_x_start * cx_stride + batch_index * context->bc_stride +
              (nc_block_start << context->log2_csize)),
      cx_stride, context->cn_stride,
      context->a_offset + group_index * context->ga_stride +
          batch_index * context->ba_stride,
      context->zero, context->zero_buffers[batch_index], &context->params,
      &context->quantization_params[batch_index]);
}

void xnn_compute_subconv2d(struct subconv_context* restrict context,
                           size_t batch_index, size_t subkernel_index,
                           size_t slice_y, size_t slice_x_start,
                           size_t nc_block_start, size_t slice_x_max,
                           size_t nc_block_size) {
  const struct subconvolution_params* subconvolution_params =
      &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY (slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY (slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size, nc_block_size, context->kc,
      subconvolution_params->scaled_kernel_size,
      (const void**)((uintptr_t)subconvolution_params->indirection_buffer +
                     slice_y * subconvolution_params->indirection_y_stride +
                     slice_x_start *
                         subconvolution_params->indirection_x_stride),
      (const void*)((uintptr_t)subconvolution_params->weights +
                    nc_block_start * subconvolution_params->w_stride),
      (void*)((uintptr_t)subconvolution_params->output +
              slice_y * context->cy_stride + slice_x_start * cx_stride +
              batch_index * context->bc_stride +
              (nc_block_start << context->log2_csize)),
      cx_stride, context->cn_stride,
      context->a_offset + batch_index * context->ba_stride, context->zero,
      &context->params);
}

void xnn_compute_dqsubconv2d(struct subconv_context* restrict context,
                             size_t batch_index, size_t subkernel_index,
                             size_t slice_y, size_t slice_x_start,
                             size_t nc_block_start, size_t slice_x_max,
                             size_t nc_block_size) {
  const struct subconvolution_params* subconvolution_params =
      &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY (slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY (slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t cx_stride = context->cx_stride;
  context->dq_ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size, nc_block_size, context->kc,
      subconvolution_params->scaled_kernel_size,
      (const void**)((uintptr_t)subconvolution_params->indirection_buffer +
                     slice_y * subconvolution_params->indirection_y_stride +
                     slice_x_start *
                         subconvolution_params->indirection_x_stride),
      (const void*)((uintptr_t)subconvolution_params->weights +
                    nc_block_start * subconvolution_params->w_stride),
      (void*)((uintptr_t)subconvolution_params->output +
              slice_y * context->cy_stride + slice_x_start * cx_stride +
              batch_index * context->bc_stride +
              (nc_block_start << context->log2_csize)),
      cx_stride, context->cn_stride,
      context->a_offset + batch_index * context->ba_stride, context->zero,
      context->zero_buffers[batch_index], &context->params,
      &context->quantization_params[batch_index]);
}

void xnn_compute_conv2d_hwc2chw(struct conv2d_context* restrict context,
                                size_t batch_index, size_t output_y_start,
                                size_t output_y_slice) {
  context->hwc2chw_ukernel(
      context->input_height, context->input_width, output_y_start,
      output_y_start + output_y_slice,
      (const void*)((uintptr_t)context->input +
                    batch_index * context->input_batch_stride),
      context->zero, context->packed_weights,
      (void*)((uintptr_t)context->output +
              batch_index * context->output_batch_stride),
      context->input_padding_top, context->output_channels,
      context->output_height_stride, context->output_channel_stride,
      &context->params);
}

void xnn_compute_dwconv_indirection(
    struct dwconv_indirection_init_context* restrict context,
    size_t output_y_start, size_t output_y_tile) {
  xnn_indirection_init_dwconv2d(
      output_y_start, output_y_start + output_y_tile,
      context->indirection_buffer, context->input, context->input_pixel_stride,
      context->zero_buffer, context->input_height, context->input_width,
      context->output_height, context->output_width, context->kernel_height,
      context->kernel_width, context->stride_height, context->stride_width,
      context->dilation_height, context->dilation_width,
      context->input_padding_top, context->input_padding_left,
      context->step_height, context->step_width, context->tile_size);
}

void xnn_compute_dwconv_unipass(struct dwconv_context* restrict context,
                                size_t batch_index, size_t output_y,
                                size_t output_c_start, size_t output_c_tile) {
  const void** indirect_input =
      (const void**)((uintptr_t)context->indirect_input +
                     output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset +
                              batch_index * context->input_batch_stride +
                              output_c_start * context->input_channel_stride;
  void* output = (void*)((uintptr_t)context->output +
                         batch_index * context->output_batch_stride +
                         output_y * context->output_height_stride +
                         output_c_start * context->output_channel_stride);
  void* weights = (void*)((uintptr_t)context->packed_weights +
                          output_c_start * context->weights_channel_stride);
  const size_t output_increment =
      context->output_pixel_stride -
      output_c_tile * context->output_channel_stride;

  context->ukernel(output_c_tile, context->output_width, indirect_input,
                   weights, output, context->indirect_input_width_stride,
                   output_increment, input_offset, /*input_pixel_stride=*/0,
                   context->zero, &context->params);
}

void xnn_compute_dwconv2d_chw(struct dwconv2d_context* restrict context,
                              size_t batch_index, size_t channel) {
  context->chw_ukernel(context->input_height, context->input_width,
                       (const void*)((uintptr_t)context->input +
                                     channel * context->input_channel_stride +
                                     batch_index * context->input_batch_stride),
                       (const void*)((uintptr_t)context->packed_weights +
                                     channel * context->weights_channel_stride),
                       context->zero,
                       (void*)((uintptr_t)context->output +
                               channel * context->output_channel_stride +
                               batch_index * context->output_batch_stride),
                       context->input_padding_top, &context->params);
}

void xnn_compute_argmax_pooling(struct argmax_pooling_context* restrict context,
                                size_t batch_index, size_t output_y) {
  const void** indirect_input =
      (const void**)((uintptr_t)context->indirect_input +
                     output_y * context->indirect_input_height_stride);
  const size_t input_offset =
      context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*)((uintptr_t)context->output +
                         batch_index * context->output_batch_stride +
                         output_y * context->output_height_stride);
  uint32_t* index = (uint32_t*)((uintptr_t)context->index +
                                batch_index * context->index_batch_stride +
                                output_y * context->index_height_stride);

  context->ukernel(context->output_width, context->pooling_size,
                   context->channels, indirect_input, input_offset,
                   /*input_pixel_stride=*/0, output, index,
                   context->input_increment, context->output_increment,
                   context->index_increment);
}

void xnn_compute_max_pooling(struct max_pooling_context* restrict context,
                             size_t batch_index, size_t output_y) {
  const void** indirect_input =
      (const void**)((uintptr_t)context->indirect_input +
                     output_y * context->indirect_input_height_stride);
  const size_t input_offset =
      context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*)((uintptr_t)context->output +
                         batch_index * context->output_batch_stride +
                         output_y * context->output_height_stride);

  context->ukernel(context->output_width, context->pooling_size,
                   context->channels, indirect_input, input_offset,
                   /*input_pixel_stride=*/0, output, context->input_increment,
                   context->output_increment, &context->params);
}

void xnn_compute_unpooling(struct unpooling_context* restrict context,
                           size_t input_y, size_t input_x) {
  const void* input = (const void*)((uintptr_t)context->input +
                                    input_y * context->input_height_stride +
                                    input_x * context->input_width_stride);
  const uint32_t* index =
      (const uint32_t*)((uintptr_t)context->index +
                        input_y * context->index_height_stride +
                        input_x * context->index_width_stride);
  void** indirect_output =
      (void**)((uintptr_t)context->indirect_output +
               input_y * context->indirect_output_height_stride +
               input_x * context->indirect_output_width_stride);

  context->ukernel(context->pooling_size, context->channels,
                   context->fill_value, input, index, indirect_output);
}

void xnn_compute_average_pooling(
    struct average_pooling_context* restrict context, size_t batch_index,
    size_t output_y) {
  // Refer to xnn_compute_average_pooling for documentation on these terms.
  const size_t indirect_y = min(output_y, context->indirect_top_height) +
                            doz(output_y + 1, context->indirect_bot_start);
  const void** indirect_input =
      (void*)((uintptr_t)context->indirect_input +
              indirect_y * context->indirect_input_height_stride);
  const size_t input_offset_for_compressed_section =
      (output_y - indirect_y) * (output_y < context->indirect_bot_start) *
      context->input_y_stride;
  const size_t input_offset = context->input_offset +
                              batch_index * context->input_batch_stride +
                              input_offset_for_compressed_section;

  const void* pixelwise_buffer =
      context->pixelwise_buffer
          ? (const void*)((uintptr_t)context->pixelwise_buffer +
                          output_y * context->pixelwise_buffer_height_stride)
          : NULL;
  void* output = (void*)((uintptr_t)context->output +
                         batch_index * context->output_batch_stride +
                         output_y * context->output_height_stride);

  context->ukernel(context->output_width, context->pooling_size,
                   context->channels, indirect_input, input_offset,
                   /*input_pixel_stride=*/0, context->zero, pixelwise_buffer,
                   output, context->input_increment, context->output_increment,
                   &context->params);
}

void xnn_compute_resize_bilinear_indirection(
    struct resize_bilinear_nhwc_indirection_init_context* restrict context,
    size_t output_y_start, size_t output_y_tile) {
  void* buffer = context->buffer;

  context->indirection_init(
      output_y_start, output_y_start + output_y_tile,
      context->input_pixel_stride, context->input_height, context->input_width,
      context->output_height, context->output_width, context->input,
      /*indirection_buffer==*/
      (const void**)((uintptr_t)buffer + context->indirect_input_offset),
      /*packed_weights=*/(void*)buffer, context->align_corners,
      context->tensorflow_legacy_mode);
}

void xnn_compute_resize_bilinear(
    struct resize_bilinear_context* restrict context, size_t batch_index,
    size_t pixel_start, size_t pixel_range) {
  void* output = (void*)((uintptr_t)context->output +
                         pixel_start * context->output_pixel_stride +
                         batch_index * context->output_batch_stride);

  context->ukernel(
      pixel_range, context->scaled_channels,
      context->indirect_input + pixel_start * 4,
      context->input_offset + batch_index * context->input_batch_stride,
      (const void*)((uintptr_t)context->packed_weights +
                    (pixel_start << context->log2_wsize)),
      output, context->output_pixel_stride - context->scaled_channels);
}

void xnn_compute_resize_bilinear_chw(
    struct resize_bilinear_chw_context* restrict context, size_t batch_index,
    size_t channel_start, size_t channel_range) {
  void* output = (void*)((uintptr_t)context->output +
                         channel_start * context->output_channel_stride +
                         batch_index * context->output_batch_stride);
  const size_t input_offset = context->input_offset +
                              batch_index * context->input_batch_stride +
                              channel_start * context->input_channel_stride;

  context->ukernel(context->output_pixels, channel_range,
                   context->indirect_input, input_offset,
                   context->packed_weights, output,
                   context->input_channel_stride);
}

void xnn_compute_pad_5d(struct pad_context* restrict context, size_t i,
                        size_t j, size_t k, size_t l, size_t m) {
  const void* input =
      (const void*)((uintptr_t)context->input + i * context->input_stride[4] +
                    j * context->input_stride[3] +
                    k * context->input_stride[2] +
                    l * context->input_stride[1] +
                    m * context->input_stride[0]);
  void* output =
      (void*)((uintptr_t)context->output + i * context->output_stride[4] +
              j * context->output_stride[3] + k * context->output_stride[2] +
              l * context->output_stride[1] + m * context->output_stride[0]);

  const size_t i_padding = context->pre_paddings[5];
  const size_t j_padding = context->pre_paddings[4];
  const size_t k_padding = context->pre_paddings[3];
  const size_t l_padding = context->pre_paddings[2];
  const size_t m_padding = context->pre_paddings[1];

  const size_t i_size = context->input_size[5];
  const size_t j_size = context->input_size[4];
  const size_t k_size = context->input_size[3];
  const size_t l_size = context->input_size[2];
  const size_t m_size = context->input_size[1];

  if XNN_LIKELY (i - i_padding < i_size && j - j_padding < j_size &&
                 k - k_padding < k_size && l - l_padding < l_size &&
                 m - m_padding < m_size) {
    context->pad_ukernel(1 /* rows */, context->input_size[0],
                         context->pre_paddings[0], context->post_paddings[0],
                         input, 0 /* input stride */, output,
                         0 /* output stride */, context->padding_value);
  } else {
    context->fill_ukernel(1 /* rows */, context->output_size[0], output,
                          0 /* output stride */, context->padding_value);
  }
}

void xnn_compute_slice_1d(struct slice_context* restrict context, size_t i) {
  const void* input =
      (const void*)((uintptr_t)context->input + i * context->input_stride[0]);
  void* output =
      (void*)((uintptr_t)context->output + i * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_2d(struct slice_context* restrict context, size_t i,
                          size_t j) {
  const void* input =
      (const void*)((uintptr_t)context->input + i * context->input_stride[1] +
                    j * context->input_stride[0]);
  void* output =
      (void*)((uintptr_t)context->output + i * context->output_stride[1] +
              j * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_3d(struct slice_context* restrict context, size_t i,
                          size_t j, size_t k) {
  const void* input =
      (const void*)((uintptr_t)context->input + i * context->input_stride[2] +
                    j * context->input_stride[1] +
                    k * context->input_stride[0]);
  void* output =
      (void*)((uintptr_t)context->output + i * context->output_stride[2] +
              j * context->output_stride[1] + k * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_4d(struct slice_context* restrict context, size_t i,
                          size_t j, size_t k, size_t l) {
  const void* input =
      (const void*)((uintptr_t)context->input + i * context->input_stride[3] +
                    j * context->input_stride[2] +
                    k * context->input_stride[1] +
                    l * context->input_stride[0]);
  void* output =
      (void*)((uintptr_t)context->output + i * context->output_stride[3] +
              j * context->output_stride[2] + k * context->output_stride[1] +
              l * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_5d(struct slice_context* restrict context, size_t i,
                          size_t j, size_t k, size_t l, size_t m) {
  const void* input =
      (const void*)((uintptr_t)context->input + i * context->input_stride[4] +
                    j * context->input_stride[3] +
                    k * context->input_stride[2] +
                    l * context->input_stride[1] +
                    m * context->input_stride[0]);
  void* output =
      (void*)((uintptr_t)context->output + i * context->output_stride[4] +
              j * context->output_stride[3] + k * context->output_stride[2] +
              l * context->output_stride[1] + m * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_elementwise_binary_1d_tile(
    struct elementwise_binary_context* restrict context, size_t offset,
    size_t count) {
  size_t a_offset = ((context->a_stride[4] == 0 ? 0 : offset));
  size_t b_offset = ((context->b_stride[4] == 0 ? 0 : offset));
  const void* a = (const void*)((uintptr_t)context->a + a_offset);
  const void* b = (const void*)((uintptr_t)context->b + b_offset);
  void* y = (void*)((uintptr_t)context->y + offset);
  context->ukernel(count, a, b, y, &context->params);
}

void xnn_compute_elementwise_binary_1d(
    struct elementwise_binary_context* context, size_t offset, size_t count) {
  for (size_t i = offset; i < offset + count; i++) {
    const void* a =
        (const void*)((uintptr_t)context->a + i * context->a_stride[4]);
    const void* b =
        (const void*)((uintptr_t)context->b + i * context->b_stride[4]);
    void* y = (void*)((uintptr_t)context->y + i * context->y_stride[4]);
    context->ukernel(context->elements, a, b, y, &context->params);
  }
}

void xnn_compute_elementwise_binary_2d(
    struct elementwise_binary_context* context, size_t i, size_t offset,
    size_t count) {
  uintptr_t a = (uintptr_t)context->a + i * context->a_stride[3];
  uintptr_t b = (uintptr_t)context->b + i * context->b_stride[3];
  uintptr_t y = (uintptr_t)context->y + i * context->y_stride[3];
  for (size_t j = offset; j < offset + count; j++) {
    context->ukernel(context->elements,
                     (const void*)(a + j * context->a_stride[4]),
                     (const void*)(b + j * context->b_stride[4]),
                     (void*)(y + j * context->y_stride[4]), &context->params);
  }
}

void xnn_compute_elementwise_binary_3d(
    struct elementwise_binary_context* context, size_t i, size_t offset_j,
    size_t offset_k, size_t count_j, size_t count_k) {
  uintptr_t a = (uintptr_t)context->a + i * context->a_stride[2];
  uintptr_t b = (uintptr_t)context->b + i * context->b_stride[2];
  uintptr_t y = (uintptr_t)context->y + i * context->y_stride[2];
  for (size_t j = offset_j; j < offset_j + count_j; j++) {
    for (size_t k = offset_k; k < offset_k + count_k; k++) {
      context->ukernel(
          context->elements,
          (const void*)(a + j * context->a_stride[3] +
                        k * context->a_stride[4]),
          (const void*)(b + j * context->b_stride[3] +
                        k * context->b_stride[4]),
          (void*)(y + j * context->y_stride[3] + k * context->y_stride[4]),
          &context->params);
    }
  }
}

void xnn_compute_elementwise_binary_4d(
    struct elementwise_binary_context* context, size_t i, size_t j,
    size_t offset_k, size_t offset_l, size_t count_k, size_t count_l) {
  uintptr_t a = (uintptr_t)context->a + +i * context->a_stride[1] +
                j * context->a_stride[2];
  uintptr_t b = (uintptr_t)context->b + i * context->b_stride[1] +
                j * context->b_stride[2];
  uintptr_t y = (uintptr_t)context->y + i * context->y_stride[1] +
                j * context->y_stride[2];
  for (size_t k = offset_k; k < offset_k + count_k; k++) {
    for (size_t l = offset_l; l < offset_l + count_l; l++) {
      context->ukernel(
          context->elements,
          (const void*)(a + k * context->a_stride[3] +
                        l * context->a_stride[4]),
          (const void*)(b + k * context->b_stride[3] +
                        l * context->b_stride[4]),
          (void*)(y + k * context->y_stride[3] + l * context->y_stride[4]),
          &context->params);
    }
  }
}

void xnn_compute_elementwise_binary_5d(
    struct elementwise_binary_context* restrict context, size_t i, size_t j,
    size_t k, size_t l, size_t m) {
  const void* a =
      (const void*)((uintptr_t)context->a + i * context->a_stride[0] +
                    j * context->a_stride[1] + k * context->a_stride[2] +
                    l * context->a_stride[3] + m * context->a_stride[4]);
  const void* b =
      (const void*)((uintptr_t)context->b + i * context->b_stride[0] +
                    j * context->b_stride[1] + k * context->b_stride[2] +
                    l * context->b_stride[3] + m * context->b_stride[4]);
  void* y = (void*)((uintptr_t)context->y + i * context->y_stride[0] +
                    j * context->y_stride[1] + k * context->y_stride[2] +
                    l * context->y_stride[3] + m * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_lut_strided(struct lut_strided_context* restrict context,
                             size_t batch_offset, size_t batch_range) {
  for (size_t batch_index = batch_offset;
       batch_index < batch_offset + batch_range; batch_index++) {
    const void* x =
        (const void*)((uintptr_t)context->x + context->x_stride * batch_index);
    void* y = (void*)((uintptr_t)context->y + context->y_stride * batch_index);

    context->ukernel(context->n, x, y, context->t);
  }
}

void xnn_compute_lut_contiguous(struct lut_contiguous_context* restrict context,
                                size_t offset, size_t size) {
  const void* x = (const void*)((uintptr_t)context->x + offset);
  void* y = (void*)((uintptr_t)context->y + offset);

  context->ukernel(size, x, y, context->t);
}

void xnn_compute_univector_strided(
    struct univector_strided_context* restrict context, size_t batch_index,
    size_t batch_range) {
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;

  const void* x = (const void*)((uintptr_t)context->x + x_stride * batch_index);
  void* y = (void*)((uintptr_t)context->y + y_stride * batch_index);
  do {
    context->ukernel(context->n, x, y, &context->params);
    x = (const void*)((uintptr_t)x + x_stride);
    y = (void*)((uintptr_t)y + y_stride);
  } while (--batch_range != 0);
}

void xnn_compute_univector_contiguous(
    struct univector_contiguous_context* restrict context, size_t offset,
    size_t size) {
  const uint32_t log2_xsize = context->log2_xsize;
  const uint32_t log2_ysize = context->log2_ysize;
  const void* x = (const void*)((uintptr_t)context->x + offset);
  void* y =
      (void*)((uintptr_t)context->y + ((offset >> log2_xsize) << log2_ysize));
  context->ukernel(size, x, y, &context->params);
}

void xnn_compute_contiguous_reduce(
    struct reduce_context* restrict context, size_t output_idx0,
    size_t output_idx1, size_t output_idx2, size_t output1_block_size,
    size_t output2_block_size) {
  assert(output1_block_size == 1);
  const size_t* input_stride = context->input_stride;
  const size_t* output_stride = context->output_stride;

  // input dimensions 1, 3 & 5 are reduced so the entirety of these dimensions
  // are processed so their indices are always 0.
  size_t input_offset = input_stride[0] * output_idx0 +
                        input_stride[2] * output_idx1 +
                        input_stride[4] * output_idx2;
  size_t output_offset =
      (output_stride[0] * output_idx0 + output_stride[1] * output_idx1 +
       output_stride[2] * output_idx2) *
      context->output_element_size;
  size_t workspace_offset =
      (output_stride[0] * output_idx0 + output_stride[1] * output_idx1 +
       output_stride[2] * output_idx2) *
      context->accumulation_element_size;
  int input_shape1 = context->input_shape[1];
  int input_shape3 = context->input_shape[3];

  void* output_ptr = NULL;
  if (context->workspace) {
    output_ptr = context->workspace;
  } else {
    output_ptr = context->output;
  }
  void* output = (void*)((uintptr_t)output_ptr + workspace_offset);
  // Reduce microkernels accumulate into the output buffer.
  if (context->identity_value == 0) {
    memset(output, 0, context->accumulation_element_size * output2_block_size);
  } else {
    context->fill_ukernel(
        1, context->accumulation_element_size * output2_block_size, output,
        context->accumulation_element_size, context->identity_value);
  }

  // Input dimension 1 is reduced.
  for (size_t i = 0; i < input_shape1; ++i) {
    const void* input = (const void*)((uintptr_t)context->input + input_offset);
    // Input dimension 3 is reduced.
    for (size_t j = 0; j < input_shape3; ++j) {
      const void* input_row = input;
      // output2_block_size output elements are written.
      for (size_t k = 0; k < output2_block_size; ++k) {
        // The microkernel reduces input dimension 5.
        context->ukernel.contiguous_reduce(context->channels, input_row, output,
                                           &context->params);
        // input_stride[4] is the number of bytes of input which have been
        // processed by the microkernel call.
        input_row = (const void*)((uintptr_t)input_row + input_stride[4]);
        // Increment output pointer by the number of output bytes which have
        // been written.
        output =
            (void*)((uintptr_t)output + context->accumulation_element_size);
      }
      // Reset the output pointer.
      output = (void*)((uintptr_t)output_ptr + workspace_offset);
      // Iterating over input_shape[3].
      input = (const void*)((uintptr_t)input + input_stride[3]);
    }
    // Iterating over input_shape[1].
    input_offset += input_stride[1];
  }
  // Convert to output datatype if accumulation type != output type.
  if (context->workspace) {
    void* workspace_ptr =
        (void*)((uintptr_t)context->workspace + workspace_offset);
    output_ptr = (void*)((uintptr_t)context->output + output_offset);

    context->cvt_ukernel(
        context->accumulation_element_size * output2_block_size, workspace_ptr,
        output_ptr, &context->cvt_params);
  }
}

void xnn_compute_discontiguous_reduce(struct reduce_context* restrict context,
                                      size_t output_idx0, size_t output_idx1,
                                      size_t output_idx2,
                                      size_t output1_block_size,
                                      size_t output2_block_size) {
  assert(output1_block_size == 1);
  const size_t* input_stride = context->input_stride;
  const size_t* output_stride = context->output_stride;

  // input dimensions 0, 2 & 4 are reduced so the entirety of these dimensions
  // are processed so their indices are always 0.
  size_t input_offset = input_stride[1] * output_idx0 +
                        input_stride[3] * output_idx1 +
                        input_stride[5] * output_idx2;
  size_t output_offset =
      (output_stride[0] * output_idx0 + output_stride[1] * output_idx1 +
       output_stride[2] * output_idx2) *
      context->output_element_size;
  size_t workspace_offset =
      (output_stride[0] * output_idx0 + output_stride[1] * output_idx1 +
       output_stride[2] * output_idx2) *
      context->accumulation_element_size;
  int input_shape0 = context->input_shape[0];
  int input_shape2 = context->input_shape[2];

  void* output_ptr = NULL;
  if (context->workspace) {
    output_ptr = context->workspace;
  } else {
    output_ptr = context->output;
  }
  void* output = (void*)((uintptr_t)output_ptr + workspace_offset);
  // Discontiguous reduce microkernels accumulate into the output buffer.
  if (context->identity_value == 0) {
    memset(output, 0, context->accumulation_element_size * output2_block_size);
  } else {
    context->fill_ukernel(
        1, context->accumulation_element_size * output2_block_size, output,
        context->accumulation_element_size, context->identity_value);
  }

  // Input dimension 0 is reduced.
  for (size_t i = 0; i < input_shape0; ++i) {
    const void* input = (const void*)((uintptr_t)context->input + input_offset);
    // Input dimension 2 is reduced.
    for (size_t j = 0; j < input_shape2; ++j) {
      const void* input_row = input;
      // The microkernel reduces input dimension 4 and iterates over
      // output_block_size elements of dimension 5.
      context->ukernel.discontiguous_reduce(
          context->channels, output2_block_size, input_row, input_stride[4],
          context->zero, output, &context->params);
      // input_stride[4] is the number of bytes of input which have been
      // processed by the microkernel call.
      input_row = (const void*)((uintptr_t)input_row + input_stride[4]);
      // Reset the output pointer.
      output = (void*)((uintptr_t)output_ptr + workspace_offset);
      // Iterating over input_shape[2].
      input = (const void*)((uintptr_t)input + input_stride[2]);
    }
    // Iterating over input_shape[0].
    input_offset += input_stride[0];
  }
  // Convert to output datatype if accumulation type != output type.
  if (context->workspace) {
    void* workspace_ptr =
        (void*)((uintptr_t)context->workspace + workspace_offset);
    output_ptr = (void*)((uintptr_t)context->output + output_offset);

    context->cvt_ukernel(
        context->accumulation_element_size * output2_block_size, workspace_ptr,
        output_ptr, &context->cvt_params);
  }
}

void xnn_compute_pad_qd8_params(
    struct f32_qd8_convert_context* restrict context, size_t batch_index) {
  const size_t batch_size = context->batch_size;
  for (size_t i = 0; i < XNN_EXTRA_QUANTIZATION_PARAMS; ++i) {
    context->quantization_params[batch_size + i].zero_point =
        context->quantization_params[batch_size - 1].zero_point;
    context->quantization_params[batch_size + i].inv_scale =
        context->quantization_params[batch_size - 1].inv_scale;
  }
}

typedef struct xnn_qd8_quantization_params(f16_quantization_params_fn)(
    xnn_float16 min, xnn_float16 max, xnn_float16* f32_scale);
typedef struct xnn_qd8_quantization_params(f32_quantization_params_fn)(
    float min, float max, float* f32_scale);

void xnn_compute_f16_qx8_convert(
    struct f16_qd8_convert_context* restrict context,
    f16_quantization_params_fn quantization_params_function,
    size_t batch_index) {
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;
  const size_t n = context->n;
  const void* input =
      (const void*)((uintptr_t)context->x + x_stride * batch_index);
  void* output = (void*)((uintptr_t)context->y + y_stride * batch_index);

  xnn_float16 minmax[2] = {xnn_float16_from_bits(UINT16_C(0x7c00)),
                           xnn_float16_from_bits(UINT16_C(0xfc00))};
  context->rminmax_ukernel(n, input, minmax, &context->params);
  xnn_float16 f16_scale;
  context->quantization_params[batch_index] =
      quantization_params_function(minmax[0], minmax[1], &f16_scale);

  struct xnn_f16_qs8_cvt_params params;
  params.scalar.scale = f16_scale;
  params.scalar.output_zero_point =
      context->quantization_params[batch_index].zero_point;
  context->convert_ukernel(n, input, output, (union xnn_unary_uparams*)&params);
}

void xnn_compute_f16_qd8_convert(
    struct f16_qd8_convert_context* restrict context, size_t batch_offset,
    size_t batch_range) {
  for (size_t batch_index = batch_offset;
       batch_index < batch_offset + batch_range; batch_index++) {
    xnn_compute_f16_qx8_convert(
        context, xnn_f16_qd8_asymmetric_quantization_params, batch_index);
  }
}

void xnn_compute_f16_qdu8_convert(
    struct f16_qd8_convert_context* restrict context, size_t batch_offset,
    size_t batch_range) {
  for (size_t batch_index = batch_offset;
       batch_index < batch_offset + batch_range; batch_index++) {
    xnn_compute_f16_qx8_convert(
        context, xnn_f16_qdu8_asymmetric_quantization_params, batch_index);
  }
}

void xnn_compute_f32_qx8_convert(
    struct f32_qd8_convert_context* restrict context,
    f32_quantization_params_fn quantization_params_function,
    size_t batch_index) {
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;
  const size_t n = context->n;
  const void* input =
      (const void*)((uintptr_t)context->x + x_stride * batch_index);
  void* output = (void*)((uintptr_t)context->y + y_stride * batch_index);

  float minmax[2] = {INFINITY, -INFINITY};
  context->rminmax_ukernel(n, input, minmax, &context->params);
  float scale;
  context->quantization_params[batch_index] =
      quantization_params_function(minmax[0], minmax[1], &scale);

  struct xnn_f32_qs8_cvt_params params;
  params.scalar.scale = scale;
  params.scalar.output_zero_point =
      context->quantization_params[batch_index].zero_point;
  context->convert_ukernel(n, input, output, (union xnn_unary_uparams*)&params);
}

void xnn_compute_f32_qd8_convert(
    struct f32_qd8_convert_context* restrict context, size_t batch_offset,
    size_t batch_range) {
  for (size_t batch_index = batch_offset;
       batch_index < batch_offset + batch_range; batch_index++) {
    xnn_compute_f32_qx8_convert(
        context, xnn_f32_qd8_asymmetric_quantization_params, batch_index);
  }
}

void xnn_compute_f32_qdu8_convert(
    struct f32_qd8_convert_context* restrict context, size_t batch_offset,
    size_t batch_range) {
  for (size_t batch_index = batch_offset;
       batch_index < batch_offset + batch_range; batch_index++) {
    xnn_compute_f32_qx8_convert(
        context, xnn_f32_qdu8_asymmetric_quantization_params, batch_index);
  }
}

void xnn_compute_pack_lh(struct pack_lh_context* restrict context,
                         size_t group_idx, size_t m_idx_start, size_t tile) {
  const void* lhs =
      (const void*)((uintptr_t)context->lhs + group_idx * context->gi_stride +
                    m_idx_start * context->lhs_stride);
  const size_t offset = context->packed_offset_fn(
      m_idx_start, context->k, context->mr, context->kr, context->sr);
  void* lhs_packed = (void*)((uintptr_t)context->lhs_packed +
                             group_idx * context->gp_stride + offset);

  context->pack_lh_ukernel(/*m=*/tile, context->k, context->mr, context->kr,
                           context->sr, /*m_idx_start=*/0, lhs,
                           context->lhs_stride, lhs_packed);
}

void xnn_compute_f32_qp8_convert(
    struct f32_qp8_convert_context* restrict context, size_t group_idx,
    size_t m_idx_start, size_t m_tile) {
  const size_t m_end = m_idx_start + m_tile;
  while (m_idx_start < m_end) {
    const size_t m_step = min(context->mr, m_end - m_idx_start);
    const float* lhs = (const float*)((const char*)context->lhs +
                                      (group_idx * context->m + m_idx_start) *
                                          context->lhs_stride);
    int8_t* lhs_packed = (int8_t*)((uintptr_t)context->lhs_packed +
                                   group_idx * context->group_stride +
                                   xnn_x8_packq_f32qp8_packed_offset(
                                       m_idx_start, context->k, context->mr,
                                       context->kr, context->sr));

    context->packq_ukernel(/*m=*/m_step, context->k, context->mr, context->kr,
                           context->sr, m_idx_start, lhs, context->lhs_stride,
                           lhs_packed);
    m_idx_start += m_step;
  }
}

void xnn_compute_u8_softmax(struct u8_softmax_context* restrict context,
                            size_t batch_index) {
  const uint8_t* x =
      (const uint8_t*)((uintptr_t)context->x + context->x_stride * batch_index);
  uint8_t* y =
      (uint8_t*)((uintptr_t)context->y + context->y_stride * batch_index);
  const size_t n = context->n;

  uint8_t x_max = 0;
  context->rmax_ukernel(n, x, &x_max, /*params=*/NULL);
  const size_t adjustment = x_max ^ 255;
  const uint32_t* t = (const uint32_t*)context->t + adjustment;
  context->lut_norm_ukernel(n, x, t, y);
}

void xnn_compute_floating_point_softmax(
    struct floating_point_softmax_context* restrict context,
    size_t batch_index) {
  const void* x =
      (const void*)((uintptr_t)context->x + context->x_stride * batch_index);
  void* y = (void*)((uintptr_t)context->y + context->y_stride * batch_index);
  const size_t n = context->n;

  // First pass: reduce-max
  union {
    float as_float;
    xnn_float16 as_half;
  } x_max;
  memcpy(&x_max, &context->rmax_init, sizeof(x_max));
  context->rmax_ukernel(n, x, &x_max, &context->rmax_params);

  // Second pass: reduce-add & store exp(x-x_max)
  union {
    float as_float;
    xnn_float16 as_half;
  } y_sum;
  context->raddstoreexpminusmax_ukernel(n, x, &x_max, y, &y_sum,
                                        &context->expminus_params);

  // Third pass: scale y
  union {
    float as_float;
    xnn_float16 as_half;
  } y_scale;
  context->compute_reciprocal(&y_sum, &y_scale);
  context->vmulc_ukernel(n, y, &y_scale, y, &context->minmax_params);
}

void xnn_compute_vmulcaddc(struct vmulcaddc_context* restrict context,
                           size_t batch_start, size_t batch_size) {
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;

  const void* x = (const void*)((uintptr_t)context->x + x_stride * batch_start);
  void* y = (void*)((uintptr_t)context->y + y_stride * batch_start);

  context->ukernel(batch_size, context->n, x, x_stride, context->w, y, y_stride,
                   &context->params);
}

void xnn_compute_rope(struct rope_context* restrict context, size_t batch_index,
                      size_t head_index, size_t sequence_index) {
  const size_t scaled_channels = context->scaled_channels;
  const size_t offset = batch_index * context->batch_stride +
                        head_index * context->head_stride +
                        sequence_index * context->sequence_stride;
  const void* input = (const void*)((uintptr_t)context->input + offset);
  const void* weights =
      (const void*)((uintptr_t)context->weights +
                    sequence_index * (scaled_channels + scaled_channels));
  void* output = (void*)((uintptr_t)context->output + offset);

  context->vcmul(scaled_channels, input, weights, output, NULL);
}

#if XNN_MAX_UARCH_TYPES > 1
void xnn_compute_hmp_gemm(struct gemm_context* restrict context,
                          uint32_t uarch_index, size_t nr_block_start,
                          size_t mr_block_start, size_t nr_block_size,
                          size_t mr_block_size) {
  const size_t a_stride = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->ukernel.function[uarch_index](
        mr_step, nr_block_size, context->k_scaled,
        (const void*)((uintptr_t)context->a + mr_block_start * a_stride),
        a_stride,
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride, context->fused_params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_dqgemm(struct gemm_context* restrict context,
                            uint32_t uarch_index, size_t nr_block_start,
                            size_t mr_block_start, size_t nr_block_size,
                            size_t mr_block_size) {
  const size_t a_stride = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->dq_ukernel.function[uarch_index](
        mr_step, nr_block_size, context->k_scaled,
        (const void*)((uintptr_t)context->a + mr_block_start * a_stride),
        a_stride,
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride, context->fused_params,
        &context->quantization_params[mr_block_start]);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_grouped_batch_igemm(
    struct igemm_context* restrict context, uint32_t uarch_index,
    size_t batch_index, size_t group_index, size_t nr_block_start,
    size_t mr_block_start, size_t nr_block_size, size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->ukernel.function[uarch_index](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index * context->gw_stride),
        (void*)((uintptr_t)context->c + group_index * context->gc_stride +
                batch_index * context->bc_stride + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + group_index * context->ga_stride +
            batch_index * context->ba_stride,
        context->zero, &context->params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_grouped_batch_dqigemm(
    struct igemm_context* restrict context, uint32_t uarch_index,
    size_t batch_index, size_t group_index, size_t nr_block_start,
    size_t mr_block_start, size_t nr_block_size, size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->dq_ukernel.function[uarch_index](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index * context->gw_stride),
        (void*)((uintptr_t)context->c + group_index * context->gc_stride +
                batch_index * context->bc_stride + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + group_index * context->ga_stride +
            batch_index * context->ba_stride,
        context->zero, context->zero_buffers[batch_index], &context->params,
        &context->quantization_params[batch_index]);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_grouped_igemm(struct igemm_context* restrict context,
                                   uint32_t uarch_index, size_t group_index,
                                   size_t nr_block_start, size_t mr_block_start,
                                   size_t nr_block_size, size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->ukernel.function[uarch_index](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index * context->gw_stride),
        (void*)((uintptr_t)context->c + group_index * context->gc_stride +
                mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + group_index * context->ga_stride, context->zero,
        &context->params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_grouped_dqigemm(struct igemm_context* restrict context,
                                     uint32_t uarch_index, size_t group_index,
                                     size_t nr_block_start,
                                     size_t mr_block_start,
                                     size_t nr_block_size,
                                     size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->dq_ukernel.function[uarch_index](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index * context->gw_stride),
        (void*)((uintptr_t)context->c + group_index * context->gc_stride +
                mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + group_index * context->ga_stride, context->zero,
        context->zero_buffers[0], &context->params,
        (const void*)((uintptr_t)context->quantization_params));
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_batch_hmp_igemm(struct igemm_context* restrict context,
                                 uint32_t uarch_index, size_t batch_index,
                                 size_t nr_block_start, size_t mr_block_start,
                                 size_t nr_block_size, size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->ukernel.function[uarch_index](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + batch_index * context->bc_stride +
                mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + batch_index * context->ba_stride, context->zero,
        &context->params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_batch_hmp_dqigemm(struct igemm_context* restrict context,
                                   uint32_t uarch_index, size_t batch_index,
                                   size_t nr_block_start, size_t mr_block_start,
                                   size_t nr_block_size, size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->dq_ukernel.function[uarch_index](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + batch_index * context->bc_stride +
                mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride,
        context->a_offset + batch_index * context->ba_stride, context->zero,
        context->zero_buffers[batch_index], &context->params,
        &context->quantization_params[batch_index]);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_igemm(struct igemm_context* restrict context,
                           uint32_t uarch_index, size_t nr_block_start,
                           size_t mr_block_start, size_t nr_block_size,
                           size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->ukernel.function[uarch_index](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride, context->a_offset, context->zero,
        &context->params);
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}

void xnn_compute_hmp_dqigemm(struct igemm_context* restrict context,
                             uint32_t uarch_index, size_t nr_block_start,
                             size_t mr_block_start, size_t nr_block_size,
                             size_t mr_block_size) {
  const size_t ks = context->ks;
  const size_t cm_stride = context->cm_stride;

  while (mr_block_size > 0) {
    const size_t mr_step = min(mr_block_size, context->mr);
    context->dq_ukernel.function[uarch_index](
        mr_step, nr_block_size, context->kc, context->ks_scaled,
        (const void**)((uintptr_t)context->indirect_a +
                       mr_block_start * ks * sizeof(void*)),
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize)),
        cm_stride, context->cn_stride, context->a_offset, context->zero,
        context->zero_buffers[0], &context->params,
        (const void*)((uintptr_t)context->quantization_params));
    mr_block_size -= mr_step;
    mr_block_start += mr_step;
  }
}
#endif  // XNN_MAX_UARCH_TYPES > 1

enum xnn_status xnn_run_operator(xnn_operator_t op, pthreadpool_t threadpool) {
  return xnn_run_operator_with_index(op, 0, 0, threadpool);
}

enum xnn_status xnn_run_operator_with_index(xnn_operator_t op,
                                            size_t opdata_index,
                                            size_t operator_object_index,
                                            pthreadpool_t threadpool) {
  switch (op->state) {
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to run operator: operator was not successfully setup");
      return xnn_status_invalid_state;
    case xnn_run_state_ready:
      xnn_log_debug("running operator %zu:%zu (%s %s)", opdata_index,
                    operator_object_index, xnn_operator_type_to_string_v2(op),
                    xnn_microkernel_type_to_string(op->ukernel.type));
      break;
    case xnn_run_state_skip:
      xnn_log_debug("skip running operator %zu:%zu (%s %s)", opdata_index,
                    operator_object_index, xnn_operator_type_to_string_v2(op),
                    xnn_microkernel_type_to_string(op->ukernel.type));
      return xnn_status_success;
    case xnn_run_state_needs_setup:
      xnn_log_error(
          "failed to run operator %zu:%zu (%s %s): operator has been reshaped "
          "but not yet setup",
          opdata_index, operator_object_index,
          xnn_operator_type_to_string_v2(op),
          xnn_microkernel_type_to_string(op->ukernel.type));
      return xnn_status_invalid_state;
  }

  uint32_t flags = PTHREADPOOL_FLAG_DISABLE_DENORMALS;
  if (op->flags & XNN_FLAG_YIELD_WORKERS) {
    flags |= PTHREADPOOL_FLAG_YIELD_WORKERS;
  }
  for (size_t i = 0; i < op->num_compute_invocations; i++) {
    struct compute_parameters* compute = &op->compute[i];
    if (compute->type == xnn_parallelization_type_invalid) {
      break;
    }
    void* context =
        (void*)((uintptr_t)(op->dynamic_context.gemm ? op->dynamic_context.gemm
                                                     : (void*)&op->context) +
                compute->context_offset);
    switch (compute->type) {
      case xnn_parallelization_type_1d:
        assert(compute->range[0] != 0);
        pthreadpool_parallelize_1d(threadpool, compute->task_1d, context,
                                   compute->range[0], flags);
        break;
      case xnn_parallelization_type_1d_with_thread:
        assert(compute->range[0] != 0);
        pthreadpool_parallelize_1d_with_thread(
            threadpool, compute->task_1d_with_thread, context,
            compute->range[0], flags);
        break;
      case xnn_parallelization_type_1d_tile_1d:
        assert(compute->range[0] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_1d_tile_1d(threadpool, compute->task_1d_tile_1d,
                                           context, compute->range[0],
                                           compute->tile[0], flags);
        break;
      case xnn_parallelization_type_1d_tile_1d_dynamic:
        assert(compute->range[0] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_1d_tile_1d_dynamic(
            threadpool, compute->task_1d_tile_1d_dynamic, context,
            compute->range[0], compute->tile[0], flags);
        break;
      case xnn_parallelization_type_1d_tile_1d_dynamic_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_1d_tile_1d_dynamic_with_thread(
            threadpool, compute->task_1d_tile_1d_dynamic_with_id, context,
            compute->range[0], compute->tile[0], flags);
        break;
      case xnn_parallelization_type_2d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        pthreadpool_parallelize_2d(threadpool, compute->task_2d, context,
                                   compute->range[0], compute->range[1], flags);
        break;
      case xnn_parallelization_type_2d_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        pthreadpool_parallelize_2d_with_thread(
            threadpool, compute->task_2d_with_thread, context,
            compute->range[0], compute->range[1], flags);
        break;
      case xnn_parallelization_type_2d_tile_1d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_2d_tile_1d(
            threadpool, compute->task_2d_tile_1d, context, compute->range[0],
            compute->range[1], compute->tile[0], flags);
        break;
      case xnn_parallelization_type_2d_tile_2d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_2d_tile_2d(
            threadpool, compute->task_2d_tile_2d, context, compute->range[0],
            compute->range[1], compute->tile[0], compute->tile[1], flags);
        break;
      case xnn_parallelization_type_2d_tile_1d_dynamic:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_2d_tile_1d_dynamic(
            threadpool, compute->task_2d_tile_1d_dynamic, context,
            compute->range[0], compute->range[1], compute->tile[0], flags);
        break;
      case xnn_parallelization_type_2d_tile_1d_dynamic_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_2d_tile_1d_dynamic_with_thread(
            threadpool, compute->task_2d_tile_1d_dynamic_with_id, context,
            compute->range[0], compute->range[1], compute->tile[0], flags);
        break;
      case xnn_parallelization_type_2d_tile_2d_dynamic:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_2d_tile_2d_dynamic(
            threadpool, compute->task_2d_tile_2d_dynamic, context,
            compute->range[0], compute->range[1], compute->tile[0],
            compute->tile[1], flags);
        break;
      case xnn_parallelization_type_2d_tile_2d_dynamic_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_2d_tile_2d_dynamic_with_thread(
            threadpool, compute->task_2d_tile_2d_dynamic_with_id, context,
            compute->range[0], compute->range[1], compute->tile[0],
            compute->tile[1], flags);
        break;
      case xnn_parallelization_type_3d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        pthreadpool_parallelize_3d(threadpool, compute->task_3d, context,
                                   compute->range[0], compute->range[1],
                                   compute->range[2], flags);
        break;
      case xnn_parallelization_type_3d_tile_1d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_3d_tile_1d(
            threadpool, compute->task_3d_tile_1d, context, compute->range[0],
            compute->range[1], compute->range[2], compute->tile[0], flags);
        break;
      case xnn_parallelization_type_3d_tile_1d_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_3d_tile_1d_with_thread(
            threadpool, compute->task_3d_tile_1d_with_thread, context,
            compute->range[0], compute->range[1], compute->range[2],
            compute->tile[0], flags);
        break;
      case xnn_parallelization_type_3d_tile_1d_dynamic_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_3d_tile_1d_dynamic_with_thread(
            threadpool, compute->task_3d_tile_1d_dynamic_with_id, context,
            compute->range[0], compute->range[1], compute->range[2],
            compute->tile[0], flags);
        break;
      case xnn_parallelization_type_3d_tile_2d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_3d_tile_2d(
            threadpool, compute->task_3d_tile_2d, context, compute->range[0],
            compute->range[1], compute->range[2], compute->tile[0],
            compute->tile[1], flags);
        break;
      case xnn_parallelization_type_3d_tile_2d_dynamic:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_3d_tile_2d_dynamic(
            threadpool, compute->task_3d_tile_2d_dynamic, context,
            compute->range[0], compute->range[1], compute->range[2],
            compute->tile[0], compute->tile[1], flags);
        break;
      case xnn_parallelization_type_3d_tile_2d_dynamic_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_3d_tile_2d_dynamic_with_thread(
            threadpool, compute->task_3d_tile_2d_dynamic_with_id, context,
            compute->range[0], compute->range[1], compute->range[2],
            compute->tile[0], compute->tile[1], flags);
        break;
      case xnn_parallelization_type_4d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->range[3] != 0);
        pthreadpool_parallelize_4d(threadpool, compute->task_4d, context,
                                   compute->range[0], compute->range[1],
                                   compute->range[2], compute->range[3], flags);
        break;
      case xnn_parallelization_type_4d_tile_2d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->range[3] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_4d_tile_2d(
            threadpool, compute->task_4d_tile_2d, context, compute->range[0],
            compute->range[1], compute->range[2], compute->range[3],
            compute->tile[0], compute->tile[1], flags);
        break;
      case xnn_parallelization_type_4d_tile_2d_dynamic:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->range[3] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_4d_tile_2d_dynamic(
            threadpool, compute->task_4d_tile_2d_dynamic, context,
            compute->range[0], compute->range[1], compute->range[2],
            compute->range[3], compute->tile[0], compute->tile[1], flags);
        break;
      case xnn_parallelization_type_5d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->range[3] != 0);
        assert(compute->range[4] != 0);
        pthreadpool_parallelize_5d(threadpool, compute->task_5d, context,
                                   compute->range[0], compute->range[1],
                                   compute->range[2], compute->range[3],
                                   compute->range[4], flags);
        break;
      case xnn_parallelization_type_5d_tile_2d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->range[3] != 0);
        assert(compute->range[4] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_5d_tile_2d(
            threadpool, compute->task_5d_tile_2d, context, compute->range[0],
            compute->range[1], compute->range[2], compute->range[3],
            compute->range[4], compute->tile[0], compute->tile[1], flags);
        break;
      case xnn_parallelization_type_6d_tile_2d:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->range[3] != 0);
        assert(compute->range[4] != 0);
        assert(compute->range[5] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_6d_tile_2d(
            threadpool, compute->task_6d_tile_2d, context, compute->range[0],
            compute->range[1], compute->range[2], compute->range[3],
            compute->range[4], compute->range[5], compute->tile[0],
            compute->tile[1], flags);
        break;
#if XNN_MAX_UARCH_TYPES > 1
      case xnn_parallelization_type_2d_tile_1d_with_uarch:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_2d_tile_1d_with_uarch(
            threadpool, compute->task_2d_tile_1d_with_id, context,
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->tile[0], flags);
        break;
      case xnn_parallelization_type_1d_tile_1d_dynamic_with_uarch_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_1d_tile_1d_dynamic_with_uarch_with_thread(
            threadpool, compute->task_1d_tile_1d_dynamic_with_id_with_thread,
            context,
            /*default_uarch_index=*/0, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->tile[0], flags);
        break;
      case xnn_parallelization_type_2d_tile_1d_dynamic_with_uarch_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_2d_tile_1d_dynamic_with_uarch_with_thread(
            threadpool, compute->task_2d_tile_1d_dynamic_with_id_with_thread,
            context,
            /*default_uarch_index=*/0, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->tile[0], flags);
        break;
      case xnn_parallelization_type_2d_tile_2d_with_uarch:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_2d_tile_2d_with_uarch(
            threadpool, compute->task_2d_tile_2d_with_id, context,
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->tile[0],
            compute->tile[1], flags);
        break;
      case xnn_parallelization_type_2d_tile_2d_dynamic_with_uarch:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_2d_tile_2d_dynamic_with_uarch(
            threadpool, compute->task_2d_tile_2d_dynamic_with_id, context,
            /*default_uarch_index=*/0, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->tile[0],
            compute->tile[1], flags);
        break;
      case xnn_parallelization_type_3d_tile_1d_with_uarch:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_3d_tile_1d_with_uarch(
            threadpool, compute->task_3d_tile_1d_with_id, context,
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->range[2],
            compute->tile[0], flags);
        break;
      case xnn_parallelization_type_3d_tile_1d_with_uarch_with_thread:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
            threadpool, compute->task_3d_tile_1d_with_id_with_thread, context,
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->range[2],
            compute->tile[0], flags);
        break;
      case xnn_parallelization_type_3d_tile_2d_with_uarch:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_3d_tile_2d_with_uarch(
            threadpool, compute->task_3d_tile_2d_with_id, context,
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->range[2],
            compute->tile[0], compute->tile[1], flags);
        break;
      case xnn_parallelization_type_3d_tile_2d_dynamic_with_uarch:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_3d_tile_2d_dynamic_with_uarch(
            threadpool, compute->task_3d_tile_2d_dynamic_with_id, context,
            /*default_uarch_index=*/0, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->range[2],
            compute->tile[0], compute->tile[1], flags);
        break;
      case xnn_parallelization_type_4d_tile_2d_with_uarch:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->range[3] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_4d_tile_2d_with_uarch(
            threadpool, compute->task_4d_tile_2d_with_id, context,
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->range[2],
            compute->range[3], compute->tile[0], compute->tile[1], flags);
        break;
      case xnn_parallelization_type_4d_tile_2d_dynamic_with_uarch:
        assert(compute->range[0] != 0);
        assert(compute->range[1] != 0);
        assert(compute->range[2] != 0);
        assert(compute->range[3] != 0);
        assert(compute->tile[0] != 0);
        assert(compute->tile[1] != 0);
        pthreadpool_parallelize_4d_tile_2d_dynamic_with_uarch(
            threadpool, compute->task_4d_tile_2d_dynamic_with_id, context,
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            compute->range[0], compute->range[1], compute->range[2],
            compute->range[3], compute->tile[0], compute->tile[1], flags);
        break;
#endif  // XNN_MAX_UARCH_TYPES > 1
      default:
        XNN_UNREACHABLE;
    }
  }
  return xnn_status_success;
}
