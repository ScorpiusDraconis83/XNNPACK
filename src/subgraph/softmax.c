// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


enum xnn_status xnn_define_softmax(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define SoftMax operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define SoftMax operator with input ID #%" PRIu32 ": invalid Value ID",
      input_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define SoftMax operator with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_softmax;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}
