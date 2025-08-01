# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_benchmark",
    "xnnpack_cxx_library",
    "xnnpack_slow_benchmark_tags",
)

xnnpack_cxx_library(
    name = "models",
    testonly = 1,
    srcs = [
        "fp32-attention.cc",
        "fp32-depthwise-separable.cc",
        "fp32-elementwise.cc",
        "fp32-l2-norm.cc",
        "fp32-layer-norm.cc",
        "fp32-mobilenet-v1.cc",
        "fp32-mobilenet-v2.cc",
        "fp32-mobilenet-v3-large.cc",
        "fp32-mobilenet-v3-small.cc",
        "fp32-softmax.cc",
        "fp32-transformer.cc",
        "qd8-attention.cc",
        "qd8-transformer.cc",
        "qs8-mobilenet-v2.cc",
    ],
    hdrs = [
        "models.h",
    ],
    deps = [
        "//:XNNPACK",
    ],
)

xnnpack_benchmark(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    tags = xnnpack_slow_benchmark_tags(),
    deps = [
        ":models",
        "//:allocator",
        "//:subgraph_h",
        "//:xnnpack_h",
        "//bench:bench_utils",
        "@pthreadpool",
    ],
)
