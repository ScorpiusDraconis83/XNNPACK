# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for neonfma
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_NEONFMA_MICROKERNEL_SRCS
  src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-neonfma-acc2.c
  src/f32-gemm/gen/f32-gemm-1x8s4-minmax-neonfma.c
  src/f32-gemm/gen/f32-gemm-4x8s4-minmax-neonfma.c
  src/f32-gemm/gen/f32-gemm-6x8s4-minmax-neonfma.c
  src/f32-ibilinear-chw/gen/f32-ibilinear-chw-neonfma-p8.c
  src/f32-ibilinear/gen/f32-ibilinear-neonfma-c8.c
  src/f32-igemm/gen/f32-igemm-1x8s4-minmax-neonfma.c
  src/f32-igemm/gen/f32-igemm-4x8s4-minmax-neonfma.c
  src/f32-igemm/gen/f32-igemm-6x8s4-minmax-neonfma.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u16-acc2.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-neonfma-pipelined.c
  src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u16.c
  src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u8.c
  src/f32-vmulcaddc/gen/f32-vmulcaddc-c4-minmax-neonfma-2x.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u16.c)

SET(NON_PROD_NEONFMA_MICROKERNEL_SRCS
  src/bf16-gemm/gen/bf16-gemm-1x4c8-minmax-neonfma-shland.c
  src/bf16-gemm/gen/bf16-gemm-1x4c8-minmax-neonfma-zip.c
  src/bf16-gemm/gen/bf16-gemm-2x4c8-minmax-neonfma-shland.c
  src/bf16-gemm/gen/bf16-gemm-2x4c8-minmax-neonfma-zip.c
  src/bf16-gemm/gen/bf16-gemm-3x4c8-minmax-neonfma-shland.c
  src/bf16-gemm/gen/bf16-gemm-3x4c8-minmax-neonfma-zip.c
  src/bf16-gemm/gen/bf16-gemm-4x4c8-minmax-neonfma-shland.c
  src/bf16-gemm/gen/bf16-gemm-4x4c8-minmax-neonfma-zip.c
  src/bf16-gemm/gen/bf16-gemm-5x4c8-minmax-neonfma-shland.c
  src/bf16-gemm/gen/bf16-gemm-5x4c8-minmax-neonfma-zip.c
  src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-neonfma.c
  src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-neonfma-acc2.c
  src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-neonfma.c
  src/f32-gemm/gen/f32-gemm-1x8-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen/f32-gemm-4x8-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen/f32-gemm-4x8-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen/f32-gemm-6x8-minmax-neonfma-dup-ld64.c
  src/f32-gemm/gen/f32-gemm-6x8-minmax-neonfma-dup-ld128.c
  src/f32-gemm/gen/f32-gemm-8x8s4-minmax-neonfma.c
  src/f32-ibilinear-chw/gen/f32-ibilinear-chw-neonfma-p4.c
  src/f32-ibilinear-chw/gen/f32-ibilinear-chw-neonfma-p16.c
  src/f32-ibilinear/gen/f32-ibilinear-neonfma-c4.c
  src/f32-igemm/gen/f32-igemm-1x8-minmax-neonfma-dup-ld64.c
  src/f32-igemm/gen/f32-igemm-4x8-minmax-neonfma-dup-ld64.c
  src/f32-igemm/gen/f32-igemm-4x8-minmax-neonfma-dup-ld128.c
  src/f32-igemm/gen/f32-igemm-6x8-minmax-neonfma-dup-ld64.c
  src/f32-igemm/gen/f32-igemm-6x8-minmax-neonfma-dup-ld128.c
  src/f32-igemm/gen/f32-igemm-8x8s4-minmax-neonfma.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-neonfma-dup-ld64.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-neonfma-dup-ld64.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-neonfma-dup-ld64.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-neonfma-dup-ld64.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8s4-minmax-neonfma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-neonfma-dup-ld64.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8s4-minmax-neonfma.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-neonfma-dup-ld64.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8s4-minmax-neonfma.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u4.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u8-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u16-acc4.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u4.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u8-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u16-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u16-acc4.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-neonfma-x2.c
  src/f32-spmm/gen/f32-spmm-4x1-minmax-neonfma.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-neonfma-x2.c
  src/f32-spmm/gen/f32-spmm-8x1-minmax-neonfma.c
  src/f32-spmm/gen/f32-spmm-12x1-minmax-neonfma.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-neonfma-pipelined.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-neonfma-x2.c
  src/f32-spmm/gen/f32-spmm-16x1-minmax-neonfma.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-neonfma-x2.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-neonfma.c
  src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u4.c
  src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u8.c
  src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u12.c
  src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u4.c
  src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u12.c
  src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u16.c
  src/f32-vmulcaddc/gen/f32-vmulcaddc-c8-minmax-neonfma-2x.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u4.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u12.c
  src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u16.c)

SET(ALL_NEONFMA_MICROKERNEL_SRCS ${PROD_NEONFMA_MICROKERNEL_SRCS} + ${NON_PROD_NEONFMA_MICROKERNEL_SRCS})
