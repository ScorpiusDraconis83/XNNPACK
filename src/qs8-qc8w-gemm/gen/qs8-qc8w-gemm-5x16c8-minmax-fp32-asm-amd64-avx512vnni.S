// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

      .p2align 6, 0x0
.PERMUTATION:
      .long   0
      .long   2
      .long   4
      .long   6
      .long   8
      .long   10
      .long   12
      .long   14
      .long   16
      .long   18
      .long   20
      .long   22
      .long   24
      .long   26
      .long   28
      .long   30
.SIGN_MASK:
      .quad   -9187201950435737472  # 0x8080808080808080

BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni

      .intel_syntax noprefix
      # Free up GP registers.
      # Save register arguments for tail call to msan annotation helper.
      push rdi
      push rsi
      push rbx
      push rbp
      push r15
      push r14
      push r13
      push r12

      # load params to free up GP registers
      mov r13, [rsp + 96] # params

      movsx         eax, word ptr [r13]
      vpbroadcastd zmm31, eax

      vpbroadcastb xmm0, byte ptr [r13 + 2]

      movsx         eax, word ptr [r13 + 4]
      vpbroadcastd  zmm1, eax
      vpsubd        zmm1, zmm1, zmm31
      vcvtdq2ps     zmm1, zmm1


      # Load c pointer.
      mov r10, [rsp + 72]
      # Load cm_stride.
      mov r11, [rsp + 80]

      add rdx, 7
      and rdx, -8

      # Align the stack pointer.
      mov r13, rsp
      sub rsp, 64
      and rsp, 0xFFFFFFFFFFFFFFC0
      # Store the old stack pointer containing the return address
      mov [rsp], r13

      # Allocate some space on the stack.
      sub rsp, 448

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

      # Clamp a & c pointers if mr <= 2
      mov r15, rax
      add r15, r8
      mov rbx, r13
      add rbx, r11
      cmp rdi, 2
      cmovle r15, rax
      cmovle rbx, r13

      # Clamp a & c pointers if mr <= 3
      mov r14, r15
      add r14, r8
      mov rbp, rbx
      add rbp, r11
      cmp rdi, 3
      cmovle r14, r15
      cmovle rbp, rbx

      # Clamp a & c pointers if mr <= 4
      mov r12, r14
      add r12, r8
      mov r8, rbp
      add r8, r11
      cmp rdi, 4
      cmovle r12, r14
      cmovle r8, rbp

      # Load 0x80 for xoring the weights
      vbroadcastsd  zmm13, qword ptr [rip + .SIGN_MASK]


.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with bias
      vmovaps zmm5, [r9 + 0]
      vmovaps zmm12, [r9 + 0]
      vmovaps zmm14, [r9 + 0]
      vmovaps zmm15, [r9 + 0]
      vmovaps zmm16, [r9 + 0]
      add r9, 64
      # Interleave with zeros.
      vextracti64x4 ymm17, zmm5, 1
      vpmovzxdq zmm17, ymm17
      vpmovzxdq zmm5, ymm5
      vextracti64x4 ymm18, zmm12, 1
      vpmovzxdq zmm18, ymm18
      vpmovzxdq zmm12, ymm12
      vextracti64x4 ymm19, zmm14, 1
      vpmovzxdq zmm19, ymm19
      vpmovzxdq zmm14, ymm14
      vextracti64x4 ymm20, zmm15, 1
      vpmovzxdq zmm20, ymm20
      vpmovzxdq zmm15, ymm15
      vextracti64x4 ymm21, zmm16, 1
      vpmovzxdq zmm21, ymm21
      vpmovzxdq zmm16, ymm16

.Linner_loop:
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      add r9, 128
      vpxorq zmm2, zmm13, qword ptr [rcx + r11]{1to8}
      vpdpbusd  zmm5, zmm2, zmm6
      vpdpbusd  zmm17, zmm2, zmm7
      vpxorq zmm2, zmm13, qword ptr [rax + r11]{1to8}
      vpdpbusd  zmm12, zmm2, zmm6
      vpdpbusd  zmm18, zmm2, zmm7
      vpxorq zmm2, zmm13, qword ptr [r15 + r11]{1to8}
      vpdpbusd  zmm14, zmm2, zmm6
      vpdpbusd  zmm19, zmm2, zmm7
      vpxorq zmm2, zmm13, qword ptr [r14 + r11]{1to8}
      vpdpbusd  zmm15, zmm2, zmm6
      vpdpbusd  zmm20, zmm2, zmm7
      vpxorq zmm2, zmm13, qword ptr [r12 + r11]{1to8}
      vpdpbusd  zmm16, zmm2, zmm6
      vpdpbusd  zmm21, zmm2, zmm7

      add r11, 8
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:
      vpsrlq zmm6, zmm5, 32
      vpaddd zmm5, zmm5, zmm6
      vpsrlq zmm6, zmm12, 32
      vpaddd zmm12, zmm12, zmm6
      vpsrlq zmm6, zmm14, 32
      vpaddd zmm14, zmm14, zmm6
      vpsrlq zmm6, zmm15, 32
      vpaddd zmm15, zmm15, zmm6
      vpsrlq zmm6, zmm16, 32
      vpaddd zmm16, zmm16, zmm6
      vpsrlq zmm6, zmm17, 32
      vpaddd zmm17, zmm17, zmm6
      vpsrlq zmm6, zmm18, 32
      vpaddd zmm18, zmm18, zmm6
      vpsrlq zmm6, zmm19, 32
      vpaddd zmm19, zmm19, zmm6
      vpsrlq zmm6, zmm20, 32
      vpaddd zmm20, zmm20, zmm6
      vpsrlq zmm6, zmm21, 32
      vpaddd zmm21, zmm21, zmm6
      vmovaps zmm6, zmmword ptr [rip + .PERMUTATION]
      vpermt2ps zmm5, zmm6, zmm17
      vpermt2ps zmm12, zmm6, zmm18
      vpermt2ps zmm14, zmm6, zmm19
      vpermt2ps zmm15, zmm6, zmm20
      vpermt2ps zmm16, zmm6, zmm21

      # Convert from int32 to float.
      vcvtdq2ps zmm5, zmm5
      vcvtdq2ps zmm12, zmm12
      vcvtdq2ps zmm14, zmm14
      vcvtdq2ps zmm15, zmm15
      vcvtdq2ps zmm16, zmm16
      vmovaps zmm10, [r9 + 0]
      add r9, 64
      vmulps zmm5, zmm5, zmm10
      vmulps zmm12, zmm12, zmm10
      vmulps zmm14, zmm14, zmm10
      vmulps zmm15, zmm15, zmm10
      vmulps zmm16, zmm16, zmm10
      vminps zmm5, zmm5, zmm1
      vminps zmm12, zmm12, zmm1
      vminps zmm14, zmm14, zmm1
      vminps zmm15, zmm15, zmm1
      vminps zmm16, zmm16, zmm1
      vcvtps2dq zmm5, zmm5
      vcvtps2dq zmm12, zmm12
      vcvtps2dq zmm14, zmm14
      vcvtps2dq zmm15, zmm15
      vcvtps2dq zmm16, zmm16
      vpaddd zmm5, zmm5, zmm31
      vpaddd zmm12, zmm12, zmm31
      vpaddd zmm14, zmm14, zmm31
      vpaddd zmm15, zmm15, zmm31
      vpaddd zmm16, zmm16, zmm31
      vpmovsdb xmm5, zmm5
      vpmovsdb xmm12, zmm12
      vpmovsdb xmm14, zmm14
      vpmovsdb xmm15, zmm15
      vpmovsdb xmm16, zmm16
      vpmaxsb xmm5, xmm5, xmm0
      vpmaxsb xmm12, xmm12, xmm0
      vpmaxsb xmm14, xmm14, xmm0
      vpmaxsb xmm15, xmm15, xmm0
      vpmaxsb xmm16, xmm16, xmm0

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail

      vmovups  [r10], xmm5
      vmovups  [r13], xmm12
      vmovups  [rbx], xmm14
      vmovups  [rbp], xmm15
      vmovups  [r8], xmm16
      add r10, 16
      add r13, 16
      add rbx, 16
      add rbp, 16
      add r8, 16

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovdqu8  xmmword ptr [r10]{k1}, xmm5
      vmovdqu8  xmmword ptr [r13]{k1}, xmm12
      vmovdqu8  xmmword ptr [rbx]{k1}, xmm14
      vmovdqu8  xmmword ptr [rbp]{k1}, xmm15
      vmovdqu8  xmmword ptr [r8]{k1}, xmm16

.Lreturn:
      add rsp, 448
      mov r13, [rsp]
      mov rsp, r13
      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      pop rsi
      pop rdi
      #if XNN_HAS_FEATURE(memory_sanitizer)
      jmp xnn_gemm_ukernel_msan_sizeof_c_4
      #else
      ret
      #endif
END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__asm_amd64_avx512vnni.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__