// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"
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

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_1x16c2__asm_amd64_avx512f_broadcast

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
      vbroadcastss zmm0, dword ptr [r13]
      vbroadcastss zmm1, dword ptr [r13 + 4]

      # Load c pointer.
      mov r10, [rsp + 72]
      # Load cm_stride.
      mov r11, [rsp + 80]

      # Align the stack pointer.
      mov r13, rsp
      sub rsp, 64
      and rsp, 0xFFFFFFFFFFFFFFC0
      # Store the old stack pointer containing the return address
      mov [rsp], r13

      # Allocate some space on the stack.
      sub rsp, 128

      # Copy k and flip bit.
      mov r11, rdx
      and r11, 0x4
      and rdx, 0xFFFFFFFFFFFFFFFB
      mov [rsp + 40], r11
      mov r11, 0x5555
      kmovw k3, r11d

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      vmovaps  zmm7, [r9 + 0]
      # Interleave with zeros.
      vpmovzxdq zmm11, ymm7
      vextracti64x4 ymm7, zmm7, 1
      vpmovzxdq zmm12, ymm7
      add r9, 64

      # Are there at least 8 bytes?
      cmp rdx, 8
      js .Linner_loop_tail

.Linner_loop:
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      add r9, 128
      vbroadcastsd zmm2, qword ptr [rcx + r11]
      vfmadd231ps  zmm11, zmm2, zmm7
      vfmadd231ps  zmm12, zmm2, zmm8

      add r11, 8
      cmp rdx, r11
      jne .Linner_loop

      # Store nc_register.
      mov [rsp + 48], rsi
      # Load odd k bit.
      mov rsi, [rsp + 40]
      # Check if channels are odd.
      test rsi, rsi
      mov rsi, [rsp + 48]
      jz .Linner_loop_end

.Linner_loop_tail:
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      add r9, 128
      vbroadcastsd zmm2, qword ptr [rcx + r11]
      vfmadd231ps  zmm11{k3}, zmm2, zmm7
      vfmadd231ps  zmm12{k3}, zmm2, zmm8

.Linner_loop_end:
      vpsrlq zmm7, zmm11, 32
      vaddps zmm11, zmm11, zmm7
      vpsrlq zmm7, zmm12, 32
      vaddps zmm12, zmm12, zmm7
      vmovups zmm7, zmmword ptr [rip + .PERMUTATION]
      vpermt2ps zmm11, zmm7, zmm12
      # Min/max clamping.
      vminps  zmm11, zmm1, zmm11
      vmaxps  zmm11, zmm0, zmm11

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail

      vmovups  [r10], zmm11
      add r10, 64

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  zmmword ptr [r10]{k1}, zmm11

.Lreturn:
      add rsp, 128
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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_1x16c2__asm_amd64_avx512f_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_1x16c2__asm_amd64_avx512f_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_1x16c2__asm_amd64_avx512f_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__