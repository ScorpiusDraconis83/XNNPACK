// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_bf16_f32_gemm_minmax_ukernel_2x32c2__asm_amd64_avx512bf16_broadcast

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

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

      # Copy k and flip bit.
      mov r11, rdx
      and r11, 0x2
      and rdx, 0xFFFFFFFFFFFFFFFD
      mov [rsp + 56], r11

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with the biases.
      vmovaps  zmm11, [r9 + 0]
      vmovaps  zmm13, [r9 + 64]
      vmovaps zmm12, zmm11
      vmovaps zmm14, zmm13
      add r9, 128

      # Are there at least 4 bytes?
      cmp rdx, 4
      js .Linner_loop_tail

.Linner_loop:
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      add r9, 128
      vbroadcastss zmm2, dword ptr [rcx + r11]
      vdpbf16ps  zmm11, zmm2, zmm7
      vdpbf16ps  zmm13, zmm2, zmm8
      vbroadcastss zmm3, dword ptr [rax + r11]
      vdpbf16ps  zmm12, zmm3, zmm7
      vdpbf16ps  zmm14, zmm3, zmm8

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

      # Store nc_register.
      mov [rsp + 64], rsi
      # Load odd k bit.
      mov rsi, [rsp + 56]
      # Check if channels are odd.
      test rsi, rsi
      mov rsi, [rsp + 64]
      jz .Linner_loop_end

.Linner_loop_tail:
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      add r9, 128
      vbroadcastss zmm2, dword ptr [rcx + r11]
      vpslld zmm2, zmm2, 16
      vpsrld zmm2, zmm2, 16
      vdpbf16ps  zmm11, zmm2, zmm7

      vpslld zmm2, zmm2, 16
      vpsrld zmm2, zmm2, 16
      vdpbf16ps  zmm13, zmm2, zmm8

      vbroadcastss zmm3, dword ptr [rax + r11]
      vpslld zmm3, zmm3, 16
      vpsrld zmm3, zmm3, 16
      vdpbf16ps  zmm12, zmm3, zmm7

      vpslld zmm3, zmm3, 16
      vpsrld zmm3, zmm3, 16
      vdpbf16ps  zmm14, zmm3, zmm8


.Linner_loop_end:
      # Min/max clamping.
      vminps  zmm11, zmm1, zmm11
      vminps  zmm13, zmm1, zmm13
      vminps  zmm12, zmm1, zmm12
      vminps  zmm14, zmm1, zmm14
      vmaxps  zmm11, zmm0, zmm11
      vmaxps  zmm13, zmm0, zmm13
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm14, zmm0, zmm14

      # Check whether full or partial store.
      cmp rsi, 32
      jl .Ltail

      vmovups  [r10], zmm11
      vmovups  [r10 + 64], zmm13
      vmovups  [r13], zmm12
      vmovups  [r13 + 64], zmm14
      add r10, 128
      add r13, 128

      sub rsi, 32
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      shr r11d, 16
      kmovw k2, r11d
      vmovups  zmmword ptr [r10]{k1}, zmm11
      vmovups  zmmword ptr [r10 + 64]{k2}, zmm13
      vmovups  zmmword ptr [r13]{k1}, zmm12
      vmovups  zmmword ptr [r13 + 64]{k2}, zmm14

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
END_FUNCTION xnn_bf16_f32_gemm_minmax_ukernel_2x32c2__asm_amd64_avx512bf16_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_bf16_f32_gemm_minmax_ukernel_2x32c2__asm_amd64_avx512bf16_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_bf16_f32_gemm_minmax_ukernel_2x32c2__asm_amd64_avx512bf16_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__