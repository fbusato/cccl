//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_INTRINSIC_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_INTRINSIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8) || __cccl_ptx_isa >= 860ULL
#  define _CCCL_HAS_SIMD_F32X2() 1
#else
#  define _CCCL_HAS_SIMD_F32X2() 0
#endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8) || __cccl_ptx_isa >= 860ULL

#if _CCCL_HAS_SIMD_F32X2()

#  include <cuda/std/__cstddef/types.h>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_API inline void __add_f32x2(
  const float __lhs1,
  const float __lhs2,
  const float __rhs1,
  const float __rhs2,
  float& __result1,
  float& __result2) noexcept
{
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
  const auto __result = ::__fadd2_rn(::float2{__lhs1, __lhs2}, ::float2{__rhs1, __rhs2});
  __result1           = __result.x;
  __result2           = __result.y;
#  elif __cccl_ptx_isa >= 860ULL
  asm("add.f32x2 {%0, %1}, {%2, %3}, {%4, %5};"
      : "=f"(__result1), "=f"(__result2)
      : "f"(__lhs1), "f"(__lhs2), "f"(__rhs1), "f"(__rhs2));
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
}

template <size_t _Np>
_CCCL_API constexpr void
__plus_f32x2(const float (&__lhs)[_Np], const float (&__rhs)[_Np], float (&__result)[_Np]) noexcept
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (size_t __i = 0; __i < (_Np / 2) * 2; __i += 2)
  {
    ::cuda::std::simd::__add_f32x2(
      __lhs[__i], __lhs[__i + 1], __rhs[__i], __rhs[__i + 1], __result[__i], __result[__i + 1]);
  }
  if (_Np % 2 != 0)
  {
    __result[_Np - 1] = __lhs[_Np - 1] + __rhs[_Np - 1];
  }
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_SIMD_F32X2()
#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_INTRINSIC_H
