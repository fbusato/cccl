//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_FP32X2_INTRINSICS_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_FP32X2_INTRINSICS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8) || (__cccl_ptx_isa >= 860ULL)
#  define _CCCL_HAS_SIMD_F32X2() 1
#else
#  define _CCCL_HAS_SIMD_F32X2() 0
#endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8) || (__cccl_ptx_isa >= 860ULL)

#if _CCCL_HAS_SIMD_F32X2()

#  include <cuda/std/__simd/specializations/fixed_size_storage.h>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_DEVICE_API inline void __add_f32x2(
  const float __lhs1,
  const float __lhs2,
  const float __rhs1,
  const float __rhs2,
  float& __result1,
  float& __result2) noexcept
{
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
  // clang-format off
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100,
               (const auto __result = ::__fadd2_rn(::float2{__lhs1, __lhs2}, ::float2{__rhs1, __rhs2});
                __result1           = __result.x;
                __result2           = __result.y;))
  // clang-format on
#  elif (__cccl_ptx_isa >= 860ULL) // PTX ISA 8.6
  asm("{.reg .b64 __lhs, __rhs, __result;"
      "mov.b64 __lhs, {%2, %3};"
      "mov.b64 __rhs, {%4, %5};"
      "add.f32x2 __result, __lhs, __rhs;"
      "mov.b64 {%0, %1}, __result;}"
      : "=f"(__result1), "=f"(__result2)
      : "f"(__lhs1), "f"(__lhs2), "f"(__rhs1), "f"(__rhs2));
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
}

_CCCL_DEVICE_API inline void __mul_f32x2(
  const float __lhs1,
  const float __lhs2,
  const float __rhs1,
  const float __rhs2,
  float& __result1,
  float& __result2) noexcept
{
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
  // clang-format off
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100,
               (const auto __result = ::__fmul2_rn(::float2{__lhs1, __lhs2}, ::float2{__rhs1, __rhs2});
                __result1           = __result.x;
                __result2           = __result.y;))
  // clang-format on
#  elif (__cccl_ptx_isa >= 860ULL) // PTX ISA 8.6
  asm("{.reg .b64 __lhs, __rhs, __result;"
      "mov.b64 __lhs, {%2, %3};"
      "mov.b64 __rhs, {%4, %5};"
      "mul.f32x2 __result, __lhs, __rhs;"
      "mov.b64 {%0, %1}, __result;}"
      : "=f"(__result1), "=f"(__result2)
      : "f"(__lhs1), "f"(__lhs2), "f"(__rhs1), "f"(__rhs2));
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
}

_CCCL_DEVICE_API inline void __sub_f32x2(
  const float __lhs1,
  const float __lhs2,
  const float __rhs1,
  const float __rhs2,
  float& __result1,
  float& __result2) noexcept
{
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
  // clang-format off
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100,
               (const auto __result = ::__fadd2_rn(::float2{__lhs1, __lhs2}, ::float2{-__rhs1, -__rhs2});
                __result1           = __result.x;
                __result2           = __result.y;))
  // clang-format on
#  elif (__cccl_ptx_isa >= 860ULL) // PTX ISA 8.6
  // clang-format off
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100,
               (asm("{.reg .b64 __lhs, __rhs, __result;"
                    "mov.b64 __lhs, {%2, %3};"
                    "mov.b64 __rhs, {%4, %5};"
                    "sub.f32x2 __result, __lhs, __rhs;"
                    "mov.b64 {%0, %1}, __result;}"
                    : "=f"(__result1), "=f"(__result2)
                    : "f"(__lhs1), "f"(__lhs2), "f"(__rhs1), "f"(__rhs2));))
  // clang-format on
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
}

_CCCL_DEVICE_API inline void __fma_f32x2(
  const float __lhs1,
  const float __lhs2,
  const float __rhs1,
  const float __rhs2,
  const float __add1,
  const float __add2,
  float& __result1,
  float& __result2) noexcept
{
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
  // clang-format off
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100,
               (const auto __result =
                  ::__ffma2_rn(::float2{__lhs1, __lhs2}, ::float2{__rhs1, __rhs2}, ::float2{__add1, __add2});
                __result1 = __result.x;
                __result2 = __result.y;))
  // clang-format on
#  elif (__cccl_ptx_isa >= 860ULL) // PTX ISA 8.6
  asm("{.reg .b64 __lhs, __rhs, __add, __result;"
      "mov.b64 __lhs, {%2, %3};"
      "mov.b64 __rhs, {%4, %5};"
      "mov.b64 __add, {%6, %7};"
      "fma.rn.f32x2 __result, __lhs, __rhs, __add;"
      "mov.b64 {%0, %1}, __result;}"
      : "=f"(__result1), "=f"(__result2)
      : "f"(__lhs1), "f"(__lhs2), "f"(__rhs1), "f"(__rhs2), "f"(__add1), "f"(__add2));
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 8)
}

template <__simd_size_type _Np>
using __simd_storage_f32 = __simd_storage<float, __fixed_size<_Np>>;

template <__simd_size_type _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __simd_storage_f32<_Np>
__plus_f32x2(const __simd_storage_f32<_Np>& __lhs, const __simd_storage_f32<_Np>& __rhs) noexcept
{
  __simd_storage_f32<_Np> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < (_Np / 2) * 2; __i += 2)
  {
    ::cuda::std::simd::__add_f32x2(
      __lhs.__data[__i],
      __lhs.__data[__i + 1],
      __rhs.__data[__i],
      __rhs.__data[__i + 1],
      __result.__data[__i],
      __result.__data[__i + 1]);
  }
  if (_Np % 2 != 0)
  {
    __result.__data[_Np - 1] = __lhs.__data[_Np - 1] + __rhs.__data[_Np - 1];
  }
  return __result;
}

template <__simd_size_type _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __simd_storage_f32<_Np>
__minus_f32x2(const __simd_storage_f32<_Np>& __lhs, const __simd_storage_f32<_Np>& __rhs) noexcept
{
  __simd_storage_f32<_Np> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < (_Np / 2) * 2; __i += 2)
  {
    ::cuda::std::simd::__sub_f32x2(
      __lhs.__data[__i],
      __lhs.__data[__i + 1],
      __rhs.__data[__i],
      __rhs.__data[__i + 1],
      __result.__data[__i],
      __result.__data[__i + 1]);
  }
  if (_Np % 2 != 0)
  {
    __result.__data[_Np - 1] = __lhs.__data[_Np - 1] - __rhs.__data[_Np - 1];
  }
  return __result;
}

template <__simd_size_type _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __simd_storage_f32<_Np>
__multiplies_f32x2(const __simd_storage_f32<_Np>& __lhs, const __simd_storage_f32<_Np>& __rhs) noexcept
{
  __simd_storage_f32<_Np> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < (_Np / 2) * 2; __i += 2)
  {
    ::cuda::std::simd::__mul_f32x2(
      __lhs.__data[__i],
      __lhs.__data[__i + 1],
      __rhs.__data[__i],
      __rhs.__data[__i + 1],
      __result.__data[__i],
      __result.__data[__i + 1]);
  }
  if (_Np % 2 != 0)
  {
    __result.__data[_Np - 1] = __lhs.__data[_Np - 1] * __rhs.__data[_Np - 1];
  }
  return __result;
}

template <__simd_size_type _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __simd_storage_f32<_Np>
__fma_f32x2(const __simd_storage_f32<_Np>& __lhs,
            const __simd_storage_f32<_Np>& __rhs,
            const __simd_storage_f32<_Np>& __add) noexcept
{
  __simd_storage_f32<_Np> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < (_Np / 2) * 2; __i += 2)
  {
    ::cuda::std::simd::__fma_f32x2(
      __lhs.__data[__i],
      __lhs.__data[__i + 1],
      __rhs.__data[__i],
      __rhs.__data[__i + 1],
      __add.__data[__i],
      __add.__data[__i + 1],
      __result.__data[__i],
      __result.__data[__i + 1]);
  }
  if (_Np % 2 != 0)
  {
    __result.__data[_Np - 1] = __lhs.__data[_Np - 1] * __rhs.__data[_Np - 1] + __add.__data[_Np - 1];
  }
  return __result;
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_SIMD_F32X2()
#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_FP32X2_INTRINSICS_H
