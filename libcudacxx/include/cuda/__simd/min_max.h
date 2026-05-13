//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SIMD_MIN_MAX_H
#define _CUDA___SIMD_MIN_MAX_H

#include <cuda/std/detail/__config>

#include <cuda/std/__cccl/compiler.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// TODO(fbusato): remove this path once the compiler applies the optimization automatically

#include <cuda/std/__cmath/min_max.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/exposition.h>
#include <cuda/std/__simd/math.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/array>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

#if (_CCCL_HAS_NVFP16() || _CCCL_HAS_NVBF16()) && _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(typename _FpX2, typename _Tp, typename _Abi, typename _Vec = ::cuda::std::simd::basic_vec<_Tp, _Abi>)
[[nodiscard]]
_CCCL_DEVICE_API constexpr _Vec
__fmin_x2(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input1,
          const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input2,
          const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input3) noexcept
{
  constexpr auto __size = ::cuda::std::simd::__simd_size_v<_Tp, _Abi>;
  ::cuda::std::array<_Tp, __size> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (auto __i = 0; __i < __size; __i += 2)
  {
    const auto __input1_x2 = _FpX2{__input1[__i], __input1[__i + 1]};
    const auto __input2_x2 = _FpX2{__input2[__i], __input2[__i + 1]};
    const auto __input3_x2 = _FpX2{__input3[__i], __input3[__i + 1]};
    const auto __tmp_x2    = ::__hmin2(__input1_x2, __input2_x2);
    const auto __result_x2 = ::__hmin2(__tmp_x2, __input3_x2);
    __result[__i]          = __result_x2.x;
    __result[__i + 1]      = __result_x2.y;
  }
  if constexpr (__size % 2 == 1)
  {
    const auto __tmp     = ::cuda::std::fmin(__input1[__size - 1], __input2[__size - 1]);
    __result[__size - 1] = ::cuda::std::fmin(__tmp, __input3[__size - 1]);
  }
  return _Vec{__result};
}

_CCCL_TEMPLATE(typename _FpX2, typename _Tp, typename _Abi, typename _Vec = ::cuda::std::simd::basic_vec<_Tp, _Abi>)
[[nodiscard]]
_CCCL_DEVICE_API constexpr _Vec
__fmax_x2(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input1,
          const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input2,
          const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input3) noexcept
{
  constexpr auto __size = ::cuda::std::simd::__simd_size_v<_Tp, _Abi>;
  ::cuda::std::array<_Tp, __size> __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (auto __i = 0; __i < __size; __i += 2)
  {
    const auto __input1_x2 = _FpX2{__input1[__i], __input1[__i + 1]};
    const auto __input2_x2 = _FpX2{__input2[__i], __input2[__i + 1]};
    const auto __input3_x2 = _FpX2{__input3[__i], __input3[__i + 1]};
    const auto __tmp_x2    = ::__hmax2(__input1_x2, __input2_x2);
    const auto __result_x2 = ::__hmax2(__tmp_x2, __input3_x2);
    __result[__i]          = __result_x2.x;
    __result[__i + 1]      = __result_x2.y;
  }
  if constexpr (__size % 2 == 1)
  {
    const auto __tmp     = ::cuda::std::fmax(__input1[__size - 1], __input2[__size - 1]);
    __result[__size - 1] = ::cuda::std::fmax(__tmp, __input3[__size - 1]);
  }
  return _Vec{__result};
}

#endif // (_CCCL_HAS_NVFP16() || _CCCL_HAS_NVBF16()) && _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vec = ::cuda::std::simd::basic_vec<_Tp, _Abi>)
[[nodiscard]]
_CCCL_API constexpr _Vec fmin(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input1,
                              const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input2,
                              const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input3) noexcept
{
#if _CCCL_HAS_NVFP16() || _CCCL_HAS_NVBF16()
  constexpr auto __size = ::cuda::std::simd::__simd_size_v<_Tp, _Abi>;
  if constexpr (__size >= 2 && (::cuda::std::is_same_v<_Tp, ::__half> || ::cuda::std::is_same_v<_Tp, ::__nv_bfloat16>) )
  {
    _CCCL_IF_NOT_CONSTEVAL
    {
#  if _CCCL_HAS_NVFP16()
      if constexpr (::cuda::std::is_same_v<_Tp, ::__half>)
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return ::cuda::simd::__fmin_x2<::__half2>(__input1, __input2, __input3);))
      }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
      else if constexpr (::cuda::std::is_same_v<_Tp, ::__nv_bfloat16>)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_80,
                     (return ::cuda::simd::__fmin_x2<::__nv_bfloat162>(__input1, __input2, __input3);))
      }
#  endif // _CCCL_HAS_NVBF16()
    }
  }
#endif // _CCCL_HAS_NVFP16() || _CCCL_HAS_NVBF16()
  return ::cuda::std::simd::fmin(::cuda::std::simd::fmin(__input1, __input2), __input3);
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vec = ::cuda::std::simd::basic_vec<_Tp, _Abi>)
[[nodiscard]]
_CCCL_API constexpr _Vec fmax(const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input1,
                              const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input2,
                              const ::cuda::std::simd::basic_vec<_Tp, _Abi>& __input3) noexcept
{
#if _CCCL_HAS_NVFP16() || _CCCL_HAS_NVBF16()
  constexpr auto __size = ::cuda::std::simd::__simd_size_v<_Tp, _Abi>;
  if constexpr (__size >= 2 && (::cuda::std::is_same_v<_Tp, ::__half> || ::cuda::std::is_same_v<_Tp, ::__nv_bfloat16>) )
  {
    _CCCL_IF_NOT_CONSTEVAL
    {
#  if _CCCL_HAS_NVFP16()
      if constexpr (::cuda::std::is_same_v<_Tp, ::__half>)
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return ::cuda::simd::__fmax_x2<::__half2>(__input1, __input2, __input3);))
      }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
      else if constexpr (::cuda::std::is_same_v<_Tp, ::__nv_bfloat16>)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_80,
                     (return ::cuda::simd::__fmax_x2<::__nv_bfloat162>(__input1, __input2, __input3);))
      }
#  endif // _CCCL_HAS_NVBF16()
    }
  }
#endif // _CCCL_HAS_NVFP16() || _CCCL_HAS_NVBF16()
  return ::cuda::std::simd::fmax(::cuda::std::simd::fmax(__input1, __input2), __input3);
}

_CCCL_END_NAMESPACE_CUDA_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___SIMD_MIN_MAX_H
