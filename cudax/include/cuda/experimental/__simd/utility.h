//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_UTILITY_H
#define _CUDAX___SIMD_UTILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__simd/exposition.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::simd
{


template <class _From, class _To, class = void>
inline constexpr bool __is_non_narrowing_convertible_v = false;

template <typename _From, typename _To>
inline constexpr bool
  __is_non_narrowing_convertible_v<_From, _To, ::cuda::std::void_t<decltype(_To{::cuda::std::declval<_From>()})>> =
    true;



template <::cuda::std::size_t _Bytes>
constexpr bool __has_integer_from_v =
  (_Bytes == 1 || _Bytes == 2 || _Bytes == 4 || _Bytes == 8
#if _CCCL_HAS_INT128()
   || _Bytes == 16
#endif // _CCCL_HAS_INT128()
  );

template <typename _Tp>
constexpr bool __is_abi_tag_v = false;

template <int _Np>
constexpr bool __is_abi_tag_v<simd_abi::fixed_size_simple<_Np>> = true;

template <typename _Tp, typename _Generator, ::cuda::std::size_t _Idx, typename = void>
constexpr bool __is_well_formed = false;

template <typename _Tp, typename _Generator, ::cuda::std::size_t _Idx>
constexpr bool __is_well_formed<_Tp,
                                _Generator,
                                _Idx,
                                ::cuda::std::void_t<decltype(::cuda::std::declval<_Generator>()(
                                  ::cuda::std::integral_constant<__simd_size_type, _Idx>()))>> = ::cuda::std::
  is_same_v<_Tp, decltype(::cuda::std::declval<_Generator>()(::cuda::std::integral_constant<__simd_size_type, _Idx>()))>;

template <typename _Tp, typename _Generator, ::cuda::std::size_t... _Indices>
_CCCL_API constexpr bool __can_generate(::cuda::std::integer_sequence<__simd_size_type, _Indices...>)
{
  return (true && ... && __is_well_formed<_Tp, _Generator, _Indices>);
}

template <typename _Tp, typename _Generator, ::cuda::std::size_t _Size>
constexpr bool __can_generate_v = ::cuda::experimental::simd::__can_generate<_Tp, _Generator>(
  ::cuda::std::make_integer_sequence<__simd_size_type, _Size>());
} // namespace cuda::experimental::simd

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_UTILITY_H
