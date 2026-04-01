//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA__TYPE_TRAITS_IS_BITWISE_COMPARABLE_H
#define __CUDA__TYPE_TRAITS_IS_BITWISE_COMPARABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_vector_type.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__type_traits/has_unique_object_representation.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! Users are allowed to specialize this variable template for their own types
template <typename _Tp>
constexpr bool is_bitwise_comparable_v =
  ::cuda::std::has_unique_object_representations_v<::cuda::std::remove_cv_t<_Tp>>
  && !::cuda::std::__is_extended_floating_point_v<::cuda::std::remove_cv_t<_Tp>>
#if _CCCL_HAS_CTK()
  && !::cuda::is_extended_fp_vector_type_v<::cuda::std::remove_cv_t<_Tp>>
#endif // _CCCL_HAS_CTK()
  ;

template <typename _Tp>
constexpr bool is_bitwise_comparable_v<_Tp[]> = is_bitwise_comparable_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
constexpr bool is_bitwise_comparable_v<_Tp[_Size]> = is_bitwise_comparable_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
constexpr bool is_bitwise_comparable_v<::cuda::std::array<_Tp, _Size>> = is_bitwise_comparable_v<_Tp>;

template <typename _T1, typename _T2>
constexpr bool is_bitwise_comparable_v<::cuda::std::pair<_T1, _T2>> =
  (sizeof(::cuda::std::pair<_T1, _T2>) == sizeof(_T1) + sizeof(_T2))
  && is_bitwise_comparable_v<_T1> && is_bitwise_comparable_v<_T2>;

template <typename... _Ts>
constexpr bool is_bitwise_comparable_v<::cuda::std::tuple<_Ts...>> =
  (sizeof...(_Ts) == 0 || sizeof(::cuda::std::tuple<_Ts...>) == (sizeof(_Ts) + ... + 0))
  && (is_bitwise_comparable_v<_Ts> && ...);

// defined as alias so users cannot specialize it (they should specialize the variable template instead)
template <typename _Tp>
using is_bitwise_comparable = ::cuda::std::bool_constant<is_bitwise_comparable_v<_Tp>>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA__TYPE_TRAITS_IS_BITWISE_COMPARABLE_H
