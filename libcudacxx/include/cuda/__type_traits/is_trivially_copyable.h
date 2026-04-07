//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA__TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_H
#define __CUDA__TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_H

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
#include <cuda/std/__type_traits/aggregate_members.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/remove_const.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp, typename = void>
constexpr bool __is_aggregate_trivially_copyable_v = false;

//! Users are allowed to specialize this variable template for their own types
template <typename _Tp>
constexpr bool is_trivially_copyable_v =
  ::cuda::std::is_trivially_copyable_v<::cuda::std::remove_const_t<_Tp>>
  || ::cuda::std::__is_extended_floating_point_v<::cuda::std::remove_const_t<_Tp>>
#if _CCCL_HAS_CTK()
  || ::cuda::is_extended_fp_vector_type_v<::cuda::std::remove_const_t<_Tp>>
#endif // _CCCL_HAS_CTK()
  || __is_aggregate_trivially_copyable_v<::cuda::std::remove_const_t<_Tp>>;

template <typename _Tp>
constexpr bool is_trivially_copyable_v<_Tp[]> = is_trivially_copyable_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
constexpr bool is_trivially_copyable_v<_Tp[_Size]> = is_trivially_copyable_v<_Tp>;

template <typename _Tp, ::cuda::std::size_t _Size>
constexpr bool is_trivially_copyable_v<::cuda::std::array<_Tp, _Size>> = is_trivially_copyable_v<_Tp>;

template <typename _T1, typename _T2>
constexpr bool is_trivially_copyable_v<::cuda::std::pair<_T1, _T2>> =
  is_trivially_copyable_v<_T1> && is_trivially_copyable_v<_T2>;

template <typename... _Ts>
constexpr bool is_trivially_copyable_v<::cuda::std::tuple<_Ts...>> = (is_trivially_copyable_v<_Ts> && ...);

// if all the previous conditions fail, check if the type is an aggregate and all its members are trivially copyable
template <typename _Tp>
using __is_trivially_copyable_callable = ::cuda::std::bool_constant<is_trivially_copyable_v<_Tp>>;

template <typename _Tp>
constexpr bool __is_aggregate_trivially_copyable_v<_Tp, ::cuda::std::enable_if_t<::cuda::std::is_aggregate_v<_Tp>>> =
  ::cuda::std::__aggregate_all_of<__is_trivially_copyable_callable, _Tp>::value;

// defined as alias so users cannot specialize it (they should specialize the variable template instead)
template <typename _Tp>
using is_trivially_copyable = ::cuda::std::bool_constant<is_trivially_copyable_v<_Tp>>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA__TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_H
