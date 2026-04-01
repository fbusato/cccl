//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA__TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_RELAXED_H
#define __CUDA__TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_RELAXED_H

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
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
constexpr bool __is_trivially_copyable_relaxed_impl_v =
  ::cuda::std::__is_extended_floating_point_v<_Up> || ::cuda::is_extended_fp_vector_type_v<_Up>
  || ::cuda::std::is_trivially_copyable_v<_Up>;

template <typename _Tp>
struct is_trivially_copyable_relaxed : ::cuda::std::bool_constant<::cuda::__is_trivially_copyable_relaxed_impl_v<_Tp>>
{};

template <typename _Tp>
struct is_trivially_copyable_relaxed<_Tp[]> : is_trivially_copyable_relaxed<_Tp>
{};

template <typename _Tp, ::cuda::std::size_t _Size>
struct is_trivially_copyable_relaxed<_Tp[_Size]> : is_trivially_copyable_relaxed<_Tp>
{};

template <typename _Tp, ::cuda::std::size_t _Size>
struct is_trivially_copyable_relaxed<::cuda::std::array<_Tp, _Size>> : is_trivially_copyable_relaxed<_Tp>
{};

// cuda::std::pair
template <typename _T1, typename _T2>
struct is_trivially_copyable_relaxed<::cuda::std::pair<_T1, _T2>>
    : ::cuda::std::bool_constant<__is_trivially_copyable_relaxed_impl_v<_T1> && __is_trivially_copyable_relaxed_impl_v<_T2>>
{};

// cuda::std::tuple
template <typename... _Ts>
struct is_trivially_copyable_relaxed<::cuda::std::tuple<_Ts...>>
    : ::cuda::std::bool_constant<(__is_trivially_copyable_relaxed_impl_v<_Ts> && ...)>
{};

template <typename _Tp>
constexpr bool is_trivially_copyable_relaxed_v = is_trivially_copyable_relaxed<_Tp>::value;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA__TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_RELAXED_H
