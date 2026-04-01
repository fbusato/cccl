//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_UTILITY_H
#define _CUDA_STD___SIMD_UTILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/is_aligned.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__simd/flag.h>
#include <cuda/std/__simd/type_traits.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::std::simd
{
template <typename _Tp>
constexpr bool __is_abi_tag_v = false;

template <__simd_size_type _Np>
constexpr bool __is_abi_tag_v<__fixed_size_simple<_Np>> = true;

template <typename _Tp, typename _Generator, __simd_size_type _Idx, typename = void>
constexpr bool __is_well_formed = false;

template <typename _Tp, typename _Generator, __simd_size_type _Idx>
constexpr bool __is_well_formed<_Tp,
                                _Generator,
                                _Idx,
                                ::cuda::std::void_t<decltype(::cuda::std::declval<_Generator>()(
                                  ::cuda::std::integral_constant<__simd_size_type, _Idx>()))>> =
  ::cuda::std::is_convertible_v<
    decltype(::cuda::std::declval<_Generator>()(::cuda::std::integral_constant<__simd_size_type, _Idx>())),
    _Tp>;

template <typename _Tp, typename _Generator, __simd_size_type... _Indices>
[[nodiscard]]
_CCCL_API constexpr bool __can_generate(::cuda::std::integer_sequence<__simd_size_type, _Indices...>) noexcept
{
  return (true && ... && __is_well_formed<_Tp, _Generator, _Indices>);
}

template <typename _Tp, typename _Generator, __simd_size_type _Size>
constexpr bool __can_generate_v =
  ::cuda::std::simd::__can_generate<_Tp, _Generator>(::cuda::std::make_integer_sequence<__simd_size_type, _Size>());

// Proxy for ranges::size(r) is a constant expression
template <typename _Range>
_CCCL_CONCEPT __has_static_size =
  _CCCL_REQUIRES_EXPR((_Range))((__simd_size_type{::cuda::std::tuple_size_v<::cuda::std::remove_cvref_t<_Range>>}));

template <typename _Range>
constexpr __simd_size_type __static_range_size_v =
  __simd_size_type{::cuda::std::tuple_size_v<::cuda::std::remove_cvref_t<_Range>>};

// [simd.flags] alignment assertion for load/store pointers
template <typename _Vec, typename _Up, typename... _Flags>
_CCCL_API constexpr void __assert_load_store_alignment([[maybe_unused]] const _Up* __data) noexcept
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    if constexpr (__has_aligned_flag_v<_Flags...>)
    {
      _CCCL_ASSERT(::cuda::is_aligned(__data, alignment_v<_Vec, _Up>),
                   "flag_aligned requires data to be aligned to alignment_v<V, range_value_t<R>>");
    }
    else if constexpr (__has_overaligned_flag_v<_Flags...>)
    {
      _CCCL_ASSERT(::cuda::is_aligned(__data, __overaligned_alignment_v<_Flags...>),
                   "flag_overaligned<N> requires data to be aligned to N");
    }
  }
}
} // namespace cuda::std::simd

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_UTILITY_H
