//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_CONCEPTS_H
#define _CUDAX___SIMD_CONCEPTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/experimental/__simd/declaration.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
template <typename _To, typename _From>
bool constexpr __is_value_preserving_broadcast_impl()
{
  // TODO
  return true;
}

template <typename _To, typename _From>
_CCCL_CONCEPT __is_value_preserving_convertible = __is_value_preserving_broadcast_impl<_From, _To>();

template <typename _To, typename _From>
_CCCL_CONCEPT __explicitly_convertible_to =
  _CCCL_REQUIRES_EXPR((_To, _From))(requires(static_cast<_To>(::cuda::std::declval<_From>())));

template <typename _Tp>
_CCCL_CONCEPT __constexpr_wrapper_like =
  ::cuda::std::convertible_to<_Tp, decltype(_Tp::value)>
  && ::cuda::std::equality_comparable_with<_Tp, decltype(_Tp::value)> && (_Tp() == _Tp::value)
  && (static_cast<decltype(_Tp::value)>(_Tp()) == _Tp::value);

template <typename _Tp, typename _ValueType>
_CCCL_CONCEPT __is_simd_ctor_explicit_from_value =
  ::cuda::std::convertible_to<_Tp, _ValueType>
  && ((!::cuda::std::is_arithmetic_v<_Tp> && !__constexpr_wrapper_like<_Tp>)
      || (::cuda::std::is_arithmetic_v<_Tp> && __is_value_preserving_convertible<_Tp, _ValueType>)
      || (__constexpr_wrapper_like<_Tp> && ::cuda::std::is_arithmetic_v<::cuda::std::remove_cvref_t<_Tp>>
          && __is_value_preserving_convertible<_Tp, _ValueType>);


template <typename _Tp>
inline constexpr int __integer_conversion_rank = 0;

template <>
inline constexpr int __integer_conversion_rank<signed char>        = 1;
template <>
inline constexpr int __integer_conversion_rank<unsigned char>      = 1;
template <>
inline constexpr int __integer_conversion_rank<char>               = 1;
template <>
inline constexpr int __integer_conversion_rank<short>              = 2;
template <>
inline constexpr int __integer_conversion_rank<unsigned short>     = 2;
template <>
inline constexpr int __integer_conversion_rank<int>                = 3;
template <>
inline constexpr int __integer_conversion_rank<unsigned int>       = 3;
template <>
inline constexpr int __integer_conversion_rank<long>               = 4;
template <>
inline constexpr int __integer_conversion_rank<unsigned long>      = 4;
template <>
inline constexpr int __integer_conversion_rank<long long>          = 5;
template <>
inline constexpr int __integer_conversion_rank<unsigned long long> = 5;
#if defined(_CCCL_HAS_INT128)
template <>
inline constexpr int __integer_conversion_rank<__int128_t> = 6;
template <>
inline constexpr int __integer_conversion_rank<__uint128_t> = 6;
#endif // defined(_CCCL_HAS_INT128)

template <typename _Tp>
inline constexpr int __fp_conversion_rank = 0;

#if _CCCL_HAS_NVFP16()
template <>
inline constexpr int __fp_conversion_rank<__half> = 1;
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
template <>
inline constexpr int __fp_conversion_rank<__nv_bfloat16> = 1;
#endif // _CCCL_HAS_NVBF16()
template <>
inline constexpr int __fp_conversion_rank<float> = 2;
template <>
inline constexpr int __fp_conversion_rank<double> = 3;
#if _CCCL_HAS_LONG_DOUBLE()
template <>
inline constexpr int __fp_conversion_rank<long double> = 4;
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
template <>
inline constexpr int __fp_conversion_rank<__float128> = 5;
#endif // _CCCL_HAS_FLOAT128()


template <typename _Tp, typename _Up>
_CCCL_CONCEPT __is_simd_ctor_explicit_from_vec =
  !__is_value_preserving_convertible<_Tp, _Up> ||
(::cuda::std::is_integral_v<_Tp> && ::cuda::std::is_integral_v<_Up> && 
__integer_conversion_rank<_Tp> > __integer_conversion_rank<_Up>) ||
(::cuda::is_floating_point_v<_Tp> && ::cuda::is_floating_point_v<_Up> && 
(__fp_conversion_rank<_Tp> > __fp_conversion_rank<_Up>));



template <typename _Tp>
_CCCL_CONCEPT __has_pre_increment = _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)((++__t));

template <typename _Tp>
_CCCL_CONCEPT __has_post_increment = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t++));

template <typename _Tp>
_CCCL_CONCEPT __has_pre_decrement = _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)((--__t));

template <typename _Tp>
_CCCL_CONCEPT __has_post_decrement = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t--));

template <typename _Tp>
_CCCL_CONCEPT __has_negate = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((!__t));

template <typename _Tp>
_CCCL_CONCEPT __has_bitwise_not = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((~__t));

template <typename _Tp>
_CCCL_CONCEPT __has_plus = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((+__t));

template <typename _Tp>
_CCCL_CONCEPT __has_unary_minus = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((-__t));

template <typename _Tp>
_CCCL_CONCEPT __has_minus = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t - __t));

template <typename _Tp>
_CCCL_CONCEPT __has_multiplies = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t * __t));

template <typename _Tp>
_CCCL_CONCEPT __has_divides = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t / __t));

template <typename _Tp>
_CCCL_CONCEPT __has_modulo = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t % __t));

template <typename _Tp>
_CCCL_CONCEPT __has_bitwise_and = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t & __t));

template <typename _Tp>
_CCCL_CONCEPT __has_bitwise_or = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t | __t));

template <typename _Tp>
_CCCL_CONCEPT __has_bitwise_xor = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t ^ __t));

template <typename _Tp>
_CCCL_CONCEPT __has_shift_left = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t << __t));

template <typename _Tp>
_CCCL_CONCEPT __has_shift_right = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t >> __t));

template <typename _Tp>
_CCCL_CONCEPT __has_shift_left_size = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t << __simd_size_type{}));

template <typename _Tp>
_CCCL_CONCEPT __has_shift_right_size = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t >> __simd_size_type{}));

template <typename _Tp>
_CCCL_CONCEPT __has_equal_to = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t == __t));

template <typename _Tp>
_CCCL_CONCEPT __has_not_equal_to = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t != __t));

template <typename _Tp>
_CCCL_CONCEPT __has_greater_equal = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t >= __t));

template <typename _Tp>
_CCCL_CONCEPT __has_less_equal = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t <= __t));

template <typename _Tp>
_CCCL_CONCEPT __has_greater = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t > __t));

template <typename _Tp>
_CCCL_CONCEPT __has_less = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t < __t));
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_CONCEPTS_H
