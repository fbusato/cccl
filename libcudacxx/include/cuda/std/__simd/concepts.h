//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_CONCEPTS_H
#define _CUDA_STD___SIMD_CONCEPTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/overflow_cast.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__floating_point/conversion_rank_order.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::std::simd
{
// [simd.expos], explicitly-convertible-to concept

template <typename _To, typename _From>
_CCCL_CONCEPT __explicitly_convertible_to =
  _CCCL_REQUIRES_EXPR((_To, _From))((static_cast<_To>(::cuda::std::declval<_From>())));

// [simd.expos], constexpr-wrapper-like concept

template <typename _Tp>
_CCCL_CONCEPT __constexpr_wrapper_like =
  ::cuda::std::convertible_to<_Tp, decltype(_Tp::value)>
  && ::cuda::std::equality_comparable_with<_Tp, decltype(_Tp::value)>
  && ::cuda::std::bool_constant<(_Tp() == _Tp::value)>::value
  && ::cuda::std::bool_constant<(static_cast<decltype(_Tp::value)>(_Tp()) == _Tp::value)>::value;

// (c++draft)The conversion from an arithmetic type U to a vectorizable type T is value-preserving if all possible
// values of U can be represented with type T.
template <typename _From, typename _To>
constexpr bool __is_value_preserving_v =
  (::cuda::std::is_integral_v<_From> && ::cuda::std::is_integral_v<_To>
   && ::cuda::__is_integer_representable_v<_From, _To>)
  || (::cuda::is_floating_point_v<_From> && ::cuda::is_floating_point_v<_To>
      && ::cuda::std::__fp_is_implicit_conversion_v<_From, _To>)
  || (::cuda::std::is_integral_v<_From> && ::cuda::is_floating_point_v<_To>
      && ::cuda::std::numeric_limits<_From>::digits <= ::cuda::std::numeric_limits<_To>::digits);

template <typename _From, typename _ValueType, typename = void>
constexpr bool __is_constexpr_wrapper_value_preserving_v = false;

template <typename _From, typename _ValueType>
constexpr bool __is_constexpr_wrapper_value_preserving_v<_From, _ValueType, ::cuda::std::void_t<decltype(_From::value)>> =
  ::cuda::std::is_arithmetic_v<::cuda::std::remove_cvref_t<decltype(_From::value)>>
  && __is_value_preserving_v<::cuda::std::remove_cvref_t<decltype(_From::value)>, _ValueType>;

// [simd.ctor] implicit value constructor
template <typename _Up, typename _ValueType, typename _From = ::cuda::std::remove_cvref_t<_Up>>
_CCCL_CONCEPT __is_value_ctor_implicit =
  ::cuda::std::convertible_to<_Up, _ValueType>
  && ((!::cuda::std::is_arithmetic_v<_From> && !__constexpr_wrapper_like<_From>)
      || (::cuda::std::is_arithmetic_v<_From> && __is_value_preserving_v<_From, _ValueType>)
      || (__constexpr_wrapper_like<_From> && __is_constexpr_wrapper_value_preserving_v<_From, _ValueType>) );

// [conv.rank], integer conversion rank for [simd.ctor] p7

template <typename _Tp>
inline constexpr int __integer_conversion_rank = 0;

template <>
inline constexpr int __integer_conversion_rank<signed char> = 1;
template <>
inline constexpr int __integer_conversion_rank<unsigned char> = 1;
template <>
inline constexpr int __integer_conversion_rank<char> = 1;
template <>
inline constexpr int __integer_conversion_rank<short> = 2;
template <>
inline constexpr int __integer_conversion_rank<unsigned short> = 2;
template <>
inline constexpr int __integer_conversion_rank<int> = 3;
template <>
inline constexpr int __integer_conversion_rank<unsigned int> = 3;
template <>
inline constexpr int __integer_conversion_rank<long> = 4;
template <>
inline constexpr int __integer_conversion_rank<unsigned long> = 4;
template <>
inline constexpr int __integer_conversion_rank<long long> = 5;
template <>
inline constexpr int __integer_conversion_rank<unsigned long long> = 5;
#if _CCCL_HAS_INT128()
template <>
inline constexpr int __integer_conversion_rank<__int128_t> = 6;
template <>
inline constexpr int __integer_conversion_rank<__uint128_t> = 6;
#endif // _CCCL_HAS_INT128()

// [conv.rank], floating-point conversion rank for [simd.ctor] p7

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

// [simd.ctor] p7: explicit(see below) for basic_vec(const basic_vec<U, UAbi>&)
// explicit evaluates to true if either:
//   - conversion from U to value_type is not value-preserving, or
//   - both U and value_type are integral and integer_conversion_rank(U) > rank(value_type), or
//   - both U and value_type are floating-point and fp_conversion_rank(U) > rank(value_type)
template <typename _Up, typename _ValueType>
constexpr bool __is_vec_ctor_explicit =
  !__is_value_preserving_v<_Up, _ValueType>
  || (::cuda::std::is_integral_v<_Up> && ::cuda::std::is_integral_v<_ValueType>
      && __integer_conversion_rank<_Up> > __integer_conversion_rank<_ValueType>)
  || (::cuda::is_floating_point_v<_Up> && ::cuda::is_floating_point_v<_ValueType>
      && __fp_conversion_rank<_Up> > __fp_conversion_rank<_ValueType>);

// [simd.unary], operator constraints

template <typename _Tp>
_CCCL_CONCEPT __has_pre_increment = _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)((++__t));

template <typename _Tp>
_CCCL_CONCEPT __has_post_increment = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t++));

template <typename _Tp>
_CCCL_CONCEPT __has_pre_decrement = _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)((--__t));

template <typename _Tp>
_CCCL_CONCEPT __has_post_decrement = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t--));

template <typename _Tp>
_CCCL_CONCEPT __has_negate = _CCCL_REQUIRES_EXPR((_Tp), const _Tp __t)((!__t));

template <typename _Tp>
_CCCL_CONCEPT __has_bitwise_not = _CCCL_REQUIRES_EXPR((_Tp), const _Tp __t)((~__t));

template <typename _Tp>
_CCCL_CONCEPT __has_unary_plus = _CCCL_REQUIRES_EXPR((_Tp), const _Tp __t)((+__t));

template <typename _Tp>
_CCCL_CONCEPT __has_unary_minus = _CCCL_REQUIRES_EXPR((_Tp), const _Tp __t)((-__t));

// [simd.binary], binary operator constraints

template <typename _Tp>
_CCCL_CONCEPT __has_binary_plus = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a + __b));

template <typename _Tp>
_CCCL_CONCEPT __has_binary_minus = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a - __b));

template <typename _Tp>
_CCCL_CONCEPT __has_multiplies = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a * __b));

template <typename _Tp>
_CCCL_CONCEPT __has_divides = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a / __b));

template <typename _Tp>
_CCCL_CONCEPT __has_modulo = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a % __b));

template <typename _Tp>
_CCCL_CONCEPT __has_bitwise_and = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a & __b));

template <typename _Tp>
_CCCL_CONCEPT __has_bitwise_or = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a | __b));

template <typename _Tp>
_CCCL_CONCEPT __has_bitwise_xor = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a ^ __b));

template <typename _Tp>
_CCCL_CONCEPT __has_shift_left = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a << __b));

template <typename _Tp>
_CCCL_CONCEPT __has_shift_right = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a >> __b));

template <typename _Tp>
_CCCL_CONCEPT __has_shift_left_size = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t << __simd_size_type{}));

template <typename _Tp>
_CCCL_CONCEPT __has_shift_right_size = _CCCL_REQUIRES_EXPR((_Tp), _Tp __t)((__t >> __simd_size_type{}));

// [simd.comparison], comparison operator constraints

template <typename _Tp>
_CCCL_CONCEPT __has_equal_to = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a == __b));

template <typename _Tp>
_CCCL_CONCEPT __has_not_equal_to = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a != __b));

template <typename _Tp>
_CCCL_CONCEPT __has_greater_equal = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a >= __b));

template <typename _Tp>
_CCCL_CONCEPT __has_less_equal = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a <= __b));

template <typename _Tp>
_CCCL_CONCEPT __has_greater = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a > __b));

template <typename _Tp>
_CCCL_CONCEPT __has_less = _CCCL_REQUIRES_EXPR((_Tp), _Tp __a, _Tp __b)((__a < __b));
} // namespace cuda::std::simd

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_CONCEPTS_H
