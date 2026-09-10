//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_ABI_H
#define _CUDA_STD___SIMD_ABI_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/complex.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_volatile.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

using __simd_size_type = ptrdiff_t;

template <__simd_size_type _Np>
using __simd_size_constant = integral_constant<__simd_size_type, _Np>;

template <size_t _Bytes>
inline constexpr bool __is_vectorizable_byte_size_v =
  (_Bytes == 1 || _Bytes == 2 || _Bytes == 4 || _Bytes == 8
#if _CCCL_HAS_INT128()
   || _Bytes == 16
#endif // _CCCL_HAS_INT128()
  );

template <typename _Tp>
inline constexpr bool __is_vectorizable_floating_point_v =
  ::cuda::is_floating_point_v<_Tp> && !is_same_v<_Tp, long double> && !is_const_v<_Tp> && !is_volatile_v<_Tp>;

// complex<T> where T is a vectorizable floating-point type
template <typename _Tp>
inline constexpr bool __is_complex_vectorizable_v = false;

template <typename _Tp>
inline constexpr bool __is_complex_vectorizable_v<::cuda::std::complex<_Tp>> = __is_vectorizable_floating_point_v<_Tp>;

template <typename _Tp>
inline constexpr bool __is_complex_vectorizable_v<::cuda::complex<_Tp>> = __is_vectorizable_floating_point_v<_Tp>;

#if _CCCL_HAS_HOST_STD_LIB()

template <typename _Tp>
inline constexpr bool __is_complex_vectorizable_v<::std::complex<_Tp>> = __is_vectorizable_floating_point_v<_Tp>;

#endif // _CCCL_HAS_HOST_STD_LIB()

// [simd.expos], vectorizable types:
// all standard integer types, character types, and the types float and double ([basic.fundamental]);
// std::float16_t, std::float32_t, and std::float64_t if defined ([basic.extended.fp]); and
// complex<T> where T is a vectorizable floating-point type.
template <typename _Tp>
inline constexpr bool __is_vectorizable_v =
  ((__is_extended_arithmetic_v<_Tp> && !is_same_v<_Tp, long double>) || __is_complex_vectorizable_v<_Tp>)
  && !is_same_v<_Tp, bool> && !is_const_v<_Tp> && !is_volatile_v<_Tp>;

// [simd.expos.abi], simd ABI tags
template <__simd_size_type _Np>
struct __fixed_size; // internal ABI tag

template <__simd_size_type _Np>
using fixed_size = __fixed_size<_Np>; // implementation-defined ABI

template <typename _Abi>
inline constexpr bool __is_enabled_abi_v = false;

template <__simd_size_type _Np>
inline constexpr bool __is_enabled_abi_v<__fixed_size<_Np>> = (_Np >= 1 && _Np <= 64);

struct __invalid_abi;

template <typename _Tp, __simd_size_type _Np>
struct __deduce_abi
{
  using type = conditional_t<__is_vectorizable_v<_Tp> && _Np >= 1 && _Np <= 64, fixed_size<_Np>, __invalid_abi>;
};

template <typename _Tp, __simd_size_type _Np>
using __deduce_abi_t = typename __deduce_abi<_Tp, _Np>::type; // exposition-only

// TODO(fbusato): this could be optimized by using max access size / sizeof(T)
template <typename _Tp>
using native = __deduce_abi_t<_Tp, 1>; // implementation-defined ABI

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_ABI_H
