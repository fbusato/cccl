//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_SIMPLE_VEC_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_SIMPLE_VEC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__simd/declaration.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::std::simd
{
namespace simd_abi
{
template <__simd_size_type _Np>
struct __fixed_size_simple
{
  static constexpr __simd_size_type __simd_size = _Np;
};
} // namespace simd_abi

// Element-per-slot simd storage for fixed_size_simple ABI
template <typename _Tp, __simd_size_type _Np>
struct __simd_storage<_Tp, simd_abi::__fixed_size_simple<_Np>>
{
  using value_type = _Tp;
  _Tp __data[_Np];

  [[nodiscard]] _CCCL_API constexpr _Tp __get(__simd_size_type __idx) const noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, __simd_size_type{0}, _Np), "Index is out of bounds");
    return __data[__idx];
  }

  _CCCL_API constexpr void __set(__simd_size_type __idx, _Tp __v) noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, __simd_size_type{0}, _Np), "Index is out of bounds");
    __data[__idx] = __v;
  }
};

#define _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_STORAGE_TYPE, _NAME, _OP) \
  [[nodiscard]] _CCCL_API static constexpr _STORAGE_TYPE _NAME(                \
    const _STORAGE_TYPE& __lhs, const _STORAGE_TYPE& __rhs) noexcept           \
  {                                                                            \
    _STORAGE_TYPE __result;                                                    \
    _CCCL_PRAGMA_UNROLL_FULL()                                                 \
    for (__simd_size_type __i = 0; __i < _Np; ++__i)                           \
    {                                                                          \
      __result.__data[__i] = (__lhs.__data[__i] _OP __rhs.__data[__i]);        \
    }                                                                          \
    return __result;                                                           \
  }

#define _CUDA_STD_SIMD_FIXED_SIZE_BINARY_CMP_OP(_NAME, _OP)                \
  [[nodiscard]] _CCCL_API static constexpr _MaskStorage _NAME(              \
    const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept          \
  {                                                                         \
    _MaskStorage __result;                                                  \
    _CCCL_PRAGMA_UNROLL_FULL()                                              \
    for (__simd_size_type __i = 0; __i < _Np; ++__i)                        \
    {                                                                       \
      __result.__data[__i] = (__lhs.__data[__i] _OP __rhs.__data[__i]);     \
    }                                                                       \
    return __result;                                                        \
  }

// Simd operations for fixed_size_simple ABI
template <typename _Tp, __simd_size_type _Np>
struct __simd_operations<_Tp, simd_abi::__fixed_size_simple<_Np>>
{
  using _SimdStorage = __simd_storage<_Tp, simd_abi::__fixed_size_simple<_Np>>;
  using _MaskStorage = __mask_storage<sizeof(_Tp), simd_abi::__fixed_size_simple<_Np>>;

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __broadcast(_Tp __v) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = __v;
    }
    return __result;
  }

  template <typename _Generator, __simd_size_type... _Is>
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __generate_init(_Generator&& __g, ::cuda::std::integer_sequence<__simd_size_type, _Is...>)
  {
    return _SimdStorage{{__g(::cuda::std::integral_constant<__simd_size_type, _Is>())...}};
  }

  template <typename _Generator>
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __generate(_Generator&& __g)
  {
    return __generate_init(__g, ::cuda::std::make_integer_sequence<__simd_size_type, _Np>());
  }

  // Unary operations

  _CCCL_API static constexpr void __increment(_SimdStorage& __s) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __s.__data[__i] += 1;
    }
  }

  _CCCL_API static constexpr void __decrement(_SimdStorage& __s) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __s.__data[__i] -= 1;
    }
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __negate(const _SimdStorage& __s) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = !__s.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __bitwise_not(const _SimdStorage& __s) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = ~__s.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __unary_minus(const _SimdStorage& __s) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = -__s.__data[__i];
    }
    return __result;
  }

  // Binary arithmetic operations

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __plus, +)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __minus, -)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __multiplies, *)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __divides, /)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __modulo, %)

  // Comparison operations

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_CMP_OP(__equal_to, ==)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_CMP_OP(__not_equal_to, !=)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_CMP_OP(__less, <)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_CMP_OP(__less_equal, <=)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_CMP_OP(__greater, >)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_CMP_OP(__greater_equal, >=)

  // Bitwise and shift operations

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __bitwise_and, &)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __bitwise_or, |)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __bitwise_xor, ^)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __shift_left, <<)

  _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP(_SimdStorage, __shift_right, >>)
};

#undef _CUDA_STD_SIMD_FIXED_SIZE_BINARY_STORAGE_OP
#undef _CUDA_STD_SIMD_FIXED_SIZE_BINARY_CMP_OP
} // namespace cuda::std::simd

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_SIMPLE_VEC_H
