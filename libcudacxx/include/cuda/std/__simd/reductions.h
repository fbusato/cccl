//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_REDUCTIONS_H
#define _CUDA_STD___SIMD_REDUCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__simd/basic_mask.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/declaration.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD
// [simd.expos], reduction-binary-operation concept

template <typename _BinaryOp, typename _Tp>
_CCCL_CONCEPT __reduction_binary_operation = _CCCL_REQUIRES_EXPR(
  (_BinaryOp, _Tp), const _BinaryOp __binary_op, const vec<_Tp, 1> __v)(_Same_as(vec<_Tp, 1>) __binary_op(__v, __v));

template <typename _Tp, typename _BinaryOp>
inline constexpr bool __is_nothrow_reduction_binary_operation_v = noexcept(
  ::cuda::std::declval<const _BinaryOp&>()(::cuda::std::declval<vec<_Tp, 1>>(), ::cuda::std::declval<vec<_Tp, 1>>()));

template <typename _Tp, typename _BinaryOp>
constexpr bool __is_reduce_default_supported_operation_v =
  ::cuda::std::is_same_v<_BinaryOp, ::cuda::std::plus<>> //
  || ::cuda::std::is_same_v<_BinaryOp, ::cuda::std::multiplies<>>
  || ::cuda::std::is_same_v<_BinaryOp, ::cuda::std::bit_and<>>
  || ::cuda::std::is_same_v<_BinaryOp, ::cuda::std::bit_or<>>
  || ::cuda::std::is_same_v<_BinaryOp, ::cuda::std::bit_xor<>>;

template <typename _Tp, typename _BinaryOp>
[[nodiscard]] _CCCL_API constexpr _Tp __default_identity_element() noexcept
{
  if constexpr (::cuda::std::is_same_v<_BinaryOp, ::cuda::std::plus<>>
                || ::cuda::std::is_same_v<_BinaryOp, ::cuda::std::bit_or<>>
                || ::cuda::std::is_same_v<_BinaryOp, ::cuda::std::bit_xor<>>)
  {
    return _Tp{};
  }
  else if constexpr (::cuda::std::is_same_v<_BinaryOp, ::cuda::std::multiplies<>>)
  {
    return _Tp(1);
  }
  else if constexpr (::cuda::std::is_same_v<_BinaryOp, ::cuda::std::bit_and<>>)
  {
    return static_cast<_Tp>(~_Tp{});
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>,
                  "No default identity element for this BinaryOperation; provide one explicitly");
    return _Tp{};
  }
}

// [simd.reductions], reduce

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _BinaryOperation = ::cuda::std::plus<>)
_CCCL_REQUIRES(__reduction_binary_operation<_BinaryOperation, _Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp
reduce(const basic_vec<_Tp, _Abi>& __x, _BinaryOperation __binary_op = ::cuda::std::plus<>{}) noexcept(
  __is_nothrow_reduction_binary_operation_v<_Tp, _BinaryOperation>)
{
  vec<_Tp, 1> __result{__x[0]};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 1; __i < __x.__size; ++__i)
  {
    __result = __binary_op(__result, vec<_Tp, 1>{__x[__i]});
  }
  return __result[0];
}

// We need two overloads:
// 1) An argument for identity_element is provided for the invocation
// 2) unless BinaryOperation is one of plus<>, multiplies<>, bit_and<>, bit_or<>, or bit_xor<>
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _BinaryOperation)
_CCCL_REQUIRES(__reduction_binary_operation<_BinaryOperation, _Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp reduce(
  const basic_vec<_Tp, _Abi>& __x,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  _BinaryOperation __binary_op,
  const ::cuda::std::type_identity_t<_Tp>
    __identity_element) noexcept(__is_nothrow_reduction_binary_operation_v<_Tp, _BinaryOperation>)
{
  vec<_Tp, 1> __result{__identity_element};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __x.__size; ++__i)
  {
    if (__mask[__i])
    {
      __result = __binary_op(__result, vec<_Tp, 1>{__x[__i]});
    }
  }
  return __result[0];
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _BinaryOperation = ::cuda::std::plus<>)
_CCCL_REQUIRES(__reduction_binary_operation<_BinaryOperation, _Tp> _CCCL_AND
                 __is_reduce_default_supported_operation_v<_Tp, _BinaryOperation>)
[[nodiscard]] _CCCL_API constexpr _Tp
reduce(const basic_vec<_Tp, _Abi>& __x,
       const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
       _BinaryOperation __binary_op =
         ::cuda::std::plus<>{}) noexcept(__is_nothrow_reduction_binary_operation_v<_Tp, _BinaryOperation>)
{
  return ::cuda::std::simd::reduce(
    __x, __mask, __binary_op, ::cuda::std::simd::__default_identity_element<_Tp, _BinaryOperation>());
}

// [simd.reductions], reduce_min

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::totally_ordered<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp reduce_min(const basic_vec<_Tp, _Abi>& __x) noexcept
{
  auto __result = __x[0];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 1; __i < __x.__size; ++__i)
  {
    const auto __val = __x[__i];
    if (!(__result < __val))
    {
      __result = __val;
    }
  }
  return __result;
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::totally_ordered<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp
reduce_min(const basic_vec<_Tp, _Abi>& __x, const typename basic_vec<_Tp, _Abi>::mask_type& __mask) noexcept
{
  auto __result = ::cuda::std::numeric_limits<_Tp>::max();
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __x.__size; ++__i)
  {
    if (__mask[__i])
    {
      const auto __val = __x[__i];
      if (!(__result < __val))
      {
        __result = __val;
      }
    }
  }
  return __result;
}

// [simd.reductions], reduce_max

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::totally_ordered<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp reduce_max(const basic_vec<_Tp, _Abi>& __x) noexcept
{
  auto __result = __x[0];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 1; __i < __x.__size; ++__i)
  {
    const auto __val = __x[__i];
    if (!(__val < __result))
    {
      __result = __val;
    }
  }
  return __result;
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::std::totally_ordered<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp
reduce_max(const basic_vec<_Tp, _Abi>& __x, const typename basic_vec<_Tp, _Abi>::mask_type& __mask) noexcept
{
  auto __result = ::cuda::std::numeric_limits<_Tp>::lowest();
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __x.__size; ++__i)
  {
    if (__mask[__i])
    {
      const auto __val = __x[__i];
      if (!(__val < __result))
      {
        __result = __val;
      }
    }
  }
  return __result;
}

// [simd.mask.reductions], mask reductions

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr bool all_of(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __k.__size; ++__i)
  {
    if (!__k[__i])
    {
      return false;
    }
  }
  return true;
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr bool any_of(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __k.__size; ++__i)
  {
    if (__k[__i])
    {
      return true;
    }
  }
  return false;
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr bool none_of(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  return !::cuda::std::simd::any_of(__k);
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_count(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  __simd_size_type __count = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __k.__size; ++__i)
  {
    __count += static_cast<__simd_size_type>(__k[__i]);
  }
  return __count;
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_min_index(const basic_mask<_Bytes, _Abi>& __k)
{
  _CCCL_ASSERT(::cuda::std::simd::any_of(__k), "No bits are set");
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __k.__size; ++__i)
  {
    if (__k[__i])
    {
      return __i;
    }
  }
  _CCCL_UNREACHABLE();
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_max_index(const basic_mask<_Bytes, _Abi>& __k)
{
  _CCCL_ASSERT(::cuda::std::simd::any_of(__k), "No bits are set");
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = __k.__size - 1; __i >= 0; --__i)
  {
    if (__k[__i])
    {
      return __i;
    }
  }
  _CCCL_UNREACHABLE();
}

// Scalar bool overloads

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr bool all_of(_Tp __x) noexcept
{
  return __x;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr bool any_of(_Tp __x) noexcept
{
  return __x;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr bool none_of(_Tp __x) noexcept
{
  return !__x;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_count(_Tp __x) noexcept
{
  return __x;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_min_index(_Tp __x)
{
  _CCCL_ASSERT(__x, "No bits are set");
  return 0;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_max_index(_Tp __x)
{
  _CCCL_ASSERT(__x, "No bits are set");
  return 0;
}
_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_REDUCTIONS_H
