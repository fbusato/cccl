//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_STORE_H
#define _CUDA_STD___SIMD_STORE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/data.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/concepts.h>
#include <cuda/std/__simd/flag.h>
#include <cuda/std/__simd/utility.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.loadstore] helper: core partial store to pointer + count + mask
template <typename _Tp, typename _Abi, typename _Up, typename... _Flags>
_CCCL_API constexpr void __partial_store_to_ptr(
  const basic_vec<_Tp, _Abi>& __v,
  _Up* __ptr,
  __simd_size_type __count,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask)
{
  static_assert(__is_vectorizable_v<_Up>, "range_value_t<R> must be a vectorizable type");
  static_assert(__explicitly_convertible_to<_Up, _Tp>,
                "value_type must satisfy explicitly-convertible-to<range_value_t<R>>");
  static_assert(__has_convert_flag_v<_Flags...> || __is_value_preserving_v<_Tp, _Up>,
                "Conversion from value_type to range_value_t<R> is not value-preserving; use flag_convert");
  ::cuda::std::simd::__assert_load_store_alignment<basic_vec<_Tp, _Abi>, _Up, _Flags...>(__ptr);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < basic_vec<_Tp, _Abi>::size; ++__i)
  {
    if (__mask[__i] && __i < __count)
    {
      __ptr[__i] = static_cast<_Up>(__v[__i]);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] partial_store

// partial_store: range, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range> _CCCL_AND
                 __explicitly_convertible_to<ranges::range_value_t<_Range>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Range&& __r,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> = {})
{
  static_assert(indirectly_writable<ranges::iterator_t<_Range>, ranges::range_value_t<_Range>>,
                "ranges::iterator_t<R> must model indirectly_writable<ranges::range_value_t<R>>");
  using _Up = ranges::range_value_t<_Range>;
  ::cuda::std::simd::__partial_store_to_ptr<_Tp, _Abi, _Up, _Flags...>(
    __v, ::cuda::std::ranges::data(__r), static_cast<__simd_size_type>(::cuda::std::ranges::size(__r)), __mask);
}

// partial_store: range, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range> _CCCL_AND
                 __explicitly_convertible_to<ranges::range_value_t<_Range>, _Tp>)
_CCCL_API constexpr void partial_store(const basic_vec<_Tp, _Abi>& __v, _Range&& __r, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::partial_store(
    __v, ::cuda::std::forward<_Range>(__r), typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

// partial_store: iterator + count, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Ip __first,
  iter_difference_t<_Ip> __n,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> = {})
{
  static_assert(indirectly_writable<_Ip, iter_value_t<_Ip>>, "I must model indirectly_writable<iter_value_t<I>>");
  using _Up = iter_value_t<_Ip>;
  ::cuda::std::simd::__partial_store_to_ptr<_Tp, _Abi, _Up, _Flags...>(
    __v, ::cuda::std::to_address(__first), static_cast<__simd_size_type>(__n), __mask);
}

// partial_store: iterator + count, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void
partial_store(const basic_vec<_Tp, _Abi>& __v, _Ip __first, iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::partial_store(__v, __first, __n, typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

// partial_store: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Ip __first,
  _Sp __last,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> = {})
{
  static_assert(indirectly_writable<_Ip, iter_value_t<_Ip>>, "I must model indirectly_writable<iter_value_t<I>>");
  using _Up = iter_value_t<_Ip>;
  ::cuda::std::simd::__partial_store_to_ptr<_Tp, _Abi, _Up, _Flags...>(
    __v,
    ::cuda::std::to_address(__first),
    static_cast<__simd_size_type>(::cuda::std::distance(__first, __last)),
    __mask);
}

// partial_store: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void
partial_store(const basic_vec<_Tp, _Abi>& __v, _Ip __first, _Sp __last, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::partial_store(__v, __first, __last, typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] unchecked_store

// unchecked_store: range, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range> _CCCL_AND
                 __explicitly_convertible_to<ranges::range_value_t<_Range>, _Tp>)
_CCCL_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Range&& __r,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  if constexpr (__has_static_size<_Range>)
  {
    static_assert(__static_range_size_v<_Range> >= basic_vec<_Tp, _Abi>::size(),
                  "unchecked_store requires ranges::size(r) >= V::size()");
  }
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::ranges::size(__r), __v.size),
               "unchecked_store requires ranges::size(r) >= V::size()");
  ::cuda::std::simd::partial_store(__v, ::cuda::std::forward<_Range>(__r), __mask, __f);
}

// unchecked_store: range, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range> _CCCL_AND
                 __explicitly_convertible_to<ranges::range_value_t<_Range>, _Tp>)
_CCCL_API constexpr void unchecked_store(const basic_vec<_Tp, _Abi>& __v, _Range&& __r, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::unchecked_store(
    __v, ::cuda::std::forward<_Range>(__r), typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

// unchecked_store: iterator + count, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Ip __first,
  iter_difference_t<_Ip> __n,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__n, __v.size), "unchecked_store requires n >= V::size()");
  ::cuda::std::simd::partial_store(__v, __first, __n, __mask, __f);
}

// unchecked_store: iterator + count, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void
unchecked_store(const basic_vec<_Tp, _Abi>& __v, _Ip __first, iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::unchecked_store(__v, __first, __n, typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

// unchecked_store: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Ip __first,
  _Sp __last,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::distance(__first, __last), __v.size),
               "unchecked_store requires distance(first, last) >= V::size()");
  ::cuda::std::simd::partial_store(__v, __first, __last, __mask, __f);
}

// unchecked_store: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void
unchecked_store(const basic_vec<_Tp, _Abi>& __v, _Ip __first, _Sp __last, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::unchecked_store(__v, __first, __last, typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_STORE_H
