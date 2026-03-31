//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_LOAD_STORE_H
#define _CUDA_STD___SIMD_LOAD_STORE_H

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
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/data.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/concepts.h>
#include <cuda/std/__simd/flag.h>
#include <cuda/std/__simd/utility.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::std::simd
{
// [simd.loadstore] helper: resolves default V template parameter for load functions
// When _Vp = void (default), resolves to basic_vec<_Up>; otherwise uses the explicit _Vp
template <typename _Vp, typename _Up>
struct __load_vec_type
{
  using type = _Vp;
};

template <typename _Up>
struct __load_vec_type<void, _Up>
{
  using type = basic_vec<_Up>;
};

template <typename _Vp, typename _Up>
using __load_vec_t = typename __load_vec_type<_Vp, _Up>::type;

// [simd.loadstore] helper: core partial load from pointer + count + mask
template <typename _Result, typename _Up, typename... _Flags>
[[nodiscard]] _CCCL_API constexpr _Result
__partial_load_from_ptr(const _Up* __ptr, __simd_size_type __count, const typename _Result::mask_type& __mask)
{
  using _Tp = typename _Result::value_type;
  static_assert(::cuda::std::same_as<::cuda::std::remove_cvref_t<_Result>, _Result>,
                "V must not be a reference or cv-qualified type");
  static_assert(__is_vectorizable_v<_Tp> && __is_abi_tag_v<typename _Result::abi_type>,
                "V must be an enabled specialization of basic_vec");
  static_assert(__is_vectorizable_v<_Up>, "range_value_t<R> must be a vectorizable type");
  static_assert(__explicitly_convertible_to<_Tp, _Up>,
                "range_value_t<R> must satisfy explicitly-convertible-to<value_type>");
  static_assert(__has_convert_flag_v<_Flags...> || __is_value_preserving_v<_Up, _Tp>,
                "Conversion from range_value_t<R> to value_type is not value-preserving; use flag_convert");
  ::cuda::std::simd::__assert_load_store_alignment<_Result, _Up, _Flags...>(__ptr);
  _Result __result{};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < _Result::size; ++__i)
  {
    if (__mask[__i] && __i < __count)
    {
      __result.__set(__i, static_cast<_Tp>(__ptr[__i]));
    }
  }
  return __result;
}

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
// [simd.loadstore] partial_load

// partial_load: range, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>> partial_load(
  _Range&& __r,
  const typename __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>::mask_type& __mask,
  flags<_Flags...> = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>;
  using _Up     = ::cuda::std::ranges::range_value_t<_Range>;
  return ::cuda::std::simd::__partial_load_from_ptr<_Result, _Up, _Flags...>(
    ::cuda::std::ranges::data(__r), static_cast<__simd_size_type>(::cuda::std::ranges::size(__r)), __mask);
}

// partial_load: range, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>
partial_load(_Range&& __r, flags<_Flags...> __f = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>;
  return ::cuda::std::simd::partial_load<_Vp>(
    ::cuda::std::forward<_Range>(__r), typename _Result::mask_type(true), __f);
}

// partial_load: iterator + count, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>> partial_load(
  _Ip __first,
  ::cuda::std::iter_difference_t<_Ip> __n,
  const typename __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>;
  using _Up     = ::cuda::std::iter_value_t<_Ip>;
  return ::cuda::std::simd::__partial_load_from_ptr<_Result, _Up, _Flags...>(
    ::cuda::std::to_address(__first), static_cast<__simd_size_type>(__n), __mask);
}

// partial_load: iterator + count, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>
partial_load(_Ip __first, ::cuda::std::iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>;
  return ::cuda::std::simd::partial_load<_Vp>(__first, __n, typename _Result::mask_type(true), __f);
}

// partial_load: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip> _CCCL_AND ::cuda::std::sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>> partial_load(
  _Ip __first,
  _Sp __last,
  const typename __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>;
  using _Up     = ::cuda::std::iter_value_t<_Ip>;
  return ::cuda::std::simd::__partial_load_from_ptr<_Result, _Up, _Flags...>(
    ::cuda::std::to_address(__first), static_cast<__simd_size_type>(::cuda::std::distance(__first, __last)), __mask);
}

// partial_load: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip> _CCCL_AND ::cuda::std::sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>
partial_load(_Ip __first, _Sp __last, flags<_Flags...> __f = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>;
  return ::cuda::std::simd::partial_load<_Vp>(__first, __last, typename _Result::mask_type(true), __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] unchecked_load

// unchecked_load: range, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>> unchecked_load(
  _Range&& __r,
  const typename __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>;
  if constexpr (__has_static_size<_Range>)
  {
    static_assert(__static_range_size_v<_Range> >= _Result::size(),
                  "unchecked_load requires ranges::size(r) >= V::size()");
  }
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::ranges::size(__r), _Result::size()),
               "unchecked_load requires ranges::size(r) >= V::size()");
  return ::cuda::std::simd::partial_load<_Vp>(::cuda::std::forward<_Range>(__r), __mask, __f);
}

// unchecked_load: range, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>
unchecked_load(_Range&& __r, flags<_Flags...> __f = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>;
  return ::cuda::std::simd::unchecked_load<_Vp>(
    ::cuda::std::forward<_Range>(__r), typename _Result::mask_type(true), __f);
}

// unchecked_load: iterator + count, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>> unchecked_load(
  _Ip __first,
  ::cuda::std::iter_difference_t<_Ip> __n,
  const typename __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__n, _Result::size()), "unchecked_load requires n >= V::size()");
  return ::cuda::std::simd::partial_load<_Vp>(__first, __n, __mask, __f);
}

// unchecked_load: iterator + count, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>
unchecked_load(_Ip __first, ::cuda::std::iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>;
  return ::cuda::std::simd::unchecked_load<_Vp>(__first, __n, typename _Result::mask_type(true), __f);
}

// unchecked_load: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip> _CCCL_AND ::cuda::std::sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>> unchecked_load(
  _Ip __first,
  _Sp __last,
  const typename __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::distance(__first, __last), _Result::size()),
               "unchecked_load requires distance(first, last) >= V::size()");
  return ::cuda::std::simd::partial_load<_Vp>(__first, __last, __mask, __f);
}

// unchecked_load: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip> _CCCL_AND ::cuda::std::sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>
unchecked_load(_Ip __first, _Sp __last, flags<_Flags...> __f = {})
{
  using _Result = __load_vec_t<_Vp, ::cuda::std::iter_value_t<_Ip>>;
  return ::cuda::std::simd::unchecked_load<_Vp>(__first, __last, typename _Result::mask_type(true), __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] partial_store

// partial_store: range, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>
                 _CCCL_AND __explicitly_convertible_to<::cuda::std::ranges::range_value_t<_Range>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Range&& __r,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> = {})
{
  static_assert(
    ::cuda::std::indirectly_writable<::cuda::std::ranges::iterator_t<_Range>, ::cuda::std::ranges::range_value_t<_Range>>,
    "ranges::iterator_t<R> must model indirectly_writable<ranges::range_value_t<R>>");
  using _Up = ::cuda::std::ranges::range_value_t<_Range>;
  ::cuda::std::simd::__partial_store_to_ptr<_Tp, _Abi, _Up, _Flags...>(
    __v, ::cuda::std::ranges::data(__r), static_cast<__simd_size_type>(::cuda::std::ranges::size(__r)), __mask);
}

// partial_store: range, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>
                 _CCCL_AND __explicitly_convertible_to<::cuda::std::ranges::range_value_t<_Range>, _Tp>)
_CCCL_API constexpr void partial_store(const basic_vec<_Tp, _Abi>& __v, _Range&& __r, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::partial_store(
    __v, ::cuda::std::forward<_Range>(__r), typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

// partial_store: iterator + count, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(
  ::cuda::std::contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<::cuda::std::iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Ip __first,
  ::cuda::std::iter_difference_t<_Ip> __n,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> = {})
{
  static_assert(::cuda::std::indirectly_writable<_Ip, ::cuda::std::iter_value_t<_Ip>>,
                "I must model indirectly_writable<iter_value_t<I>>");
  using _Up = ::cuda::std::iter_value_t<_Ip>;
  ::cuda::std::simd::__partial_store_to_ptr<_Tp, _Abi, _Up, _Flags...>(
    __v, ::cuda::std::to_address(__first), static_cast<__simd_size_type>(__n), __mask);
}

// partial_store: iterator + count, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(
  ::cuda::std::contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<::cuda::std::iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v, _Ip __first, ::cuda::std::iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::partial_store(__v, __first, __n, typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

// partial_store: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip> _CCCL_AND ::cuda::std::sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<::cuda::std::iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Ip __first,
  _Sp __last,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> = {})
{
  static_assert(::cuda::std::indirectly_writable<_Ip, ::cuda::std::iter_value_t<_Ip>>,
                "I must model indirectly_writable<iter_value_t<I>>");
  using _Up = ::cuda::std::iter_value_t<_Ip>;
  ::cuda::std::simd::__partial_store_to_ptr<_Tp, _Abi, _Up, _Flags...>(
    __v,
    ::cuda::std::to_address(__first),
    static_cast<__simd_size_type>(::cuda::std::distance(__first, __last)),
    __mask);
}

// partial_store: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip> _CCCL_AND ::cuda::std::sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<::cuda::std::iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void
partial_store(const basic_vec<_Tp, _Abi>& __v, _Ip __first, _Sp __last, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::partial_store(__v, __first, __last, typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] unchecked_store

// unchecked_store: range, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>
                 _CCCL_AND __explicitly_convertible_to<::cuda::std::ranges::range_value_t<_Range>, _Tp>)
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
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>
                 _CCCL_AND __explicitly_convertible_to<::cuda::std::ranges::range_value_t<_Range>, _Tp>)
_CCCL_API constexpr void unchecked_store(const basic_vec<_Tp, _Abi>& __v, _Range&& __r, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::unchecked_store(
    __v, ::cuda::std::forward<_Range>(__r), typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

// unchecked_store: iterator + count, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(
  ::cuda::std::contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<::cuda::std::iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Ip __first,
  ::cuda::std::iter_difference_t<_Ip> __n,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__n, __v.size), "unchecked_store requires n >= V::size()");
  ::cuda::std::simd::partial_store(__v, __first, __n, __mask, __f);
}

// unchecked_store: iterator + count, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(
  ::cuda::std::contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<::cuda::std::iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v, _Ip __first, ::cuda::std::iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::unchecked_store(__v, __first, __n, typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}

// unchecked_store: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip> _CCCL_AND ::cuda::std::sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<::cuda::std::iter_value_t<_Ip>, _Tp>)
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
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Ip> _CCCL_AND ::cuda::std::sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<::cuda::std::iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void
unchecked_store(const basic_vec<_Tp, _Abi>& __v, _Ip __first, _Sp __last, flags<_Flags...> __f = {})
{
  ::cuda::std::simd::unchecked_store(__v, __first, __last, typename basic_vec<_Tp, _Abi>::mask_type(true), __f);
}
} // namespace cuda::std::simd

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_LOAD_STORE_H
