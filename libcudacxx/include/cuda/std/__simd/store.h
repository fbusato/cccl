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

#include <cuda/__memory/ptr_rebind.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/assume_aligned.h>
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

template <typename _Tp, typename _Abi, typename _Up, typename... _Flags>
_CCCL_API constexpr void __check_store_preconditions(_Up* const __ptr, flags<_Flags...>) noexcept
{
  static_assert(__is_vectorizable_v<_Up>, "range_value_t<R> must be a vectorizable type");
  static_assert(__explicitly_convertible_to<_Up, _Tp>,
                "value_type must satisfy explicitly-convertible-to<range_value_t<R>>");
  static_assert(__has_convert_flag_v<_Flags...> || __is_value_preserving_v<_Tp, _Up>,
                "Conversion from value_type to range_value_t<R> is not value-preserving; use flag_convert");
  ::cuda::std::simd::__assert_load_store_alignment<basic_vec<_Tp, _Abi>, _Up, _Flags...>(__ptr);
}

// [simd.loadstore] helper: core partial store to pointer + count + mask
template <typename _Tp, typename _Abi, typename _Up, typename... _Flags>
_CCCL_API constexpr void __partial_store_to_ptr(
  const basic_vec<_Tp, _Abi>& __v,
  _Up* const __ptr,
  const __simd_size_type __count,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __flags = {}) noexcept
{
  ::cuda::std::simd::__check_store_preconditions<_Tp, _Abi>(__ptr, __flags);
  constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
  {
    if (__mask[__i] && __i < __count)
    {
      __ptr[__i] = static_cast<_Up>(__v[__i]);
    }
  }
}

template <typename _Tp, typename _Abi, typename _Up, typename... _Flags>
_CCCL_API constexpr void
__full_store_to_ptr(const basic_vec<_Tp, _Abi>& __v, _Up* const __ptr, flags<_Flags...> __flags) noexcept
{
  ::cuda::std::simd::__check_store_preconditions<_Tp, _Abi>(__ptr, __flags);
  constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;

  if constexpr (__has_aligned_flag_v<_Flags...> || __has_overaligned_flag_v<_Flags...>)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      constexpr auto __base_alignment = alignment_v<basic_vec<_Tp, _Abi>, _Up>;
      constexpr auto __ptr_alignment  = ::cuda::std::max(__base_alignment, __overaligned_value_v<_Flags...>);
      constexpr auto __data_size      = __simd_size * sizeof(_Up);

      // When _CCCL_IF_NOT_CONSTEVAL falls back to __builtin_is_constant_evaluated(),
      // EDG requires locals to be initialized for the function to remain constexpr
#if __cpp_if_consteval >= 202106L || _CCCL_HAS_IF_CONSTEVAL_IN_CXX20()
      _Up __tmp[__simd_size];
#else
      _Up __tmp[__simd_size] = {};
#endif
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
      {
        __tmp[__i] = static_cast<_Up>(__v[__i]);
      }
      // vectorized store to pointer
      if constexpr (__is_cuda_vectoriazable_v<_Up> && __simd_size > 1 && __ptr_alignment >= __data_size)
      {
        struct alignas(__data_size) __aligned_t
        {
          char __data[__data_size];
        };
        // nvcc performance bug: memcpy to pointer could not be vectorized
        const auto __aligned_ptr = ::cuda::ptr_rebind<__aligned_t>(__ptr);
#if __cpp_if_consteval >= 202106L || _CCCL_HAS_IF_CONSTEVAL_IN_CXX20()
        __aligned_t __data;
#else
        __aligned_t __data{};
#endif
        ::cuda::std::memcpy(&__data, &__tmp, sizeof(__tmp));
        *::cuda::std::assume_aligned<__ptr_alignment>(__aligned_ptr) = __data;
      }
      // rely on compiler vectorization
      else
      {
        const auto __aligned_ptr = ::cuda::std::assume_aligned<__ptr_alignment>(__ptr);
        _CCCL_PRAGMA_UNROLL_FULL()
        for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
        {
          __aligned_ptr[__i] = __tmp[__i];
        }
      }
      return;
    }
  }
  constexpr auto __true_mask = typename basic_vec<_Tp, _Abi>::mask_type(true);

  ::cuda::std::simd::__partial_store_to_ptr(__v, __ptr, __simd_size, __true_mask, __flags);
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
  flags<_Flags...> __f = {})
{
  static_assert(indirectly_writable<ranges::iterator_t<_Range>, ranges::range_value_t<_Range>>,
                "ranges::iterator_t<R> must model indirectly_writable<ranges::range_value_t<R>>");
  const auto __size = static_cast<__simd_size_type>(::cuda::std::ranges::size(__r));

  ::cuda::std::simd::__partial_store_to_ptr(__v, ::cuda::std::ranges::data(__r), __size, __mask, __f);
}

// partial_store: range, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range> _CCCL_AND
                 __explicitly_convertible_to<ranges::range_value_t<_Range>, _Tp>)
_CCCL_API constexpr void partial_store(const basic_vec<_Tp, _Abi>& __v, _Range&& __r, flags<_Flags...> __f = {})
{
  constexpr auto __true_mask = typename basic_vec<_Tp, _Abi>::mask_type(true);

  ::cuda::std::simd::partial_store(__v, ::cuda::std::forward<_Range>(__r), __true_mask, __f);
}

// partial_store: iterator + count, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  const _Ip __first,
  const iter_difference_t<_Ip> __n,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  static_assert(indirectly_writable<_Ip, iter_value_t<_Ip>>, "I must model indirectly_writable<iter_value_t<I>>");
  const auto __size = static_cast<__simd_size_type>(__n);

  ::cuda::std::simd::__partial_store_to_ptr(__v, ::cuda::std::to_address(__first), __size, __mask, __f);
}

// partial_store: iterator + count, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v, const _Ip __first, const iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  constexpr auto __true_mask = typename basic_vec<_Tp, _Abi>::mask_type(true);

  ::cuda::std::simd::partial_store(__v, __first, __n, __true_mask, __f);
}

// partial_store: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  const _Ip __first,
  const _Sp __last,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  static_assert(indirectly_writable<_Ip, iter_value_t<_Ip>>, "I must model indirectly_writable<iter_value_t<I>>");
  const auto __size = static_cast<__simd_size_type>(::cuda::std::distance(__first, __last));

  ::cuda::std::simd::__partial_store_to_ptr(__v, ::cuda::std::to_address(__first), __size, __mask, __f);
}

// partial_store: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void
partial_store(const basic_vec<_Tp, _Abi>& __v, const _Ip __first, const _Sp __last, flags<_Flags...> __f = {})
{
  constexpr auto __true_mask = typename basic_vec<_Tp, _Abi>::mask_type(true);

  ::cuda::std::simd::partial_store(__v, __first, __last, __true_mask, __f);
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
  constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;
  if constexpr (__has_static_size<_Range>)
  {
    static_assert(__static_range_size_v<_Range> >= __simd_size,
                  "unchecked_store requires ranges::size(r) >= V::size()");
  }
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::ranges::size(__r), __v.size()),
               "unchecked_store requires ranges::size(r) >= V::size()");

  ::cuda::std::simd::__partial_store_to_ptr(__v, ::cuda::std::ranges::data(__r), __simd_size, __mask, __f);
}

// unchecked_store: range, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range> _CCCL_AND
                 __explicitly_convertible_to<ranges::range_value_t<_Range>, _Tp>)
_CCCL_API constexpr void unchecked_store(const basic_vec<_Tp, _Abi>& __v, _Range&& __r, flags<_Flags...> __f = {})
{
  if constexpr (__has_static_size<_Range>)
  {
    static_assert(__static_range_size_v<_Range> >= basic_vec<_Tp, _Abi>::__size,
                  "unchecked_store requires ranges::size(r) >= V::size()");
  }
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::ranges::size(__r), __v.size()),
               "unchecked_store requires ranges::size(r) >= V::size()");

  ::cuda::std::simd::__full_store_to_ptr(__v, ::cuda::std::ranges::data(__r), __f);
}

// unchecked_store: iterator + count, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v,
  const _Ip __first,
  const iter_difference_t<_Ip> __n,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__n, __v.size()), "unchecked_store requires n >= V::size()");
  constexpr auto __simd_size = basic_vec<_Tp, _Abi>::size();

  ::cuda::std::simd::__partial_store_to_ptr(__v, ::cuda::std::to_address(__first), __simd_size, __mask, __f);
}

// unchecked_store: iterator + count, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v, const _Ip __first, const iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__n, __v.size()), "unchecked_store requires n >= V::size()");

  ::cuda::std::simd::__full_store_to_ptr(__v, ::cuda::std::to_address(__first), __f);
}

// unchecked_store: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v,
  const _Ip __first,
  const _Sp __last,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::distance(__first, __last), __v.size()),
               "unchecked_store requires distance(first, last) >= V::size()");
  constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;

  ::cuda::std::simd::__partial_store_to_ptr(__v, ::cuda::std::to_address(__first), __simd_size, __mask, __f);
}

// unchecked_store: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip> _CCCL_AND
                 __explicitly_convertible_to<iter_value_t<_Ip>, _Tp>)
_CCCL_API constexpr void
unchecked_store(const basic_vec<_Tp, _Abi>& __v, const _Ip __first, const _Sp __last, flags<_Flags...> __f = {})
{
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::distance(__first, __last), __v.size()),
               "unchecked_store requires distance(first, last) >= V::size()");

  ::cuda::std::simd::__full_store_to_ptr(__v, ::cuda::std::to_address(__first), __f);
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_STORE_H
