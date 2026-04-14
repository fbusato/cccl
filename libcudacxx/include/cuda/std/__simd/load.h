//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_LOAD_H
#define _CUDA_STD___SIMD_LOAD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/__memory/ptr_rebind.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__memory/pointer_traits.h>
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

template <typename _Result, typename _Up, typename... _Flags>
_CCCL_API constexpr void __check_load_preconditions(const _Up* __ptr)
{
  using _Tp = typename _Result::value_type;
  static_assert(same_as<remove_cvref_t<_Result>, _Result>, "V must not be a reference or cv-qualified type");
  static_assert(__is_vectorizable_v<_Tp> && __is_abi_tag_v<typename _Result::abi_type>,
                "V must be an enabled specialization of basic_vec");
  static_assert(__is_vectorizable_v<_Up>, "range_value_t<R> must be a vectorizable type");
  static_assert(__explicitly_convertible_to<_Tp, _Up>,
                "range_value_t<R> must satisfy explicitly-convertible-to<value_type>");
  static_assert(__has_convert_flag_v<_Flags...> || __is_value_preserving_v<_Up, _Tp>,
                "Conversion from range_value_t<R> to value_type is not value-preserving; use flag_convert");
  ::cuda::std::simd::__assert_load_store_alignment<_Result, _Up, _Flags...>(__ptr);
}

// [simd.loadstore] helper: core partial load from pointer + count + mask
template <typename _Result, typename _Up, typename... _Flags>
[[nodiscard]] _CCCL_API constexpr _Result
__partial_load_from_ptr(const _Up* __ptr, __simd_size_type __count, const typename _Result::mask_type& __mask)
{
  using _Tp = typename _Result::value_type;
  ::cuda::std::simd::__check_load_preconditions<_Result, _Up, _Flags...>(__ptr);
  constexpr auto __simd_size = _Result::size();

  _Result __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
  {
    const auto __value = (__mask[__i] && __i < __count) ? static_cast<_Tp>(__ptr[__i]) : _Tp{};
    __result.__set(__i, __value);
  }
  return __result;
}

template <typename _Result, typename _Up, typename... _Flags>
[[nodiscard]] _CCCL_API constexpr _Result
__full_load_from_ptr(const _Up* __ptr, const typename _Result::mask_type& __mask)
{
  using _Tp = typename _Result::value_type;
  ::cuda::std::simd::__check_load_preconditions<_Result, _Up, _Flags...>(__ptr);

  if constexpr (__has_aligned_flag_v<_Flags...> || __has_overaligned_flag_v<_Flags...>)
  {
    _CCCL_IF_NOT_CONSTEVAL
    {
      constexpr auto __base_alignment = alignment_v<_Result, _Up>; // minimum condition for pointer alignment
      constexpr auto __ptr_alignment  = ::cuda::std::max(__base_alignment, __overaligned_value_v<_Flags...>);
      constexpr auto __simd_size      = _Result::size();
      constexpr auto __data_size      = __simd_size * sizeof(_Up);

      // When _CCCL_IF_NOT_CONSTEVAL falls back to __builtin_is_constant_evaluated(),
      // EDG requires locals to be initialized for the function to remain constexpr
#if __cpp_if_consteval >= 202106L || _CCCL_HAS_IF_CONSTEVAL_IN_CXX20()
      _Up __tmp[__simd_size];
#else
      _Up __tmp[__simd_size] = {};
#endif
      // vectorized load from pointer
      if constexpr (__is_cuda_vectoriazable_v<_Up> && __ptr_alignment >= __data_size)
      {
        struct alignas(__data_size) __aligned_t
        {
          char __data[__data_size];
        };
        // nvcc performance bug: memcpy from pointer could not be vectorized
        const auto __aligned_ptr = ::cuda::ptr_rebind<__aligned_t>(__ptr);
        const auto __data        = *::cuda::std::assume_aligned<__ptr_alignment>(__aligned_ptr);
        ::cuda::std::memcpy(&__tmp, &__data, sizeof(__tmp));
      }
      // rely on compiler vectorization
      else
      {
        const auto __aligned_ptr = ::cuda::std::assume_aligned<__ptr_alignment>(__ptr);
        _CCCL_PRAGMA_UNROLL_FULL()
        for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
        {
          __tmp[__i] = __aligned_ptr[__i];
        }
      }
      _Result __result;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
      {
        const auto __value = (!__mask[__i]) ? _Tp{} : static_cast<_Tp>(__tmp[__i]);
        __result.__set(__i, __value);
      }
      return __result;
    }
  }
  return ::cuda::std::simd::__partial_load_from_ptr<_Result, _Up, _Flags...>(__ptr, _Result::size, __mask);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] partial_load

// partial_load: range, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ranges::range_value_t<_Range>> partial_load(
  _Range&& __r,
  const typename __load_vec_t<_Vp, ranges::range_value_t<_Range>>::mask_type& __mask,
  flags<_Flags...> = {})
{
  using __result_t = __load_vec_t<_Vp, ranges::range_value_t<_Range>>;
  using __input_t  = ranges::range_value_t<_Range>;
  return ::cuda::std::simd::__partial_load_from_ptr<__result_t, __input_t, _Flags...>(
    ::cuda::std::ranges::data(__r), static_cast<__simd_size_type>(::cuda::std::ranges::size(__r)), __mask);
}

// partial_load: range, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ranges::range_value_t<_Range>>
partial_load(_Range&& __r, flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, ranges::range_value_t<_Range>>;
  return ::cuda::std::simd::partial_load<_Vp>(
    ::cuda::std::forward<_Range>(__r), typename __result_t::mask_type(true), __f);
}

// partial_load: iterator + count, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>> partial_load(
  _Ip __first,
  iter_difference_t<_Ip> __n,
  const typename __load_vec_t<_Vp, iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  using __input_t  = iter_value_t<_Ip>;
  return ::cuda::std::simd::__partial_load_from_ptr<__result_t, __input_t, _Flags...>(
    ::cuda::std::to_address(__first), static_cast<__simd_size_type>(__n), __mask);
}

// partial_load: iterator + count, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>>
partial_load(_Ip __first, iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  return ::cuda::std::simd::partial_load<_Vp>(__first, __n, typename __result_t::mask_type(true), __f);
}

// partial_load: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>> partial_load(
  _Ip __first, _Sp __last, const typename __load_vec_t<_Vp, iter_value_t<_Ip>>::mask_type& __mask, flags<_Flags...> = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  using __input_t  = iter_value_t<_Ip>;
  return ::cuda::std::simd::__partial_load_from_ptr<__result_t, __input_t, _Flags...>(
    ::cuda::std::to_address(__first), static_cast<__simd_size_type>(::cuda::std::distance(__first, __last)), __mask);
}

// partial_load: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>>
partial_load(_Ip __first, _Sp __last, flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  return ::cuda::std::simd::partial_load<_Vp>(__first, __last, typename __result_t::mask_type(true), __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] unchecked_load

// unchecked_load: range, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ranges::range_value_t<_Range>> unchecked_load(
  _Range&& __r,
  const typename __load_vec_t<_Vp, ranges::range_value_t<_Range>>::mask_type& __mask,
  flags<_Flags...> = {})
{
  using __result_t = __load_vec_t<_Vp, ranges::range_value_t<_Range>>;
  using __input_t  = ranges::range_value_t<_Range>;
  if constexpr (__has_static_size<_Range>)
  {
    static_assert(__static_range_size_v<_Range> >= __result_t::size(),
                  "unchecked_load requires ranges::size(r) >= V::size()");
  }
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::ranges::size(__r), __result_t::size()),
               "unchecked_load requires ranges::size(r) >= V::size()");
  return ::cuda::std::simd::__full_load_from_ptr<__result_t, __input_t, _Flags...>(
    ::cuda::std::ranges::data(__r), __mask);
}

// unchecked_load: range, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(ranges::contiguous_range<_Range> _CCCL_AND ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, ranges::range_value_t<_Range>>
unchecked_load(_Range&& __r, flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, ranges::range_value_t<_Range>>;
  return ::cuda::std::simd::unchecked_load<_Vp>(
    ::cuda::std::forward<_Range>(__r), typename __result_t::mask_type(true), __f);
}

// unchecked_load: iterator + count, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>> unchecked_load(
  _Ip __first,
  iter_difference_t<_Ip> __n,
  const typename __load_vec_t<_Vp, iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  using __input_t  = iter_value_t<_Ip>;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__n, __result_t::size()), "unchecked_load requires n >= V::size()");
  return ::cuda::std::simd::__full_load_from_ptr<__result_t, __input_t, _Flags...>(
    ::cuda::std::to_address(__first), __mask);
}

// unchecked_load: iterator + count, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>>
unchecked_load(_Ip __first, iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  return ::cuda::std::simd::unchecked_load<_Vp>(__first, __n, typename __result_t::mask_type(true), __f);
}

// unchecked_load: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>> unchecked_load(
  _Ip __first, _Sp __last, const typename __load_vec_t<_Vp, iter_value_t<_Ip>>::mask_type& __mask, flags<_Flags...> = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  using __input_t  = iter_value_t<_Ip>;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::distance(__first, __last), __result_t::size()),
               "unchecked_load requires distance(first, last) >= V::size()");
  return ::cuda::std::simd::__full_load_from_ptr<__result_t, __input_t, _Flags...>(
    ::cuda::std::to_address(__first), __mask);
}

// unchecked_load: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>>
unchecked_load(_Ip __first, _Sp __last, flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  return ::cuda::std::simd::unchecked_load<_Vp>(__first, __last, typename __result_t::mask_type(true), __f);
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_LOAD_H
