//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_CREATION_H
#define _CUDA_STD___SIMD_CREATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__simd/basic_mask.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/exposition.h>
#include <cuda/std/__simd/type_traits.h>
#include <cuda/std/__simd/utility.h>
#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------
// Local traits: "enabled specialization of basic_vec/basic_mask" and "mask-element-size"
//
// The C++26 SIMD draft refers to "an enabled specialization of basic_vec/basic_mask". Mirror that with file-local
// traits until the public simd-vec-type / simd-mask-type concepts are introduced.

// TODO(fbusato): remove duplications across other PRs, move to a common place for basic_vec.h and basic_mask.h
template <typename _Tp>
inline constexpr bool __is_enabled_basic_vec_v = false;

template <typename _Tp, typename _Abi>
inline constexpr bool __is_enabled_basic_vec_v<basic_vec<_Tp, _Abi>> =
  __is_vectorizable_v<_Tp> && __is_enabled_abi_v<_Abi>;

template <typename _Tp>
inline constexpr bool __is_enabled_basic_mask_v = false;

template <size_t _Bytes, typename _Abi>
inline constexpr bool __is_enabled_basic_mask_v<basic_mask<_Bytes, _Abi>> =
  __is_vectorizable_byte_size_v<_Bytes> && __is_enabled_abi_v<_Abi>;

// get the element size of a basic_mask
// TODO(fbusato): remove if duplicated in other PRs
template <typename _Tp>
inline constexpr size_t __mask_element_size_v = 0;

template <size_t _Bytes, typename _Abi>
inline constexpr size_t __mask_element_size_v<basic_mask<_Bytes, _Abi>> = _Bytes;

// Shorthand for the integer_sequence<__simd_size_type, N> / make_integer_sequence<__simd_size_type, N>
template <__simd_size_type... _Ns>
using __simd_size_seq = integer_sequence<__simd_size_type, _Ns...>;

template <__simd_size_type _Np>
using __make_simd_size_seq = make_integer_sequence<__simd_size_type, _Np>;

// "If basic_vec<​typename T​::​​value_type, Abi>​::​size() % T​::​size() is not 0, then
// resize_t<basic_vec<​typename T​::​​value_type, Abi>​::​size() % T​::​size(), T> is valid and denotes
// a type."
//
// Vector: resize_t<V> is valid if __deduce_abi_t<V> is a specialized ABI type
// Mask: resize_t<M> is valid if __deduce_abi_t<__integer_from<M>> is a specialized ABI type
template <typename _Tp,
          typename _Abi,
          typename _ValueType = typename _Tp::value_type,
          size_t _Rem         = (basic_vec<_ValueType, _Abi>::__size % _Tp::__size)>
inline constexpr bool __chunk_vec_tail_ok_v = _Rem == 0 || __is_enabled_abi_v<__deduce_abi_t<_ValueType, _Rem>>;

template <typename _Tp,
          typename _Abi,
          size_t _ElemSize = __mask_element_size_v<_Tp>,
          size_t _Rem      = (basic_mask<_ElemSize, _Abi>::__size % _Tp::__size)>
inline constexpr bool __chunk_mask_tail_ok_v =
  _Rem == 0 || __is_enabled_abi_v<__deduce_abi_t<__integer_from<_ElemSize>, _Rem>>;

//----------------------------------------------------------------------------------------------------------------------
// [simd.creation], chunk building blocks

// extract _Src[Offset + {0, 1, ..., M}]
template <typename _Src, __simd_size_type _Offset>
struct __chunk_generator
{
  const _Src& __src;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Idx) const noexcept
  {
    return __src[_Offset + _Idx::value];
  }
};

// wrapper for __chunk_generator
template <typename _SubVec, __simd_size_type _Offset, typename _Src>
[[nodiscard]] _CCCL_API constexpr _SubVec __make_chunk(const _Src& __src) noexcept
{
  return _SubVec{__chunk_generator<_Src, _Offset>{__src}};
}

// Exact divisor case: return array<_SubVec, N>
template <typename _SubVec, typename _Src, __simd_size_type... _Js>
[[nodiscard]] _CCCL_API constexpr auto __make_chunk_array(const _Src& __src, __simd_size_seq<_Js...>) noexcept
{
  using __result_t = ::cuda::std::array<_SubVec, sizeof...(_Js)>;
  return __result_t{::cuda::std::simd::__make_chunk<_SubVec, _Js * _SubVec::__size>(__src)...};
}

// always returns T. Used for tuple<T, T, T, ...> below
template <typename _Tp, __simd_size_type>
using __repeat_t = _Tp;

// Remainder case: return tuple<_SubVec, ..., _SubVec, _Tail>
// where _Tail is resize_t<src.size() % _SubVec​::​size(), _SubVec​>
template <typename _SubVec, typename _Tail, typename _Src, __simd_size_type... _Js>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::tuple<__repeat_t<_SubVec, _Js>..., _Tail>
__make_chunk_tuple(const _Src& __src, __simd_size_seq<_Js...>) noexcept
{
  constexpr __simd_size_type __tail_offet = sizeof...(_Js) * _SubVec::__size;

  return ::cuda::std::tuple<__repeat_t<_SubVec, _Js>..., _Tail>{
    ::cuda::std::simd::__make_chunk<_SubVec, _Js * _SubVec::__size>(__src)..., // N copies of _SubVec (same as above)
    ::cuda::std::simd::__make_chunk<_Tail, __tail_offet>(__src)}; // fill Tail with N % M elements at the end of Src
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.creation] chunk
// split a SIMD vector of size N into a sequence of N/M sub-vectors of size M

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_enabled_basic_vec_v<_Tp> _CCCL_AND(__chunk_vec_tail_ok_v<_Tp, _Abi>))
[[nodiscard]] _CCCL_API constexpr auto chunk(const basic_vec<typename _Tp::value_type, _Abi>& __src) noexcept
{
  using __src_t                      = basic_vec<typename _Tp::value_type, _Abi>;
  constexpr __simd_size_type __nhead = __src_t::__size / _Tp::__size;
  constexpr __simd_size_type __rem   = __src_t::__size % _Tp::__size;
  if constexpr (__rem == 0) // exact divisor case
  {
    return ::cuda::std::simd::__make_chunk_array<_Tp>(__src, __make_simd_size_seq<__nhead>{});
  }
  else // remainder case
  {
    using __tail_t = resize_t<__rem, _Tp>;
    return ::cuda::std::simd::__make_chunk_tuple<_Tp, __tail_t>(__src, __make_simd_size_seq<__nhead>{});
  }
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_enabled_basic_mask_v<_Tp> _CCCL_AND(__chunk_mask_tail_ok_v<_Tp, _Abi>))
[[nodiscard]] _CCCL_API constexpr auto chunk(const basic_mask<__mask_element_size_v<_Tp>, _Abi>& __src) noexcept
{
  using __src_t                      = basic_mask<__mask_element_size_v<_Tp>, _Abi>;
  constexpr __simd_size_type __nhead = __src_t::__size / _Tp::__size;
  constexpr __simd_size_type __rem   = __src_t::__size % _Tp::__size;
  if constexpr (__rem == 0)
  {
    return ::cuda::std::simd::__make_chunk_array<_Tp>(__src, __make_simd_size_seq<__nhead>{});
  }
  else
  {
    using __tail_t = resize_t<__rem, _Tp>;
    return ::cuda::std::simd::__make_chunk_tuple<_Tp, __tail_t>(__src, __make_simd_size_seq<__nhead>{});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.creation], chunk<M>, with the size of the sub-vector M user-specified

template <__simd_size_type _Mp, typename _Up, typename _Abi>
[[nodiscard]] _CCCL_API constexpr auto chunk(const basic_vec<_Up, _Abi>& __src) noexcept
{
  static_assert(_Mp > 0, "_Mp must be greater than 0"); // avoid division by zero
  using __sub_vec_t = resize_t<_Mp, basic_vec<_Up, _Abi>>;
  return ::cuda::std::simd::chunk<__sub_vec_t, _Abi>(__src);
}

template <__simd_size_type _Mp, size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr auto chunk(const basic_mask<_Bytes, _Abi>& __src) noexcept
{
  static_assert(_Mp > 0, "_Mp must be greater than 0"); // avoid division by zero
  using __sub_vec_t = resize_t<_Mp, basic_mask<_Bytes, _Abi>>;
  return ::cuda::std::simd::chunk<__sub_vec_t, _Abi>(__src);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.creation], cat
// concatenate a sequence of SIMD vectors/masks
//
// cat(xs...) returns a data-parallel object whose i-th element is the i-th element of the concatenation of the xs
// pack. The implementation builds a generator that, given a global index _Ip, computes the target arg index and local
// offset at compile time via a comma-fold over the arg index pack and returns the corresponding element.

// Find the target arg index _K for a global index _Ip given the per-arg sizes {_Sizes...}. Scans left-to-right and
// picks the first bucket whose cumulative prefix sum covers _Ip. Returns sizeof...(_Sizes) on out-of-range
// (unreachable for well-formed calls). Always called from cat, which guarantees sizeof...(_Sizes) >= 1.
template <__simd_size_type _Ip, __simd_size_type... _Sizes>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL size_t __cat_arg_index(__simd_size_seq<_Sizes...>) noexcept
{
  const __simd_size_type __sizes[] = {_Sizes...};
  __simd_size_type __prefix        = 0;
  for (size_t __k = 0; __k < sizeof...(_Sizes); ++__k)
  {
    if (_Ip < __prefix + __sizes[__k])
    {
      return __k;
    }
    __prefix += __sizes[__k];
  }
  return sizeof...(_Sizes);
}

// Compute the local prefix sum (number of elements before arg _Kp) for a given target arg index _Kp.
template <size_t _Kp, __simd_size_type... _Sizes>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __simd_size_type __cat_local_prefix(__simd_size_seq<_Sizes...>) noexcept
{
  __simd_size_type __prefix = 0;
  // if constexpr guard is needed because _Kp may be 0, in which case __k < _Kp is a pointless comparison of
  // unsigned with zero (nvcc treats that warning as an error).
  if constexpr (_Kp > 0)
  {
    const __simd_size_type __sizes[] = {_Sizes...};
    for (size_t __k = 0; __k < _Kp; ++__k)
    {
      __prefix += __sizes[__k];
    }
  }
  return __prefix;
}

template <typename... _Vs>
struct __cat_generator
{
  ::cuda::std::tuple<const _Vs&...> __args_;

  template <typename _Ic>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Ic) const noexcept
  {
    constexpr __simd_size_type __i      = _Ic::value;
    constexpr size_t __k                = ::cuda::std::simd::__cat_arg_index<__i>(__simd_size_seq<_Vs::__size...>{});
    constexpr __simd_size_type __prefix = ::cuda::std::simd::__cat_local_prefix<__k>(__simd_size_seq<_Vs::__size...>{});
    return ::cuda::std::get<__k>(__args_)[__i - __prefix];
  }
};

template <typename... _Vs>
[[nodiscard]] _CCCL_API constexpr __cat_generator<_Vs...> __make_cat_generator(const _Vs&... __xs) noexcept
{
  return __cat_generator<_Vs...>{::cuda::std::tuple<const _Vs&...>{__xs...}};
}

// [simd.creation]/6 has no explicit _Constraints_ clause; the single requirement below is a QoI safety net that turns
// "total concatenated size exceeds the largest enabled ABI" from a hard error inside `resize_t` into SFINAE.

_CCCL_TEMPLATE(typename _Tp, typename _Abi0, typename... _Abis)
_CCCL_REQUIRES(__is_enabled_abi_v<__deduce_abi_t<
               _Tp,
               basic_vec<_Tp, _Abi0>::__size + (__simd_size_type{0} + ... + basic_vec<_Tp, _Abis>::__size)>>)
[[nodiscard]] _CCCL_API constexpr auto
cat(const basic_vec<_Tp, _Abi0>& __x0, const basic_vec<_Tp, _Abis>&... __xs) noexcept
{
  constexpr __simd_size_type __total =
    basic_vec<_Tp, _Abi0>::__size + (__simd_size_type{0} + ... + basic_vec<_Tp, _Abis>::__size);
  using __result_t = resize_t<__total, basic_vec<_Tp, _Abi0>>;
  return __result_t{::cuda::std::simd::__make_cat_generator(__x0, __xs...)};
}

_CCCL_TEMPLATE(size_t _Bytes, typename _Abi0, typename... _Abis)
_CCCL_REQUIRES(__is_enabled_abi_v<__deduce_abi_t<
               __integer_from<_Bytes>,
               basic_mask<_Bytes, _Abi0>::__size + (__simd_size_type{0} + ... + basic_mask<_Bytes, _Abis>::__size)>>)
[[nodiscard]] _CCCL_API constexpr auto
cat(const basic_mask<_Bytes, _Abi0>& __x0, const basic_mask<_Bytes, _Abis>&... __xs) noexcept
{
  constexpr __simd_size_type __total =
    basic_mask<_Bytes, _Abi0>::__size + (__simd_size_type{0} + ... + basic_mask<_Bytes, _Abis>::__size);
  using __result_t = resize_t<__total, basic_mask<_Bytes, _Abi0>>;
  return __result_t{::cuda::std::simd::__make_cat_generator(__x0, __xs...)};
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_CREATION_H
