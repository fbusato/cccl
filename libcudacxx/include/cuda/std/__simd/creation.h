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
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------
// Local traits: "enabled specialization of basic_vec/basic_mask" and "mask-element-size"
//
// The C++26 SIMD draft refers to "an enabled specialization of basic_vec/basic_mask". Mirror that with file-local
// traits until the public simd-vec-type / simd-mask-type concepts are introduced.

// TODO(fbusato): remove if duplicated across other PRs
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

// mask-element-size<basic_mask<Bytes, Abi>> is Bytes. Primary template returns 0 for non-mask types.
template <typename _Tp>
inline constexpr size_t __mask_element_size_v = 0;

template <size_t _Bytes, typename _Abi>
inline constexpr size_t __mask_element_size_v<basic_mask<_Bytes, _Abi>> = _Bytes;

// Pack-expansion helper: yields _Tp regardless of the index.
template <typename _Tp, __simd_size_type>
using __repeat_t = _Tp;

// Shorthand for the `integer_sequence<__simd_size_type, ...>` / `make_integer_sequence<__simd_size_type, N>` pattern
// that appears throughout the chunk/cat helpers below.
template <__simd_size_type... _Ns>
using __simd_size_seq = integer_sequence<__simd_size_type, _Ns...>;

template <__simd_size_type _Np>
using __make_simd_size_seq = make_integer_sequence<__simd_size_type, _Np>;

// Tail-chunk ABI validity traits: given a source basic_vec<_Up, _Abi> / basic_mask<_SrcBytes, _Abi> and a chunk type
// _Tp, these hold iff either the chunk size divides the source size (no tail) or the remainder names an enabled ABI.
// Instantiated only after `__is_enabled_basic_{vec,mask}_v<_Tp>` has been checked earlier in the _CCCL_REQUIRES chain,
// so `_Tp::__size` is always well-formed here.
template <typename _Tp, typename _Abi, typename _Up>
inline constexpr bool __chunk_vec_tail_ok_v =
  basic_vec<_Up, _Abi>::__size % _Tp::__size == 0
  || __is_enabled_abi_v<__deduce_abi_t<_Up, basic_vec<_Up, _Abi>::__size % _Tp::__size>>;

template <typename _Tp, typename _Abi, size_t _SrcBytes>
inline constexpr bool __chunk_mask_tail_ok_v =
  basic_mask<_SrcBytes, _Abi>::__size % _Tp::__size == 0
  || __is_enabled_abi_v<__deduce_abi_t<__integer_from<_SrcBytes>, basic_mask<_SrcBytes, _Abi>::__size % _Tp::__size>>;

//----------------------------------------------------------------------------------------------------------------------
// [simd.creation], chunk building blocks
//
// The chunk generator exposes x[_Offset + i] to basic_vec's / basic_mask's generator constructor, producing one
// sub-chunk at offset _Offset.

template <typename _Src, __simd_size_type _Offset>
struct __chunk_generator
{
  const _Src& __src_;

  template <typename _Ic>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Ic) const noexcept
  {
    return __src_[_Offset + _Ic::value];
  }
};

template <typename _Sub, __simd_size_type _Offset, typename _Src>
[[nodiscard]] _CCCL_API constexpr _Sub __make_chunk(const _Src& __x) noexcept
{
  return _Sub{__chunk_generator<_Src, _Offset>{__x}};
}

// Exact-divisor case: return array<_Sub, N> filled by expanding over a compile-time index pack.
template <typename _Sub, typename _Src, __simd_size_type... _Js>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::array<_Sub, sizeof...(_Js)>
__make_chunk_array(const _Src& __x, __simd_size_seq<_Js...>) noexcept
{
  return ::cuda::std::array<_Sub, sizeof...(_Js)>{::cuda::std::simd::__make_chunk<_Sub, _Js * _Sub::__size>(__x)...};
}

// Remainder case: return tuple<_Sub, ..., _Sub, _Tail> (N copies of _Sub followed by one _Tail).
//
// Alternative valid shape: pair<array<_Sub, N>, _Tail>; chosen tuple<...> to match spec wording literally.
template <typename _Sub, typename _Tail, typename _Src, __simd_size_type... _Js>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::tuple<__repeat_t<_Sub, _Js>..., _Tail>
__make_chunk_tuple(const _Src& __x, __simd_size_seq<_Js...>) noexcept
{
  constexpr __simd_size_type __tail_off = static_cast<__simd_size_type>(sizeof...(_Js)) * _Sub::__size;
  return ::cuda::std::tuple<__repeat_t<_Sub, _Js>..., _Tail>{
    ::cuda::std::simd::__make_chunk<_Sub, _Js * _Sub::__size>(__x)...,
    ::cuda::std::simd::__make_chunk<_Tail, __tail_off>(__x)};
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.creation], chunk<T, Abi>(basic_vec) and chunk<T, Abi>(basic_mask)
//
// Overload signatures follow the spec's template<class T, class Abi> shape but deduce the source element type /
// byte-size from the argument (_Up for vec, _SrcBytes for mask) and require it to match T via the constraints, so
// substitution of T's members never happens in the signature itself (avoids hard errors when T is not basic_vec /
// basic_mask). Exact-divisor and remainder cases share one function: the return type is `auto` and the two branches
// are selected with `if constexpr` on the remainder.

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Up)
_CCCL_REQUIRES(__is_enabled_basic_vec_v<_Tp> _CCCL_AND(is_same_v<_Up, typename _Tp::value_type>)
                 _CCCL_AND(__chunk_vec_tail_ok_v<_Tp, _Abi, _Up>))
[[nodiscard]] _CCCL_API constexpr auto chunk(const basic_vec<_Up, _Abi>& __x) noexcept
{
  constexpr __simd_size_type __nhead = basic_vec<_Up, _Abi>::__size / _Tp::__size;
  constexpr __simd_size_type __rem   = basic_vec<_Up, _Abi>::__size % _Tp::__size;
  if constexpr (__rem == 0)
  {
    return ::cuda::std::simd::__make_chunk_array<_Tp>(__x, __make_simd_size_seq<__nhead>{});
  }
  else
  {
    using __tail_t = resize_t<__rem, _Tp>;
    return ::cuda::std::simd::__make_chunk_tuple<_Tp, __tail_t>(__x, __make_simd_size_seq<__nhead>{});
  }
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, size_t _SrcBytes)
_CCCL_REQUIRES(__is_enabled_basic_mask_v<_Tp> _CCCL_AND(__mask_element_size_v<_Tp> == _SrcBytes)
                 _CCCL_AND(__chunk_mask_tail_ok_v<_Tp, _Abi, _SrcBytes>))
[[nodiscard]] _CCCL_API constexpr auto chunk(const basic_mask<_SrcBytes, _Abi>& __x) noexcept
{
  constexpr __simd_size_type __nhead = basic_mask<_SrcBytes, _Abi>::__size / _Tp::__size;
  constexpr __simd_size_type __rem   = basic_mask<_SrcBytes, _Abi>::__size % _Tp::__size;
  if constexpr (__rem == 0)
  {
    return ::cuda::std::simd::__make_chunk_array<_Tp>(__x, __make_simd_size_seq<__nhead>{});
  }
  else
  {
    using __tail_t = resize_t<__rem, _Tp>;
    return ::cuda::std::simd::__make_chunk_tuple<_Tp, __tail_t>(__x, __make_simd_size_seq<__nhead>{});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.creation], chunk<N>(basic_vec) / chunk<N>(basic_mask) delegating overloads

_CCCL_TEMPLATE(__simd_size_type _Np, typename _Up, typename _Abi)
_CCCL_REQUIRES((_Np >= 1) _CCCL_AND(__is_enabled_abi_v<__deduce_abi_t<_Up, _Np>>))
[[nodiscard]] _CCCL_API constexpr auto chunk(const basic_vec<_Up, _Abi>& __x) noexcept
{
  using __sub_t = resize_t<_Np, basic_vec<_Up, _Abi>>;
  return ::cuda::std::simd::chunk<__sub_t, _Abi>(__x);
}

_CCCL_TEMPLATE(__simd_size_type _Np, size_t _Bytes, typename _Abi)
_CCCL_REQUIRES((_Np >= 1) _CCCL_AND(__is_enabled_abi_v<__deduce_abi_t<__integer_from<_Bytes>, _Np>>))
[[nodiscard]] _CCCL_API constexpr auto chunk(const basic_mask<_Bytes, _Abi>& __x) noexcept
{
  using __sub_t = resize_t<_Np, basic_mask<_Bytes, _Abi>>;
  return ::cuda::std::simd::chunk<__sub_t, _Abi>(__x);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.creation], cat
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
  // `if constexpr` guard is needed because _Kp may be 0, in which case `__k < _Kp` is a pointless comparison of
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

_CCCL_TEMPLATE(typename _Tp, typename _Abi0, typename... _AbiN)
_CCCL_REQUIRES((__is_enabled_basic_vec_v<basic_vec<_Tp, _Abi0>>) _CCCL_AND(
  (... && __is_enabled_basic_vec_v<basic_vec<_Tp, _AbiN>>) )
                 _CCCL_AND(__is_enabled_abi_v<__deduce_abi_t<
                             _Tp,
                             basic_vec<_Tp, _Abi0>::__size
                               + (__simd_size_type{0} + ... + basic_vec<_Tp, _AbiN>::__size)>>))
[[nodiscard]] _CCCL_API constexpr auto
cat(const basic_vec<_Tp, _Abi0>& __x0, const basic_vec<_Tp, _AbiN>&... __xs) noexcept
{
  constexpr __simd_size_type __total =
    basic_vec<_Tp, _Abi0>::__size + (__simd_size_type{0} + ... + basic_vec<_Tp, _AbiN>::__size);
  using __result_t = resize_t<__total, basic_vec<_Tp, _Abi0>>;
  return __result_t{::cuda::std::simd::__make_cat_generator(__x0, __xs...)};
}

_CCCL_TEMPLATE(size_t _Bytes, typename _Abi0, typename... _AbiN)
_CCCL_REQUIRES((__is_enabled_basic_mask_v<basic_mask<_Bytes, _Abi0>>) _CCCL_AND(
  (... && __is_enabled_basic_mask_v<basic_mask<_Bytes, _AbiN>>) )
                 _CCCL_AND(__is_enabled_abi_v<__deduce_abi_t<
                             __integer_from<_Bytes>,
                             basic_mask<_Bytes, _Abi0>::__size
                               + (__simd_size_type{0} + ... + basic_mask<_Bytes, _AbiN>::__size)>>))
[[nodiscard]] _CCCL_API constexpr auto
cat(const basic_mask<_Bytes, _Abi0>& __x0, const basic_mask<_Bytes, _AbiN>&... __xs) noexcept
{
  constexpr __simd_size_type __total =
    basic_mask<_Bytes, _Abi0>::__size + (__simd_size_type{0} + ... + basic_mask<_Bytes, _AbiN>::__size);
  using __result_t = resize_t<__total, basic_mask<_Bytes, _Abi0>>;
  return __result_t{::cuda::std::simd::__make_cat_generator(__x0, __xs...)};
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_CREATION_H
