//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_PERMUTE_H
#define _CUDA_STD___SIMD_PERMUTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__simd/exposition.h>
#include <cuda/std/__simd/type_traits.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.permute.static], sentinels for static permute

inline constexpr __simd_size_type zero_element   = -1;
inline constexpr __simd_size_type uninit_element = -2;

//----------------------------------------------------------------------------------------------------------------------
// [simd.permute.static], Constraints detection
//
// At least one of `invoke_result_t<IdxMap&, simd-size-type>` and `invoke_result_t<IdxMap&, simd-size-type,
// simd-size-type>` must satisfy `integral`.

template <typename _IdxMap, typename _Enable, typename... _Args>
inline constexpr bool __idxmap_nargs_integral_v = false;

template <typename _IdxMap, typename... _Args>
inline constexpr bool __idxmap_nargs_integral_v<_IdxMap, void_t<invoke_result_t<_IdxMap&, _Args...>>, _Args...> =
  is_integral_v<remove_cvref_t<invoke_result_t<_IdxMap&, _Args...>>>;

template <typename _IdxMap>
inline constexpr bool __idxmap_result_is_integral_v =
  __idxmap_nargs_integral_v<remove_cvref_t<_IdxMap>, void, __simd_size_type, __simd_size_type>
  || __idxmap_nargs_integral_v<remove_cvref_t<_IdxMap>, void, __simd_size_type>;

//----------------------------------------------------------------------------------------------------------------------
// gen-fn: idxmap(i, V​::​size()) if that expression is well-formed, and idxmap(i) otherwise.

template <typename _IdxMap, __simd_size_type _Idx, __simd_size_type _Size, typename = void>
inline constexpr bool __idxmap_invocable_two_args_v = false;

template <typename _IdxMap, __simd_size_type _Idx, __simd_size_type _Size>
inline constexpr bool __idxmap_invocable_two_args_v<
  _IdxMap,
  _Idx,
  _Size,
  void_t<decltype(_IdxMap{}(__simd_size_constant<_Idx>{}, __simd_size_constant<_Size>{}))>> = true;

template <typename _IdxMap, __simd_size_type _Idx, __simd_size_type _Size>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __simd_size_type __permute_gen_fn() noexcept
{
  if constexpr (__idxmap_invocable_two_args_v<_IdxMap, _Idx, _Size>)
  {
    return _IdxMap{}(__simd_size_constant<_Idx>{}, __simd_size_constant<_Size>{});
  }
  else
  {
    return _IdxMap{}(__simd_size_constant<_Idx>{});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// permute_generator

template <typename _IdxMap, typename _Vp>
struct __permute_generator
{
  using __value_type = typename _Vp::value_type;

  const _Vp& __v_;

  template <__simd_size_type _Idx>
  [[nodiscard]] _CCCL_API constexpr __value_type operator()(__simd_size_constant<_Idx>) const noexcept
  {
    using __map_t                     = remove_cvref_t<_IdxMap>;
    constexpr __simd_size_type __size = _Vp::size();
    constexpr __simd_size_type __src  = ::cuda::std::simd::__permute_gen_fn<__map_t, _Idx, __size>();
    static_assert(__src == zero_element || __src == uninit_element || (__src >= 0 && __src < __size),
                  "cuda::std::simd::permute: idxmap(i) must return zero_element, uninit_element, or a value in [0, "
                  "V::size())");
    if constexpr (__src == zero_element || __src == uninit_element)
    {
      return __value_type{}; // unspecified-value
    }
    else
    {
      return __v_[__src];
    }
  }
};

template <typename _IdxMap, typename _Vp>
[[nodiscard]] _CCCL_API constexpr __permute_generator<_IdxMap, _Vp> __make_permute_generator(const _Vp& __v) noexcept
{
  return __permute_generator<_IdxMap, _Vp>{__v};
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.permute.static]

// instead of two overloads, we use a default value = -1
inline constexpr __simd_size_type __permute_default_n = -1;

template <typename _Tp, typename _Abi, __simd_size_type _Np>
inline constexpr __simd_size_type __permute_vec_size_v = (_Np == __permute_default_n) ? __simd_size_v<_Tp, _Abi> : _Np;

template <size_t _Bytes, typename _Abi, __simd_size_type _Np>
inline constexpr __simd_size_type __permute_mask_size_v =
  (_Np == __permute_default_n) ? __simd_size_v<__integer_from<_Bytes>, _Abi> : _Np;

template <typename _Tp, typename _Abi, __simd_size_type _Np>
using __permute_result_vec_t = resize_t<__permute_vec_size_v<_Tp, _Abi, _Np>, basic_vec<_Tp, _Abi>>;

template <size_t _Bytes, typename _Abi, __simd_size_type _Np>
using __permute_result_mask_t = resize_t<__permute_mask_size_v<_Bytes, _Abi, _Np>, basic_mask<_Bytes, _Abi>>;

_CCCL_TEMPLATE(__simd_size_type _Np = __permute_default_n, typename _Tp, typename _Abi, typename _IdxMap)
_CCCL_REQUIRES(__idxmap_result_is_integral_v<_IdxMap>)
[[nodiscard]] _CCCL_API constexpr __permute_result_vec_t<_Tp, _Abi, _Np>
permute(const basic_vec<_Tp, _Abi>& __v, _IdxMap&&)
{
  static_assert(_Np == __permute_default_n || _Np >= 0, "cuda::std::simd::permute: N must be non-negative");
  using __result_t = __permute_result_vec_t<_Tp, _Abi, _Np>;
  return __result_t{::cuda::std::simd::__make_permute_generator<_IdxMap>(__v)};
}

_CCCL_TEMPLATE(__simd_size_type _Np = __permute_default_n, size_t _Bytes, typename _Abi, typename _IdxMap)
_CCCL_REQUIRES(__idxmap_result_is_integral_v<_IdxMap>)
[[nodiscard]] _CCCL_API constexpr __permute_result_mask_t<_Bytes, _Abi, _Np>
permute(const basic_mask<_Bytes, _Abi>& __v, _IdxMap&&)
{
  static_assert(_Np == __permute_default_n || _Np >= 0, "cuda::std::simd::permute: N must be non-negative");
  using __result_t = __permute_result_mask_t<_Bytes, _Abi, _Np>;
  return __result_t{::cuda::std::simd::__make_permute_generator<_IdxMap>(__v)};
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.permute.dynamic]

template <typename _Vp, typename _Ip>
struct __permute_dynamic_generator
{
  using __value_type = typename _Vp::value_type;

  const _Vp& __v_;
  const _Ip& __indices_;

  template <__simd_size_type _Idx>
  [[nodiscard]] _CCCL_API constexpr __value_type operator()(__simd_size_constant<_Idx>) const noexcept
  {
    const auto __src = static_cast<__simd_size_type>(__indices_[_Idx]);
    _CCCL_ASSERT(::cuda::in_range(__src, __simd_size_type{0}, _Vp::size()),
                 "cuda::std::simd::permute: indices[i] must be in [0, V::size())");
    return __v_[__src];
  }
};

template <typename _Vp, typename _Ip>
[[nodiscard]] _CCCL_API constexpr __permute_dynamic_generator<_Vp, _Ip>
__make_permute_dynamic_generator(const _Vp& __v, const _Ip& __indices) noexcept
{
  return __permute_dynamic_generator<_Vp, _Ip>{__v, __indices};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Up, typename _UAbi)
_CCCL_REQUIRES(is_integral_v<_Up>)
[[nodiscard]] _CCCL_API constexpr resize_t<__simd_size_v<_Up, _UAbi>, basic_vec<_Tp, _Abi>>
permute(const basic_vec<_Tp, _Abi>& __v, const basic_vec<_Up, _UAbi>& __indices)
{
  using __result_t = resize_t<__simd_size_v<_Up, _UAbi>, basic_vec<_Tp, _Abi>>;
  return __result_t{::cuda::std::simd::__make_permute_dynamic_generator(__v, __indices)};
}

_CCCL_TEMPLATE(size_t _Bytes, typename _Abi, typename _Up, typename _UAbi)
_CCCL_REQUIRES(is_integral_v<_Up>)
[[nodiscard]] _CCCL_API constexpr resize_t<__simd_size_v<_Up, _UAbi>, basic_mask<_Bytes, _Abi>>
permute(const basic_mask<_Bytes, _Abi>& __v, const basic_vec<_Up, _UAbi>& __indices)
{
  using __result_t = resize_t<__simd_size_v<_Up, _UAbi>, basic_mask<_Bytes, _Abi>>;
  return __result_t{::cuda::std::simd::__make_permute_dynamic_generator(__v, __indices)};
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_PERMUTE_H
