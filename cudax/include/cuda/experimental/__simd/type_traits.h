//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_TYPE_TRAITS_H
#define _CUDAX___SIMD_TYPE_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/experimental/__simd/abi.h>
#include <cuda/experimental/__simd/declaration.h>
#include <cuda/experimental/__simd/exposition.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::simd
{
// [simd.traits], alignment
template <typename _Tp, typename _Up = typename _Tp::value_type>
struct alignment;

template <typename _Tp, typename _Abi, typename _Up>
struct alignment<basic_vec<_Tp, _Abi>, _Up>
    : ::cuda::std::integral_constant<::cuda::std::size_t, alignof(_Up) * __simd_size_v<_Tp, _Abi>>
{
  static_assert(__is_vectorizable_v<_Up>, "U must be a vectorizable type");
};

template <typename _Tp, typename _Up = typename _Tp::value_type>
constexpr ::cuda::std::size_t alignment_v = alignment<_Tp, _Up>::value;

// [simd.traits], rebind
template <typename _Tp, typename _Vp>
struct rebind;

template <typename _Tp, typename _Up, typename _Abi>
struct rebind<_Tp, basic_vec<_Up, _Abi>>
{
  static_assert(__is_vectorizable_v<_Tp>, "T must be a vectorizable type");
  using type = basic_vec<_Tp, simd_abi::__deduce_abi_t<_Tp, __simd_size_v<_Up, _Abi>>>;
};

template <typename _Tp, ::cuda::std::size_t _Bytes, typename _Abi>
struct rebind<_Tp, basic_mask<_Bytes, _Abi>>
{
  static_assert(__is_vectorizable_v<_Tp>, "T must be a vectorizable type");
  using __integer_t       = __integer_from<sizeof(_Tp)>;
  using __integer_bytes_t = __integer_from<_Bytes>;

  using type = basic_mask<sizeof(_Tp), simd_abi::__deduce_abi_t<__integer_t, __simd_size_v<__integer_bytes_t, _Abi>>>;
};

template <typename _Tp, typename _Vp>
using rebind_t = typename rebind<_Tp, _Vp>::type;

// [simd.traits], resize
template <__simd_size_type _Np, typename _Vp>
struct resize;

template <__simd_size_type _Np, typename _Tp, typename _Abi>
struct resize<_Np, basic_vec<_Tp, _Abi>>
{
  using type = basic_vec<_Tp, simd_abi::__deduce_abi_t<_Tp, _Np>>;
};

template <__simd_size_type _Np, ::cuda::std::size_t _Bytes, typename _Abi>
struct resize<_Np, basic_mask<_Bytes, _Abi>>
{
  using type = basic_mask<_Bytes, simd_abi::__deduce_abi_t<__integer_from<_Bytes>, _Np>>;
};

template <__simd_size_type _Np, typename _Vp>
using resize_t = typename resize<_Np, _Vp>::type;
} // namespace cuda::experimental::simd

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_TYPE_TRAITS_H
