//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_EXPOSITION_H
#define _CUDAX___SIMD_EXPOSITION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/make_nbit_int.h>

#include <cuda/experimental/__simd/abi.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::simd
{
// [simd.expos], exposition-only helpers

template <::cuda::std::size_t _Bytes>
using __integer_from = ::cuda::std::__make_nbit_int_t<_Bytes * 8, true>;

template <typename _Tp>
constexpr bool __is_vectorizable_v =
  ::cuda::std::is_arithmetic_v<_Tp> && !::cuda::std::is_const_v<_Tp> && !::cuda::std::is_volatile_v<_Tp>
  && !::cuda::std::is_same_v<_Tp, bool>;

template <typename _Tp, typename _Abi>
constexpr __simd_size_type __simd_size_v = 0;

template <typename _Tp, __simd_size_type _Np>
constexpr __simd_size_type __simd_size_v<_Tp, simd_abi::fixed_size<_Np>> = _Np;
} // namespace cuda::experimental::simd

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_EXPOSITION_H
