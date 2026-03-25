//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_DECLARATION_H
#define _CUDAX___SIMD_DECLARATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>

#include <cuda/experimental/__simd/abi.h>
#include <cuda/experimental/__simd/exposition.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::simd
{


template <typename _Tp, typename _Abi = simd_abi::native<_Tp>>
class basic_vec;

template <::cuda::std::size_t _Bytes, typename _Abi = simd_abi::native<__integer_from<_Bytes>>>
class basic_mask;

template <typename _Tp, __simd_size_type _Np = __simd_size_v<_Tp, simd_abi::native<_Tp>>>
using vec = basic_vec<_Tp, simd_abi::__deduce_abi_t<_Tp, _Np>>;

template <typename _Tp, __simd_size_type _Np = __simd_size_v<_Tp, simd_abi::native<_Tp>>>
using mask = basic_mask<sizeof(_Tp), simd_abi::__deduce_abi_t<_Tp, _Np>>;

// specializations

template <::cuda::std::size_t _Bytes, typename _Abi>
struct __mask_storage;

template <::cuda::std::size_t _Bytes, typename _Abi>
struct __mask_operations;
} // namespace cuda::experimental::simd

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_DECLARATION_H
