//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_EXPOSITION_H
#define _CUDA_STD___SIMD_EXPOSITION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__type_traits/make_nbit_int.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.expos], exposition-only helpers

template <size_t _Bytes>
using __integer_from = __make_nbit_int_t<_Bytes * 8, true>;

// [simd.expos], simd-complex-value-type
template <typename _Tp>
using __simd_complex_value_type_t = typename _Tp::value_type::value_type;

template <typename _Tp, typename _Abi>
inline constexpr __simd_size_type __simd_size_v = 0;

template <typename _Tp, __simd_size_type _Np, bool = __is_vectorizable_v<_Tp>>
inline constexpr __simd_size_type __fixed_size_simd_size_v = 0;

template <typename _Tp, __simd_size_type _Np>
inline constexpr __simd_size_type __fixed_size_simd_size_v<_Tp, _Np, true> =
  __is_vectorizable_byte_size_v<sizeof(_Tp)> && __is_enabled_abi_v<fixed_size<_Np>> ? _Np : 0;

template <typename _Tp, __simd_size_type _Np>
inline constexpr __simd_size_type __simd_size_v<_Tp, fixed_size<_Np>> = __fixed_size_simd_size_v<_Tp, _Np>;

template <size_t _Bytes, typename _Abi>
inline constexpr __simd_size_type __mask_size_v = 0;

template <size_t _Bytes, __simd_size_type _Np>
inline constexpr __simd_size_type __mask_size_v<_Bytes, fixed_size<_Np>> =
  __is_vectorizable_byte_size_v<_Bytes> && __is_enabled_abi_v<fixed_size<_Np>> ? _Np : 0;

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_EXPOSITION_H
