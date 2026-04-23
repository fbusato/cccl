//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_OPERATIONS_TRAITS_H
#define _CUDA_STD___FUNCTIONAL_OPERATIONS_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/is_same.h>

// TODO(fbusato): move to _CCCL_HOSTED()
#if !_CCCL_COMPILER(NVRTC)
#  include <functional>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <typename _BinaryOp>
inline constexpr bool __is_plus_op_v =
  is_same_v<_BinaryOp, plus<>>
#if !_CCCL_COMPILER(NVRTC)
  || is_same_v<_BinaryOp, ::std::plus<>>
#endif // !_CCCL_COMPILER(NVRTC)
  ;

template <typename _BinaryOp>
inline constexpr bool __is_multiplies_op_v =
  is_same_v<_BinaryOp, multiplies<>>
#if !_CCCL_COMPILER(NVRTC)
  || is_same_v<_BinaryOp, ::std::multiplies<>>
#endif // !_CCCL_COMPILER(NVRTC)
  ;

template <typename _BinaryOp>
inline constexpr bool __is_bit_and_op_v =
  is_same_v<_BinaryOp, bit_and<>>
#if !_CCCL_COMPILER(NVRTC)
  || is_same_v<_BinaryOp, ::std::bit_and<>>
#endif // !_CCCL_COMPILER(NVRTC)
  ;

template <typename _BinaryOp>
inline constexpr bool __is_bit_or_op_v =
  is_same_v<_BinaryOp, bit_or<>>
#if !_CCCL_COMPILER(NVRTC)
  || is_same_v<_BinaryOp, ::std::bit_or<>>
#endif // !_CCCL_COMPILER(NVRTC)
  ;

template <typename _BinaryOp>
inline constexpr bool __is_bit_xor_op_v =
  is_same_v<_BinaryOp, bit_xor<>>
#if !_CCCL_COMPILER(NVRTC)
  || is_same_v<_BinaryOp, ::std::bit_xor<>>
#endif // !_CCCL_COMPILER(NVRTC)
  ;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_OPERATIONS_TRAITS_H
