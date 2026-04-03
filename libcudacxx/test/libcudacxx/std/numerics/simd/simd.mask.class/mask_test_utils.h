//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef SIMD_MASK_TEST_UTILS_H
#define SIMD_MASK_TEST_UTILS_H

#include <cuda/std/__simd_>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

struct is_even
{
  template <typename I>
  __host__ __device__ constexpr bool operator()(I i) const noexcept
  {
    return i % 2 == 0;
  }
};

struct is_first_half
{
  template <typename I>
  __host__ __device__ constexpr bool operator()(I i) const noexcept
  {
    return i < 2;
  }
};

struct wrong_generator
{};

template <typename>
struct is_const_member_function : cuda::std::false_type
{};

template <typename R, typename C, typename... Args>
struct is_const_member_function<R (C::*)(Args...) const> : cuda::std::true_type
{};

template <typename R, typename C, typename... Args>
struct is_const_member_function<R (C::*)(Args...) const noexcept> : cuda::std::true_type
{};

template <typename T>
inline constexpr bool is_const_member_function_v = is_const_member_function<T>::value;

template <int Bytes>
struct integer_from;
template <>
struct integer_from<1>
{
  using type = cuda::std::int8_t;
};
template <>
struct integer_from<2>
{
  using type = cuda::std::int16_t;
};
template <>
struct integer_from<4>
{
  using type = cuda::std::int32_t;
};
template <>
struct integer_from<8>
{
  using type = cuda::std::int64_t;
};

#if _CCCL_HAS_INT128()
template <>
struct integer_from<16>
{
  using type = __int128_t;
};
#endif // _CCCL_HAS_INT128()

template <int Bytes>
using integer_from_t = typename integer_from<Bytes>::type;

#endif // SIMD_MASK_TEST_UTILS_H
