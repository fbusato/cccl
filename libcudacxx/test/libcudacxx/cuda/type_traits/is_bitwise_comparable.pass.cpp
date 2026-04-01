//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_bitwise_comparable()
{
  static_assert(cuda::is_bitwise_comparable<T>::value);
  static_assert(cuda::is_bitwise_comparable_v<T>);
  static_assert(cuda::is_bitwise_comparable_v<const T>);
  static_assert(cuda::is_bitwise_comparable_v<volatile T>);
  static_assert(cuda::is_bitwise_comparable_v<const volatile T>);
}

template <class T>
__host__ __device__ void test_is_not_bitwise_comparable()
{
  static_assert(!cuda::is_bitwise_comparable<T>::value);
  static_assert(!cuda::is_bitwise_comparable_v<T>);
  static_assert(!cuda::is_bitwise_comparable_v<const T>);
  static_assert(!cuda::is_bitwise_comparable_v<volatile T>);
  static_assert(!cuda::is_bitwise_comparable_v<const volatile T>);
}

struct WithPadding
{
  int x;
  char y;
};

struct UserSpecialization
{
  double value;
};

template <>
constexpr bool cuda::is_bitwise_comparable_v<UserSpecialization> = true;

__host__ __device__ void test()
{
  // types with unique object representations
  test_is_bitwise_comparable<int>();
  test_is_bitwise_comparable<unsigned>();
  test_is_bitwise_comparable<char>();
  test_is_bitwise_comparable<unsigned char>();
  test_is_bitwise_comparable<short>();
  test_is_bitwise_comparable<long long>();

  // arrays
  static_assert(cuda::is_bitwise_comparable_v<int[4]>);
  static_assert(cuda::is_bitwise_comparable_v<const int[4]>);
  static_assert(cuda::is_bitwise_comparable_v<unsigned char[8]>);

  // padding-free cuda::std::array, pair, tuple of bitwise comparable types
  static_assert(cuda::is_bitwise_comparable_v<cuda::std::array<int, 4>>);
  static_assert(cuda::is_bitwise_comparable_v<cuda::std::pair<int, unsigned>>);
  static_assert(cuda::is_bitwise_comparable_v<cuda::std::tuple<int, unsigned>>);
  static_assert(cuda::is_bitwise_comparable_v<cuda::std::tuple<>>);

  // composites with padding are not bitwise comparable
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::pair<int, char>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::tuple<int, unsigned, char>>);

  // types without unique object representations
  test_is_not_bitwise_comparable<float>();
  test_is_not_bitwise_comparable<double>();
  test_is_not_bitwise_comparable<WithPadding>();

  // extended floating-point scalar types
#if _CCCL_HAS_NVFP16()
  test_is_not_bitwise_comparable<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_is_not_bitwise_comparable<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_is_not_bitwise_comparable<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()

  // extended floating-point vector types
#if _CCCL_HAS_NVFP16()
  test_is_not_bitwise_comparable<__half2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_is_not_bitwise_comparable<__nv_bfloat162>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8()
  test_is_not_bitwise_comparable<__nv_fp8x2_e4m3>();
#endif // _CCCL_HAS_NVFP8()

  // compositions of extended floating-point types
#if _CCCL_HAS_NVFP16()
  static_assert(!cuda::is_bitwise_comparable_v<__half[4]>);
  static_assert(!cuda::is_bitwise_comparable_v<const __half[4]>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::array<__half, 4>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::pair<__half, int>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::tuple<__half, float>>);
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::array<__nv_bfloat16, 2>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::pair<__nv_bfloat16, int>>);
#endif // _CCCL_HAS_NVBF16()

  // nested compositions
#if _CCCL_HAS_NVFP16()
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::array<cuda::std::pair<__half, int>, 2>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::tuple<cuda::std::array<__half, 4>, int>>);
#endif // _CCCL_HAS_NVFP16()

  // user specialization of the variable template
  static_assert(cuda::is_bitwise_comparable_v<UserSpecialization>);
  static_assert(cuda::is_bitwise_comparable<UserSpecialization>::value);
}

int main(int, char**)
{
  test();
  return 0;
}
