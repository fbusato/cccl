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
__host__ __device__ void test_is_trivially_copyable_relaxed()
{
  static_assert(cuda::is_trivially_copyable_relaxed<T>::value);
  static_assert(cuda::is_trivially_copyable_relaxed<const T>::value);
  static_assert(cuda::is_trivially_copyable_relaxed<volatile T>::value);
  static_assert(cuda::is_trivially_copyable_relaxed<const volatile T>::value);
  static_assert(cuda::is_trivially_copyable_relaxed_v<T>);
  static_assert(cuda::is_trivially_copyable_relaxed_v<const T>);
  static_assert(cuda::is_trivially_copyable_relaxed_v<volatile T>);
  static_assert(cuda::is_trivially_copyable_relaxed_v<const volatile T>);
}

template <class T>
__host__ __device__ void test_is_not_trivially_copyable_relaxed()
{
  static_assert(!cuda::is_trivially_copyable_relaxed<T>::value);
  static_assert(!cuda::is_trivially_copyable_relaxed<const T>::value);
  static_assert(!cuda::is_trivially_copyable_relaxed<volatile T>::value);
  static_assert(!cuda::is_trivially_copyable_relaxed<const volatile T>::value);
  static_assert(!cuda::is_trivially_copyable_relaxed_v<T>);
  static_assert(!cuda::is_trivially_copyable_relaxed_v<const T>);
  static_assert(!cuda::is_trivially_copyable_relaxed_v<volatile T>);
  static_assert(!cuda::is_trivially_copyable_relaxed_v<const volatile T>);
}

struct TrivialPod
{
  int x;
  float y;
};

class NonTriviallyCopyable
{
public:
  __host__ __device__ NonTriviallyCopyable(const NonTriviallyCopyable&) {}
};

__host__ __device__ void test()
{
  // standard trivially copyable types
  test_is_trivially_copyable_relaxed<int>();
  test_is_trivially_copyable_relaxed<float>();
  test_is_trivially_copyable_relaxed<double>();
  test_is_trivially_copyable_relaxed<TrivialPod>();

  // C-style arrays of trivially copyable types
  static_assert(cuda::is_trivially_copyable_relaxed_v<int[4]>);
  static_assert(cuda::is_trivially_copyable_relaxed_v<const int[4]>);
  static_assert(cuda::is_trivially_copyable_relaxed_v<TrivialPod[2]>);

  // cuda::std::array, pair, tuple of trivially copyable types
  test_is_trivially_copyable_relaxed<cuda::std::array<int, 4>>();
  test_is_trivially_copyable_relaxed<cuda::std::pair<int, float>>();
  test_is_trivially_copyable_relaxed<cuda::std::tuple<int, float, double>>();
  test_is_trivially_copyable_relaxed<cuda::std::tuple<>>();

  // extended floating point scalar types
#if _CCCL_HAS_NVFP16()
  test_is_trivially_copyable_relaxed<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_is_trivially_copyable_relaxed<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_is_trivially_copyable_relaxed<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()

  // extended floating point vector types
#if _CCCL_HAS_NVFP16()
  test_is_trivially_copyable_relaxed<__half2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_is_trivially_copyable_relaxed<__nv_bfloat162>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8()
  test_is_trivially_copyable_relaxed<__nv_fp8x2_e4m3>();
#endif // _CCCL_HAS_NVFP8()

  // compositions of extended floating point types
#if _CCCL_HAS_NVFP16()
  static_assert(cuda::is_trivially_copyable_relaxed_v<__half[4]>);
  static_assert(cuda::is_trivially_copyable_relaxed_v<const __half[4]>);
  test_is_trivially_copyable_relaxed<cuda::std::array<__half, 4>>();
  test_is_trivially_copyable_relaxed<cuda::std::pair<__half, int>>();
  test_is_trivially_copyable_relaxed<cuda::std::tuple<__half, float>>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_is_trivially_copyable_relaxed<cuda::std::array<__nv_bfloat16, 2>>();
  test_is_trivially_copyable_relaxed<cuda::std::pair<__nv_bfloat16, int>>();
#endif // _CCCL_HAS_NVBF16()

  // nested compositions
#if _CCCL_HAS_NVFP16()
  test_is_trivially_copyable_relaxed<cuda::std::array<cuda::std::pair<__half, int>, 2>>();
  test_is_trivially_copyable_relaxed<cuda::std::tuple<cuda::std::array<__half, 4>, int>>();
  test_is_trivially_copyable_relaxed<cuda::std::pair<cuda::std::tuple<__half, float>, double>>();
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVFP16() && _CCCL_HAS_NVBF16()
  test_is_trivially_copyable_relaxed<cuda::std::tuple<__half, __nv_bfloat16, float>>();
#endif // _CCCL_HAS_NVFP16() && _CCCL_HAS_NVBF16()

  // non-trivially copyable types
  test_is_not_trivially_copyable_relaxed<NonTriviallyCopyable>();
}

int main(int, char**)
{
  test();
  return 0;
}
