//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstring>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/type_traits>

#include "test_macros.h"

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-local-typedef")

// operator== for CUDA vector types and dim3 (not provided by the toolkit)
template <class T, cuda::std::enable_if_t<cuda::is_vector_type_v<T>, int> = 0>
__host__ __device__ bool operator==(T a, T b)
{
  if constexpr (sizeof(T) == sizeof(decltype(T::x)))
  {
    return a.x == b.x;
  }
  else if constexpr (sizeof(T) == 2 * sizeof(decltype(T::x)))
  {
    return a.x == b.x && a.y == b.y;
  }
  else if constexpr (sizeof(T) == 3 * sizeof(decltype(T::x)))
  {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }
  else
  {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
  }
}

// Extended FP operator== is __device__ only on CTK < 12.3, __host__ __device__ from CTK >= 12.3.
// Provide __host__ __device__ fallbacks only when operators are explicitly disabled by the user.
#if _CCCL_HAS_NVFP16()
#  if defined(__CUDA_NO_HALF_OPERATORS__)
__host__ __device__ bool operator==(__half a, __half b)
{
  return __half2float(a) == __half2float(b);
}
#  endif // defined(__CUDA_NO_HALF_OPERATORS__)
#  if defined(__CUDA_NO_HALF2_OPERATORS__)
__host__ __device__ bool operator==(__half2 a, __half2 b)
{
  return __half2float(a.x) == __half2float(b.x) && __half2float(a.y) == __half2float(b.y);
}
#  endif // defined(__CUDA_NO_HALF2_OPERATORS__)
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
#  if defined(__CUDA_NO_BFLOAT16_OPERATORS__)
__host__ __device__ bool operator==(__nv_bfloat16 a, __nv_bfloat16 b)
{
  return __bfloat162float(a) == __bfloat162float(b);
}
#  endif // defined(__CUDA_NO_BFLOAT16_OPERATORS__)
#  if defined(__CUDA_NO_BFLOAT162_OPERATORS__)
__host__ __device__ bool operator==(__nv_bfloat162 a, __nv_bfloat162 b)
{
  return __bfloat162float(a.x) == __bfloat162float(b.x) && __bfloat162float(a.y) == __bfloat162float(b.y);
}
#  endif // defined(__CUDA_NO_BFLOAT162_OPERATORS__)
#endif // _CCCL_HAS_NVBF16()

template <typename T>
__host__ __device__ void test_memcpy_roundtrip(T from)
{
  static_assert(cuda::is_trivially_copyable_v<T>);
  T to;
  ::memcpy(static_cast<void*>(&to), static_cast<const void*>(&from), sizeof(T));

#if _CCCL_CTK_AT_LEAST(12, 3)
  assert(from == to);
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 3) ^^^ / !_CCCL_CTK_AT_LEAST(12, 3) vvv
  if constexpr (cuda::std::is_same_v<T, __nv_bfloat162>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80, (assert(from == to);));
  }
  else
  {
    NV_IF_TARGET(NV_IS_DEVICE, (assert(from == to);));
  }
#endif // ^^^ !_CCCL_CTK_AT_LEAST(12, 3) ^^^
}

#define CAST(base_type, val) static_cast<decltype(base_type##1 ::x)>(val)

#define REPEAT_1(base_type, index) CAST(base_type, input[index][0])
#define REPEAT_2(base_type, index) REPEAT_1(base_type, index), CAST(base_type, input[index][1])
#define REPEAT_3(base_type, index) REPEAT_2(base_type, index), CAST(base_type, input[index][2])
#define REPEAT_4(base_type, index) REPEAT_3(base_type, index), CAST(base_type, input[index][3])

#define TEST_CUDA_VECTOR_TYPE(base_type, size)           \
  {                                                      \
    for (base_type##size i :                             \
         {base_type##size{REPEAT_##size(base_type, 0)},  \
          base_type##size{REPEAT_##size(base_type, 1)},  \
          base_type##size{REPEAT_##size(base_type, 2)},  \
          base_type##size{REPEAT_##size(base_type, 3)},  \
          base_type##size{REPEAT_##size(base_type, 4)},  \
          base_type##size{REPEAT_##size(base_type, 5)},  \
          base_type##size{REPEAT_##size(base_type, 6)}}) \
    {                                                    \
      test_memcpy_roundtrip(i);                          \
    }                                                    \
  }

#define TEST_CUDA_VECTOR_TYPES(base_type) \
  TEST_CUDA_VECTOR_TYPE(base_type, 1)     \
  TEST_CUDA_VECTOR_TYPE(base_type, 2)     \
  TEST_CUDA_VECTOR_TYPE(base_type, 3)     \
  TEST_CUDA_VECTOR_TYPE(base_type, 4)

__host__ __device__ bool tests()
{
  // standard scalar types
  test_memcpy_roundtrip(42);
  test_memcpy_roundtrip(0.0f);
  test_memcpy_roundtrip(3.14159);
  test_memcpy_roundtrip(static_cast<short>(7));
  test_memcpy_roundtrip(static_cast<char>('A'));

  // cuda::std::pair
  using pair = cuda::std::pair<float, int>;
  for (pair i :
       {pair{0.0f, 1},
        pair{1.0f, 2},
        pair{-1.0f, 3},
        pair{10.0f, 4},
        pair{-10.0f, 5},
        pair{2.71828f, 6},
        pair{3.14159f, 7}})
  {
    test_memcpy_roundtrip(i);
  }

  // cuda::std::tuple
  using tuple = cuda::std::tuple<int, float>;
  for (tuple i :
       {tuple{1, 0.0f},
        tuple{2, 1.0f},
        tuple{3, -1.0f},
        tuple{4, 10.0f},
        tuple{5, -10.0f},
        tuple{6, 2.71828f},
        tuple{7, 3.14159f}})
  {
    test_memcpy_roundtrip(i);
  }

  // cuda::std::array
  using array = cuda::std::array<float, 2>;
  for (array i :
       {array{0.0f, 1.0f},
        array{1.0f, 2.0f},
        array{-1.0f, 3.0f},
        array{10.0f, 4.0f},
        array{-10.0f, 5.0f},
        array{2.71828f, 6.0f},
        array{3.14159f, 7.0f}})
  {
    test_memcpy_roundtrip(i);
  }

  // CUDA vector types
  constexpr double input[7][4] = {
    {0.0, 1.0, -7.0, -0.0},
    {1.0, 2.0, -7.0, -1.0},
    {-1.0, 3.0, -7.0, 1.0},
    {10.0, 4.0, -7.0, -10.0},
    {-10.0, 5.0, -7.0, 10.0},
    {2.71828, 6.0, -7.0, -2.71828},
    {3.14159, 7.0, -7.0, -3.14159}};

  TEST_CUDA_VECTOR_TYPES(char)
  TEST_CUDA_VECTOR_TYPES(short)
  TEST_CUDA_VECTOR_TYPES(int)
  TEST_CUDA_VECTOR_TYPES(float)

#if !_CCCL_CUDA_COMPILER(CLANG)
  using uchar  = unsigned char;
  using ushort = unsigned short;
  using uint   = unsigned int;
  using ulong  = unsigned long;
  TEST_CUDA_VECTOR_TYPES(uchar)
  TEST_CUDA_VECTOR_TYPES(ushort)
  TEST_CUDA_VECTOR_TYPES(uint)
  TEST_CUDA_VECTOR_TYPE(ulong, 1)
  TEST_CUDA_VECTOR_TYPE(ulong, 2)
  TEST_CUDA_VECTOR_TYPE(ulong, 3)
#endif // !_CCCL_CUDA_COMPILER(CLANG)

  TEST_CUDA_VECTOR_TYPE(long, 1)
  TEST_CUDA_VECTOR_TYPE(long, 2)
  TEST_CUDA_VECTOR_TYPE(long, 3)

  using longlong  = long long;
  using ulonglong = unsigned long long;
  TEST_CUDA_VECTOR_TYPE(longlong, 1)
  TEST_CUDA_VECTOR_TYPE(longlong, 2)
  TEST_CUDA_VECTOR_TYPE(longlong, 3)
  TEST_CUDA_VECTOR_TYPE(ulonglong, 1)
  TEST_CUDA_VECTOR_TYPE(ulonglong, 2)
  TEST_CUDA_VECTOR_TYPE(ulonglong, 3)
  TEST_CUDA_VECTOR_TYPE(double, 1)
  TEST_CUDA_VECTOR_TYPE(double, 2)
  TEST_CUDA_VECTOR_TYPE(double, 3)

  for (dim3 i :
       {dim3{0u, 1u, 2u},
        dim3{1u, 2u, 3u},
        dim3{10u, 20u, 30u},
        dim3{100u, 200u, 300u},
        dim3{255u, 128u, 64u},
        dim3{1024u, 512u, 256u},
        dim3{4096u, 2048u, 1024u}})
  {
    test_memcpy_roundtrip(i);
  }

  return true;
}

// Extended floating-point types: operator== is __device__ only on CTK < 12.3.
#if _CCCL_CTK_AT_LEAST(12, 3)
__host__ __device__ void tests_nvfp()
#else
__device__ void tests_nvfp()
#endif
{
#if _CCCL_HAS_NVFP16()
  for (__half i :
       {__float2half(0.0f),
        __float2half(1.0f),
        __float2half(-1.0f),
        __float2half(10.0f),
        __float2half(-10.0f),
        __float2half(2.71828f),
        __float2half(3.14159f)})
  {
    test_memcpy_roundtrip(i);
  }

  for (__half2 i :
       {__half2{__float2half(0.0f), __float2half(1.0f)},
        __half2{__float2half(-1.0f), __float2half(2.0f)},
        __half2{__float2half(10.0f), __float2half(-10.0f)},
        __half2{__float2half(2.71828f), __float2half(3.14159f)}})
  {
    test_memcpy_roundtrip(i);
  }

  test_memcpy_roundtrip(
    cuda::std::array<__half, 4>{__float2half(1.0f), __float2half(2.0f), __float2half(3.0f), __float2half(4.0f)});
  test_memcpy_roundtrip(cuda::std::pair<__half, __half>{__float2half(1.0f), __float2half(2.0f)});
  test_memcpy_roundtrip(cuda::std::tuple<__half, __half>{__float2half(1.0f), __float2half(2.0f)});

  test_memcpy_roundtrip(cuda::std::array<cuda::std::pair<__half, __half>, 2>{
    cuda::std::pair<__half, __half>{__float2half(1.0f), __float2half(2.0f)},
    cuda::std::pair<__half, __half>{__float2half(3.0f), __float2half(4.0f)}});
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  for (__nv_bfloat16 i :
       {__float2bfloat16(0.0f),
        __float2bfloat16(1.0f),
        __float2bfloat16(-1.0f),
        __float2bfloat16(10.0f),
        __float2bfloat16(-10.0f),
        __float2bfloat16(2.71828f),
        __float2bfloat16(3.14159f)})
  {
    test_memcpy_roundtrip(i);
  }

  for (__nv_bfloat162 i :
       {__nv_bfloat162{__float2bfloat16(0.0f), __float2bfloat16(1.0f)},
        __nv_bfloat162{__float2bfloat16(-1.0f), __float2bfloat16(2.0f)},
        __nv_bfloat162{__float2bfloat16(10.0f), __float2bfloat16(-10.0f)},
        __nv_bfloat162{__float2bfloat16(2.71828f), __float2bfloat16(3.14159f)}})
  {
    test_memcpy_roundtrip(i);
  }

  test_memcpy_roundtrip(cuda::std::array<__nv_bfloat16, 2>{__float2bfloat16(1.0f), __float2bfloat16(2.0f)});
  test_memcpy_roundtrip(cuda::std::pair<__nv_bfloat16, __nv_bfloat16>{__float2bfloat16(1.0f), __float2bfloat16(2.0f)});
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP16() && _CCCL_HAS_NVBF16()
  test_memcpy_roundtrip(cuda::std::tuple<__half, __nv_bfloat16>{__float2half(1.0f), __float2bfloat16(2.0f)});
#endif // _CCCL_HAS_NVFP16() && _CCCL_HAS_NVBF16()
}

int main(int, char**)
{
  tests();
#if _CCCL_CTK_AT_LEAST(12, 3)
  tests_nvfp();
#else
  NV_IF_TARGET(NV_IS_DEVICE, (tests_nvfp();));
#endif
  return 0;
}

_CCCL_DIAG_POP
