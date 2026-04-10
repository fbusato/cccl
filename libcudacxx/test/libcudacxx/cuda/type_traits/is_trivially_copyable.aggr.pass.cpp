//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/type_traits>

template <class _Tp>
__host__ __device__ void test_is_trivially_copyable()
{
  static_assert(cuda::is_trivially_copyable<_Tp>::value);
  static_assert(cuda::is_trivially_copyable<const _Tp>::value);
  static_assert(cuda::is_trivially_copyable_v<_Tp>);
  static_assert(cuda::is_trivially_copyable_v<const _Tp>);
}

template <class _Tp>
__host__ __device__ void test_is_not_trivially_copyable()
{
  static_assert(!cuda::is_trivially_copyable<_Tp>::value);
  static_assert(!cuda::is_trivially_copyable<const _Tp>::value);
  static_assert(!cuda::is_trivially_copyable_v<_Tp>);
  static_assert(!cuda::is_trivially_copyable_v<const _Tp>);
}

struct SingleMember
{
  int x;
};

struct DerivedStruct : SingleMember
{
  int y;
};

struct NesterStruct
{
  SingleMember x;
};

struct ArrayMember
{
  int values[2];
};

//----------------------------------------------------------------------------------------------------------------------
// non trivially copyable type

struct NonTriviallyCopyable
{
  __host__ __device__ NonTriviallyCopyable(const NonTriviallyCopyable&) {};
};

struct AggregateWithNonTriviallyCopyableMember
{
  int x;
  NonTriviallyCopyable y;
};

__host__ __device__ void test()
{
  test_is_trivially_copyable<SingleMember>();
  test_is_trivially_copyable<DerivedStruct>();
  test_is_trivially_copyable<NesterStruct>();
  test_is_trivially_copyable<ArrayMember>();
  test_is_not_trivially_copyable<NonTriviallyCopyable>();
  test_is_not_trivially_copyable<AggregateWithNonTriviallyCopyableMember>();
}

int main(int, char**)
{
  test();
  return 0;
}
