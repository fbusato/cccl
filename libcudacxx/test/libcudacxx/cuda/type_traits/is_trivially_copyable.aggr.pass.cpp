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
// custom type

struct CustomNonTrivialType
{
  int x;

  CustomNonTrivialType() = default;
  __host__ __device__ CustomNonTrivialType(const CustomNonTrivialType&) {}
};

template <>
constexpr bool cuda::is_trivially_copyable_v<CustomNonTrivialType> = true;

struct SingleMemberCustom
{
  CustomNonTrivialType x;
};

struct DerivedStructCustom : SingleMemberCustom
{
  CustomNonTrivialType y;
};

struct NestedStructCustom
{
  CustomNonTrivialType z;
};

struct ArrayMemberCustom
{
  CustomNonTrivialType values[2];
};

//----------------------------------------------------------------------------------------------------------------------
// non trivially copyable type

struct NonTriviallyCopyable
{
  __host__ __device__ NonTriviallyCopyable(const NonTriviallyCopyable&) {};
};

struct RelaxedWithNonRelaxedMember
{
  CustomNonTrivialType x;
  NonTriviallyCopyable y;
};

__host__ __device__ void test()
{
  test_is_trivially_copyable<SingleMember>();
  test_is_trivially_copyable<DerivedStruct>();
  test_is_trivially_copyable<NesterStruct>();
  test_is_trivially_copyable<ArrayMember>();
  test_is_trivially_copyable<ArrayMember>();
  test_is_trivially_copyable<DerivedStructCustom>();
  test_is_trivially_copyable<NestedStructCustom>();
  test_is_not_trivially_copyable<RelaxedWithNonRelaxedMember>();
}

int main(int, char**)
{
  test();
  return 0;
}
