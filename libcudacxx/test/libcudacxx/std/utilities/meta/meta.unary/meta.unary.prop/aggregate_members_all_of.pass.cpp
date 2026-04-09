//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>

#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// test types

struct Empty
{};

struct OneMember
{
  int x;
};

struct TwoMembers
{
  int x;
  float y;
};

struct ThreeMembers
{
  int x;
  float y;
  double z;
};

struct AllInts
{
  int a;
  int b;
  int c;
};

struct AllFloats
{
  float a;
  double b;
};

struct MixedIntFloat
{
  int a;
  float b;
};

struct LargeAggregate
{
  int m0;
  int m1;
  int m2;
  int m3;
  int m4;
  int m5;
  int m6;
  int m7;
  int m8;
  int m9;
  int m10;
  int m11;
  int m12;
  int m13;
  int m14;
  int m15;
};

struct TooLarge
{
  int m0;
  int m1;
  int m2;
  int m3;
  int m4;
  int m5;
  int m6;
  int m7;
  int m8;
  int m9;
  int m10;
  int m11;
  int m12;
  int m13;
  int m14;
  int m15;
  int m16;
};

struct Nested
{
  OneMember inner;
  int extra;
};

struct WithArray
{
  int values[4];
};

class NonAggregate
{
public:
  __host__ __device__ NonAggregate() {}
  int x;
};

struct NonTriviallyCopyable
{
  NonTriviallyCopyable() = default;
  __host__ __device__ NonTriviallyCopyable(const NonTriviallyCopyable&) {}
  int x;
};

struct HasNonTriviallyCopyableMember
{
  int x;
  NonTriviallyCopyable y;
};

//----------------------------------------------------------------------------------------------------------------------
// predicates

template <typename _Tp>
using is_integral_pred = cuda::std::is_integral<_Tp>;

template <typename _Tp>
struct always_true : cuda::std::true_type
{};

template <typename _Tp>
struct always_false : cuda::std::false_type
{};

template <typename _Tp>
using is_trivially_copyable_pred = cuda::std::is_trivially_copyable<_Tp>;

//----------------------------------------------------------------------------------------------------------------------
// __aggregate_arity_v tests

static_assert(cuda::std::__aggregate_arity_v<Empty> == 0);
static_assert(cuda::std::__aggregate_arity_v<OneMember> == 1);
static_assert(cuda::std::__aggregate_arity_v<TwoMembers> == 2);
static_assert(cuda::std::__aggregate_arity_v<ThreeMembers> == 3);
static_assert(cuda::std::__aggregate_arity_v<AllInts> == 3);
static_assert(cuda::std::__aggregate_arity_v<AllFloats> == 2);
static_assert(cuda::std::__aggregate_arity_v<LargeAggregate> == 16);
static_assert(cuda::std::__aggregate_arity_v<NonAggregate> == -1);
#if !TEST_COMPILER(MSVC) // MSVC does not perform brace elision in SFINAE contexts
static_assert(cuda::std::__aggregate_arity_v<Nested> == 2);
static_assert(cuda::std::__aggregate_arity_v<WithArray> == 4);
#endif // !TEST_COMPILER(MSVC)

//----------------------------------------------------------------------------------------------------------------------
// __aggregate_all_of tests

// empty aggregate: always true (vacuously)
static_assert(cuda::std::__aggregate_all_of<always_true, Empty>::value);
static_assert(cuda::std::__aggregate_all_of<always_false, Empty>::value);

// all members satisfy the predicate
static_assert(cuda::std::__aggregate_all_of<is_integral_pred, AllInts>::value);
static_assert(cuda::std::__aggregate_all_of<always_true, ThreeMembers>::value);

// not all members satisfy the predicate
static_assert(!cuda::std::__aggregate_all_of<is_integral_pred, MixedIntFloat>::value);
static_assert(!cuda::std::__aggregate_all_of<always_false, OneMember>::value);
static_assert(!cuda::std::__aggregate_all_of<is_integral_pred, AllFloats>::value);

// max arity aggregate
static_assert(cuda::std::__aggregate_all_of<is_integral_pred, LargeAggregate>::value);

// too many members: returns false (arity exceeds __aggregate_max_arity)
static_assert(!cuda::std::__aggregate_all_of<is_integral_pred, TooLarge>::value);

// nested aggregate / array members: brace elision lets the predicate see the flat elements
#if !TEST_COMPILER(MSVC) // MSVC does not perform brace elision in SFINAE contexts
static_assert(cuda::std::__aggregate_all_of<is_integral_pred, Nested>::value);
static_assert(cuda::std::__aggregate_all_of<is_integral_pred, WithArray>::value);
static_assert(cuda::std::__aggregate_all_of<is_trivially_copyable_pred, Nested>::value);
static_assert(cuda::std::__aggregate_all_of<is_trivially_copyable_pred, WithArray>::value);
#endif // !TEST_COMPILER(MSVC)

// non-aggregate: always false
static_assert(!cuda::std::__aggregate_all_of<always_true, NonAggregate>::value);
static_assert(!cuda::std::__aggregate_all_of<is_integral_pred, NonAggregate>::value);

// aggregate with a non-trivially-copyable member
static_assert(!cuda::std::__aggregate_all_of<is_trivially_copyable_pred, HasNonTriviallyCopyableMember>::value);

int main(int, char**)
{
  return 0;
}
