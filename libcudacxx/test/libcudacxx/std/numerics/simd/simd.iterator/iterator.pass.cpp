//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.iterator], class template __simd_iterator
//
// dereference, increment/decrement, compound assignment,
// arithmetic, comparisons, sentinel

#include "../simd_test_utils.h"

//----------------------------------------------------------------------------------------------------------------------
// dereference and subscript

template <typename T, int N>
__host__ __device__ constexpr void test_dereference()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec   = make_iota_vec<T, N>();

  auto it = vec.begin();
  for (int i = 0; i < N; ++i)
  {
    assert(*it == static_cast<T>(i));
    assert(it[0] == static_cast<T>(i));
    ++it;
  }

  auto it2 = vec.begin();
  for (int i = 0; i < N; ++i)
  {
    assert(it2[i] == static_cast<T>(i));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// increment and decrement

template <typename T, int N>
__host__ __device__ constexpr void test_increment_decrement()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec   = make_iota_vec<T, N>();

  // pre-increment
  auto it   = vec.begin();
  auto& ref = ++it;
  assert(&ref == &it);
  if constexpr (N > 1)
  {
    assert(*it == static_cast<T>(1));
  }

  // post-increment
  auto it2 = vec.begin();
  auto old = it2++;
  assert(*old == static_cast<T>(0));
  if constexpr (N > 1)
  {
    assert(*it2 == static_cast<T>(1));
  }

  if constexpr (N > 1)
  {
    // pre-decrement
    auto it3 = vec.begin();
    ++it3;
    auto& ref3 = --it3;
    assert(&ref3 == &it3);
    assert(*it3 == static_cast<T>(0));

    // post-decrement
    auto it4 = vec.begin();
    ++it4;
    auto old2 = it4--;
    assert(*old2 == static_cast<T>(1));
    assert(*it4 == static_cast<T>(0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// compound assignment

template <typename T, int N>
__host__ __device__ constexpr void test_compound_assignment()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec   = make_iota_vec<T, N>();

  auto it    = vec.begin();
  auto& ref1 = (it += N);
  assert(&ref1 == &it);
  assert(it == cuda::std::default_sentinel);

  auto& ref2 = (it -= N);
  assert(&ref2 == &it);
  assert(*it == static_cast<T>(0));
}

//----------------------------------------------------------------------------------------------------------------------
// arithmetic

template <typename T, int N>
__host__ __device__ constexpr void test_arithmetic()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec   = make_iota_vec<T, N>();

  auto it = vec.begin();

  auto it2 = it + N;
  assert(it2 == cuda::std::default_sentinel);

  auto it3 = N + it;
  assert(it3 == cuda::std::default_sentinel);

  auto it4 = it2 - N;
  assert(*it4 == static_cast<T>(0));

  assert(it2 - it == N);
  assert(it - it2 == -N);

  assert(it - cuda::std::default_sentinel == -N);
  assert(cuda::std::default_sentinel - it == N);
  assert(it2 - cuda::std::default_sentinel == 0);
  assert(cuda::std::default_sentinel - it2 == 0);
}

//----------------------------------------------------------------------------------------------------------------------
// comparisons

template <typename T, int N>
__host__ __device__ constexpr void test_comparisons()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec   = make_iota_vec<T, N>();

  auto a = vec.begin();
  auto b = vec.begin();

  assert(a == b);
  assert((a != b) == false);
  assert((a < b) == false);
  assert((a > b) == false);
  assert(a <= b);
  assert(a >= b);

  if constexpr (N > 1)
  {
    auto c = a + 1;
    assert(!(a == c));
    assert(a != c);
    assert(a < c);
    assert(!(a > c));
    assert(a <= c);
    assert(!(a >= c));
    assert(c > a);
    assert(c >= a);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// sentinel comparisons and noexcept

template <typename T, int N>
__host__ __device__ constexpr void test_sentinel()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec{};

  auto it = vec.begin();
  assert(!(it == cuda::std::default_sentinel));
  assert(it != cuda::std::default_sentinel);

  it += N;
  assert(it == cuda::std::default_sentinel);
  assert(!(it != cuda::std::default_sentinel));

  static_assert(noexcept(it == cuda::std::default_sentinel));
  static_assert(noexcept(it - cuda::std::default_sentinel));
  static_assert(noexcept(cuda::std::default_sentinel - it));
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
__host__ __device__ constexpr void test_type()
{
  test_dereference<T, N>();
  test_increment_decrement<T, N>();
  test_compound_assignment<T, N>();
  test_arithmetic<T, N>();
  test_comparisons<T, N>();
  test_sentinel<T, N>();
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
