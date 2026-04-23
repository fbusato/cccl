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

// [simd.iterator], iterator comparisons for __simd_iterator
//
// iterator-to-iterator comparisons (==, !=, <, >, <=, >=),
// iterator-to-default_sentinel_t comparisons and noexcept guarantees.

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// iterator-to-iterator comparisons

template <typename T, int N>
TEST_FUNC constexpr void test_comparisons()
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
TEST_FUNC constexpr void test_sentinel()
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
TEST_FUNC constexpr void test_type()
{
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
