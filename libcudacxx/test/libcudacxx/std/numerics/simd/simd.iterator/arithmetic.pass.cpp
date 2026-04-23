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

// [simd.iterator], iterator arithmetic for __simd_iterator
//
// pre/post increment and decrement, compound assignment (+=, -=),
// binary arithmetic (it + n, n + it, it - n, it - it), and differences
// between iterators and default_sentinel_t.

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// increment and decrement

template <typename T, int N>
TEST_FUNC constexpr void test_increment_decrement()
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
TEST_FUNC constexpr void test_compound_assignment()
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
// binary arithmetic

template <typename T, int N>
TEST_FUNC constexpr void test_arithmetic()
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

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_increment_decrement<T, N>();
  test_compound_assignment<T, N>();
  test_arithmetic<T, N>();
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
