//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// template <BidirectionalIterator Iter1, BidirectionalIterator Iter2>
//   requires HasEqualTo<Iter1, Iter2>
// bool operator!=(const reverse_iterator<Iter1>& x, const reverse_iterator<Iter2>& y); // constexpr in C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
__host__ __device__ constexpr void test(It l, It r, bool x)
{
  const cuda::std::reverse_iterator<It> r1(l);
  const cuda::std::reverse_iterator<It> r2(r);
  assert((r1 != r2) == x);
}

__host__ __device__ constexpr bool tests()
{
  const char* s = "1234567890";
  test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s), false);
  test(bidirectional_iterator<const char*>(s), bidirectional_iterator<const char*>(s + 1), true);
  test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s), false);
  test(random_access_iterator<const char*>(s), random_access_iterator<const char*>(s + 1), true);
  test(s, s, false);
  test(s, s + 1, true);
  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
