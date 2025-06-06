//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   requires LessThanComparable<Iter::value_type>
//   Iter
//   is_sorted_until(Iter first, Iter last);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
__host__ __device__ constexpr void test()
{
  {
    int a[]     = {0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a)) == Iter(a));
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }

  {
    int a[]     = {0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
  {
    int a[]     = {0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
  {
    int a[]     = {1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 1));
  }
  {
    int a[]     = {1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }

  {
    int a[]     = {0, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
  {
    int a[]     = {0, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
  {
    int a[]     = {0, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 2));
  }
  {
    int a[]     = {0, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
  {
    int a[]     = {1, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 1));
  }
  {
    int a[]     = {1, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 1));
  }
  {
    int a[]     = {1, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 2));
  }
  {
    int a[]     = {1, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }

  {
    int a[]     = {0, 0, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
  {
    int a[]     = {0, 0, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
  {
    int a[]     = {0, 0, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 3));
  }
  {
    int a[]     = {0, 0, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
  {
    int a[]     = {0, 1, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 2));
  }
  {
    int a[]     = {0, 1, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 2));
  }
  {
    int a[]     = {0, 1, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 3));
  }
  {
    int a[]     = {0, 1, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
  {
    int a[]     = {1, 0, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 1));
  }
  {
    int a[]     = {1, 0, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 1));
  }
  {
    int a[]     = {1, 0, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 1));
  }
  {
    int a[]     = {1, 0, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 1));
  }
  {
    int a[]     = {1, 1, 0, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 2));
  }
  {
    int a[]     = {1, 1, 0, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 2));
  }
  {
    int a[]     = {1, 1, 1, 0};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + 3));
  }
  {
    int a[]     = {1, 1, 1, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    assert(cuda::std::is_sorted_until(Iter(a), Iter(a + sa)) == Iter(a + sa));
  }
}

__host__ __device__ constexpr bool test()
{
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
