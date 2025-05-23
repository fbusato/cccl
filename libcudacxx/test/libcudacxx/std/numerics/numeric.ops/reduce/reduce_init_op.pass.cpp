//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template<class InputIterator, class T, class BinaryOperation>
//   T reduce(InputIterator first, InputIterator last, T init, BinaryOperation op);

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/numeric>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class T, class Op>
__host__ __device__ constexpr void test(Iter first, Iter last, T init, Op op, T x)
{
  static_assert(cuda::std::is_same<T, decltype(cuda::std::reduce(first, last, init, op))>::value, "");
  assert(cuda::std::reduce(first, last, init, op) == x);
}

template <class Iter>
__host__ __device__ constexpr void test()
{
  int ia[]    = {1, 2, 3, 4, 5, 6};
  unsigned sa = sizeof(ia) / sizeof(ia[0]);
  test(Iter(ia), Iter(ia), 0, cuda::std::plus<>(), 0);
  test(Iter(ia), Iter(ia), 1, cuda::std::multiplies<>(), 1);
  test(Iter(ia), Iter(ia + 1), 0, cuda::std::plus<>(), 1);
  test(Iter(ia), Iter(ia + 1), 2, cuda::std::multiplies<>(), 2);
  test(Iter(ia), Iter(ia + 2), 0, cuda::std::plus<>(), 3);
  test(Iter(ia), Iter(ia + 2), 3, cuda::std::multiplies<>(), 6);
  test(Iter(ia), Iter(ia + sa), 0, cuda::std::plus<>(), 21);
  test(Iter(ia), Iter(ia + sa), 4, cuda::std::multiplies<>(), 2880);
}

template <typename T, typename Init>
__host__ __device__ constexpr void test_return_type()
{
  T* p = nullptr;
  unused(p);
  static_assert(cuda::std::is_same<Init, decltype(cuda::std::reduce(p, p, Init{}, cuda::std::plus<>()))>::value, "");
}

__host__ __device__ constexpr bool test()
{
  test_return_type<char, int>();
  test_return_type<int, int>();
  test_return_type<int, unsigned long>();
  test_return_type<float, int>();
  test_return_type<short, float>();
  test_return_type<double, char>();
  test_return_type<char, double>();

  test<cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

  //  Make sure the math is done using the correct type
  {
    auto v       = {1, 2, 3, 4, 5, 6, 7, 8};
    unsigned res = cuda::std::reduce(v.begin(), v.end(), 1U, cuda::std::multiplies<>());
    assert(res == 40320); // 8! will not fit into a char
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
