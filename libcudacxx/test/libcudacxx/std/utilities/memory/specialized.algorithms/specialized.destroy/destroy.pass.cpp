//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: gcc-6

// <memory>

// template <class ForwardIt>
// constexpr void destroy(ForwardIt, ForwardIt);

// #include <cuda/std/memory>
#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

struct Counted
{
  int* counter_ = nullptr;
  __host__ __device__ constexpr Counted(int* counter)
      : counter_(counter)
  {
    ++*counter_;
  }
  __host__ __device__ constexpr Counted(Counted const& other)
      : counter_(other.counter_)
  {
    ++*counter_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~Counted()
  {
    --*counter_;
  }
  __host__ __device__ friend void operator&(Counted) = delete;
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_arrays()
{
  {
    int counter     = 0;
    Counted pool[3] = {{&counter}, {&counter}, {&counter}};
    assert(counter == 3);

    cuda::std::destroy(pool, pool + 3);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy(pool, pool + 3)), void>);
    assert(counter == 0);

    // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
    for (int i = 0; i < 3; ++i)
    {
      cuda::std::__construct_at(pool + i, &counter);
    }
  }
  {
    using Array   = Counted[2];
    int counter   = 0;
    Array pool[3] = {{{&counter}, {&counter}}, {{&counter}, {&counter}}, {{&counter}, {&counter}}};
    assert(counter == 3 * 2);

    cuda::std::destroy(pool, pool + 3);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy(pool, pool + 3)), void>);
    assert(counter == 0);

    // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 2; ++j)
      {
        cuda::std::__construct_at(pool[i] + j, &counter);
      }
    }
  }

  return true;
}

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX20 void test()
{
  int counter     = 0;
  Counted pool[5] = {{&counter}, {&counter}, {&counter}, {&counter}, {&counter}};
  assert(counter == 5);

  cuda::std::destroy(It(pool), It(pool + 5));
  static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy(It(pool), It(pool + 5))), void>);
  assert(counter == 0);

  // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
  for (int i = 0; i < 5; ++i)
  {
    cuda::std::__construct_at(pool + i, &counter);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool tests()
{
  test<Counted*>();
  test<forward_iterator<Counted*>>();
  return true;
}

int main(int, char**)
{
  tests();
  test_arrays();
#if TEST_STD_VER > 2017
#  if !TEST_COMPILER(NVRTC)
#    if TEST_COMPILER(CLANG, >, 10) || TEST_COMPILER(GCC, >, 9) || TEST_COMPILER(MSVC2022) || TEST_COMPILER(NVHPC)
  static_assert(tests());
  // TODO: Until cuda::std::__construct_at has support for arrays, it's impossible to test this
  //       in a constexpr context (see https://reviews.llvm.org/D114903).
  // static_assert(test_arrays());
#    endif
#  endif // TEST_COMPILER(NVRTC)
#endif // TEST_STD_VER > 2017
  return 0;
}
