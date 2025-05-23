//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr auto size() const requires see below;

#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  // Both are integer like and both are less than zero.
  {
    const cuda::std::ranges::iota_view<int, int> io(-10, -5);
    assert(io.size() == 5);
  }
  {
    const cuda::std::ranges::iota_view<int, int> io(-10, -10);
    assert(io.size() == 0);
  }

  // Both are integer like and "value_" is less than zero.
  {
    const cuda::std::ranges::iota_view<int, int> io(-10, 10);
    assert(io.size() == 20);
  }
  {
    // TODO: this is invalid with the current implementation. We need to file an LWG issue to
    // fix this. Essentially the issue is: An int's min and max are -2147483648 and 2147483647
    // which means the negated min cannot be represented as an integer; it needs to be cast to
    // an unsigned type first. That seems to be what the
    // to-unsigned-like(bound_) + to-unsigned-like(-value_))
    // part of https://eel.is/c++draft/range.iota#view-15 is doing, but I think it's doing it
    // wrong. It should be to-unsigned-like(bound_) - to-unsigned-like(value_)) (cast to
    // unsigned first).
    //     const cuda::std::ranges::iota_view<int, int> io(cuda::std::numeric_limits<int>::min(),
    //     cuda::std::numeric_limits<int>::max()); assert(io.size() ==
    //     (static_cast<unsigned>(cuda::std::numeric_limits<int>::max()) * 2) + 1);
  }

  // It is UB for "bound_" to be less than "value_" i.e.: iota_view<int, int> io(10, -5).

  // Both are integer like and neither less than zero.
  {
    const cuda::std::ranges::iota_view<int, int> io(10, 20);
    assert(io.size() == 10);
  }
  {
    const cuda::std::ranges::iota_view<int, int> io(10, 10);
    assert(io.size() == 0);
  }
  {
    const cuda::std::ranges::iota_view<int, int> io(0, 0);
    assert(io.size() == 0);
  }
  {
    const cuda::std::ranges::iota_view<int, int> io(0, cuda::std::numeric_limits<int>::max());
    constexpr auto imax = cuda::std::numeric_limits<int>::max();
    assert(io.size() == imax);
  }

  // Neither are integer like.
  {
    const cuda::std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(-20), SomeInt(-10));
    assert(io.size() == 10);
  }
  {
    const cuda::std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(-10), SomeInt(-10));
    assert(io.size() == 0);
  }
  {
    const cuda::std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(0), SomeInt(0));
    assert(io.size() == 0);
  }
  {
    const cuda::std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(10), SomeInt(20));
    assert(io.size() == 10);
  }
  {
    const cuda::std::ranges::iota_view<SomeInt, SomeInt> io(SomeInt(10), SomeInt(10));
    assert(io.size() == 0);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
