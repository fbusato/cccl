//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class weekday_last;

//  explicit constexpr weekday_last(const chrono::weekday& wd) noexcept;
//
//  Effects: Constructs an object of type weekday_last by initializing wd_ with wd.
//
//  constexpr chrono::weekday weekday() const noexcept;
//  constexpr bool ok() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using weekday      = cuda::std::chrono::weekday;
  using weekday_last = cuda::std::chrono::weekday_last;

  ASSERT_NOEXCEPT(weekday_last{weekday{}});

  constexpr weekday_last wdl0{weekday{}};
  static_assert(wdl0.weekday() == weekday{}, "");
  static_assert(wdl0.ok(), "");

  constexpr weekday_last wdl1{weekday{1}};
  static_assert(wdl1.weekday() == weekday{1}, "");
  static_assert(wdl1.ok(), "");

  for (unsigned i = 0; i <= 255; ++i)
  {
    weekday_last wdl{weekday{i}};
    assert(wdl.weekday() == weekday{i});
  }

  return 0;
}
