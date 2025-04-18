//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_day;

// constexpr year_month_day& operator+=(const months& m) noexcept;
// constexpr year_month_day& operator-=(const months& m) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename D, typename Ds>
__host__ __device__ constexpr bool testConstexpr(D d1)
{
  if (static_cast<unsigned>((d1).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{1}).month()) != 2)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{2}).month()) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 += Ds{12}).month()) != 4)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{1}).month()) != 3)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{2}).month()) != 1)
  {
    return false;
  }
  if (static_cast<unsigned>((d1 -= Ds{12}).month()) != 1)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year           = cuda::std::chrono::year;
  using month          = cuda::std::chrono::month;
  using day            = cuda::std::chrono::day;
  using year_month_day = cuda::std::chrono::year_month_day;
  using months         = cuda::std::chrono::months;

  static_assert(noexcept(cuda::std::declval<year_month_day&>() += cuda::std::declval<months>()));
  static_assert(noexcept(cuda::std::declval<year_month_day&>() -= cuda::std::declval<months>()));

  static_assert(cuda::std::is_same_v<year_month_day&,
                                     decltype(cuda::std::declval<year_month_day&>() += cuda::std::declval<months>())>);
  static_assert(cuda::std::is_same_v<year_month_day&,
                                     decltype(cuda::std::declval<year_month_day&>() -= cuda::std::declval<months>())>);

  static_assert(testConstexpr<year_month_day, months>(year_month_day{year{1234}, month{1}, day{1}}), "");

  for (unsigned i = 0; i <= 10; ++i)
  {
    year y{1234};
    day d{23};
    year_month_day ym(y, month{i}, d);
    assert(static_cast<unsigned>((ym += months{2}).month()) == i + 2);
    assert(ym.year() == y);
    assert(ym.day() == d);
    assert(static_cast<unsigned>((ym).month()) == i + 2);
    assert(ym.year() == y);
    assert(ym.day() == d);
    assert(static_cast<unsigned>((ym -= months{1}).month()) == i + 1);
    assert(ym.year() == y);
    assert(ym.day() == d);
    assert(static_cast<unsigned>((ym).month()) == i + 1);
    assert(ym.year() == y);
    assert(ym.day() == d);
  }

  return 0;
}
