//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_day_last;

// constexpr year_month_day_last
//   operator/(const year_month& ym, last_spec) noexcept;
// Returns: {ym.year(), month_day_last{ym.month()}}.

// constexpr year_month_day_last
//   operator/(const year& y, const month_day_last& mdl) noexcept;
// Returns: {y, mdl}.
//
// constexpr year_month_day_last
//   operator/(int y, const month_day_last& mdl) noexcept;
// Returns: year(y) / mdl.
//
// constexpr year_month_day_last
//   operator/(const month_day_last& mdl, const year& y) noexcept;
// Returns: y / mdl.
//
// constexpr year_month_day_last
//   operator/(const month_day_last& mdl, int y) noexcept;
// Returns: year(y) / mdl.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month               = cuda::std::chrono::month;
  using year_month          = cuda::std::chrono::year_month;
  using year                = cuda::std::chrono::year;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;

  constexpr month February                    = cuda::std::chrono::February;
  constexpr cuda::std::chrono::last_spec last = cuda::std::chrono::last;

  { // operator/(const year_month& ym, last_spec)
    constexpr year_month Feb2018{year{2018}, February};

    static_assert(noexcept(Feb2018 / last));
    static_assert(cuda::std::is_same_v<year_month_day_last, decltype(Feb2018 / last)>);

    static_assert((Feb2018 / last).year() == year{2018}, "");
    static_assert((Feb2018 / last).month() == February, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        year y{i};
        month m{j};
        year_month_day_last ymdl = year_month{y, m} / last;
        assert(ymdl.year() == y);
        assert(ymdl.month() == m);
      }
    }
  }

  { // operator/(const year& y, const month_day_last& mdl) (and switched)
    static_assert(noexcept(year{2018} / month_day_last{February}));
    static_assert(cuda::std::is_same_v<year_month_day_last, decltype(year{2018} / month_day_last{February})>);
    static_assert(noexcept(month_day_last{February} / year{2018}));
    static_assert(cuda::std::is_same_v<year_month_day_last, decltype(month_day_last{February} / year{2018})>);

    static_assert((year{2018} / month_day_last{February}).month() == February, "");
    static_assert((year{2018} / month_day_last{February}).year() == year{2018}, "");
    static_assert((month_day_last{February} / year{2018}).month() == February, "");
    static_assert((month_day_last{February} / year{2018}).year() == year{2018}, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        year y{i};
        month m{j};
        year_month_day_last ymdl1 = y / month_day_last{m};
        year_month_day_last ymdl2 = month_day_last{m} / y;
        assert(ymdl1.month() == m);
        assert(ymdl2.month() == m);
        assert(ymdl2.year() == y);
        assert(ymdl1.year() == y);
        assert(ymdl1 == ymdl2);
      }
    }
  }

  { // operator/(int y, const month_day_last& mdl) (and switched)
    static_assert(noexcept(2018 / month_day_last{February}));
    static_assert(cuda::std::is_same_v<year_month_day_last, decltype(2018 / month_day_last{February})>);
    static_assert(noexcept(month_day_last{February} / 2018));
    static_assert(cuda::std::is_same_v<year_month_day_last, decltype(month_day_last{February} / 2018)>);

    static_assert((2018 / month_day_last{February}).month() == February, "");
    static_assert((2018 / month_day_last{February}).year() == year{2018}, "");
    static_assert((month_day_last{February} / 2018).month() == February, "");
    static_assert((month_day_last{February} / 2018).year() == year{2018}, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        year y{i};
        month m{j};
        year_month_day_last ymdl1 = i / month_day_last{m};
        year_month_day_last ymdl2 = month_day_last{m} / i;
        assert(ymdl1.month() == m);
        assert(ymdl2.month() == m);
        assert(ymdl2.year() == y);
        assert(ymdl1.year() == y);
        assert(ymdl1 == ymdl2);
      }
    }
  }

  return 0;
}
