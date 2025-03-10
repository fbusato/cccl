//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#include <test_macros.h>

// template <class IntegerType>
//    constexpr byte operator <<(byte b, IntegerType shift) noexcept;
// These functions shall not participate in overload resolution unless
//   is_integral_v<IntegerType> is true.

int main(int, char**)
{
  constexpr cuda::std::byte b1{static_cast<cuda::std::byte>(1)};
  constexpr cuda::std::byte b3{static_cast<cuda::std::byte>(3)};

  static_assert(noexcept(b3 << 2), "");

  static_assert(cuda::std::to_integer<int>(b1 << 1) == 2, "");
  static_assert(cuda::std::to_integer<int>(b1 << 2) == 4, "");
  static_assert(cuda::std::to_integer<int>(b3 << 4) == 48, "");
  static_assert(cuda::std::to_integer<int>(b3 << 6) == 192, "");

  return 0;
}
