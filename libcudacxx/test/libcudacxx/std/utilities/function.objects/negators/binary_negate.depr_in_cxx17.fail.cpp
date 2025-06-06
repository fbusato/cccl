//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// binary_negate
//  deprecated in C++17

// UNSUPPORTED: clang-4.0
// REQUIRES: verify-support

#include <cuda/std/functional>

#include "test_macros.h"

struct Predicate
{
  typedef int first_argument_type;
  typedef int second_argument_type;
  bool operator()(first_argument_type, second_argument_type) const
  {
    return true;
  }
};

int main(int, char**)
{
  [[maybe_unused]] cuda::std::binary_negate<Predicate> f((Predicate())); // expected-error{{'binary_negate<Predicate>'
                                                                         // is deprecated}}

  return 0;
}
