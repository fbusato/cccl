//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/utility>

int main(int, char**)
{
  // cuda/std/memory not supported, however, with <complex> available this test needs to fail.
  typedef cuda::std::unique_ptr<int> upint;
  cuda::std::pair<upint, int> t(upint(new int(4)), 23);
  upint p = cuda::std::get<upint>(t);

  return 0;
}
