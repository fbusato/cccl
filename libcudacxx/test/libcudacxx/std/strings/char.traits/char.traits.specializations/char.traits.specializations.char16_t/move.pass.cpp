//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__string_>
#include <cuda/std/cassert>

__host__ __device__ constexpr bool test()
{
  char16_t s1[] = {1, 2, 3};
  assert(cuda::std::char_traits<char16_t>::move(s1, s1 + 1, 2) == s1);
  assert(s1[0] == char16_t(2));
  assert(s1[1] == char16_t(3));
  assert(s1[2] == char16_t(3));
  s1[2] = char16_t(0);
  assert(cuda::std::char_traits<char16_t>::move(s1 + 1, s1, 2) == s1 + 1);
  assert(s1[0] == char16_t(2));
  assert(s1[1] == char16_t(2));
  assert(s1[2] == char16_t(3));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
