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
  char16_t s1[]  = {1, 2, 3};
  char16_t s2[3] = {0};
  assert(cuda::std::char_traits<char16_t>::copy(s2, s1, 3) == s2);
  assert(s2[0] == char16_t(1));
  assert(s2[1] == char16_t(2));
  assert(s2[2] == char16_t(3));
  assert(cuda::std::char_traits<char16_t>::copy(nullptr, s1, 0) == nullptr);
  assert(cuda::std::char_traits<char16_t>::copy(s1, nullptr, 0) == s1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
