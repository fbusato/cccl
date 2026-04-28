//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.math], ordered comparison functions

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  Vec lhs(math_values<T>{});
  Vec rhs(positive_math_values<T>{});
  T scalar{0.5};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreater(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreaterequal(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isless(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::islessequal(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::islessgreater(lhs, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isunordered(lhs, rhs)), Mask>);
  static_assert(noexcept(cuda::std::simd::isgreater(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::isgreaterequal(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::isless(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::islessequal(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::islessgreater(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::isunordered(lhs, rhs)));

  // [simd.math]: scalar broadcast permutations must produce the same `mask_type`.
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreater(scalar, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isgreater(lhs, scalar)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isless(scalar, rhs)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::isless(lhs, scalar)), Mask>);
  static_assert(noexcept(cuda::std::simd::isgreater(scalar, rhs)));
  static_assert(noexcept(cuda::std::simd::isgreater(lhs, scalar)));

  Mask greater       = cuda::std::simd::isgreater(lhs, rhs);
  Mask greater_equal = cuda::std::simd::isgreaterequal(lhs, rhs);
  Mask less          = cuda::std::simd::isless(lhs, rhs);
  Mask less_equal    = cuda::std::simd::islessequal(lhs, rhs);
  Mask less_greater  = cuda::std::simd::islessgreater(lhs, rhs);
  Mask unordered     = cuda::std::simd::isunordered(lhs, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(greater[i] == cuda::std::isgreater(lhs[i], rhs[i]));
    assert(greater_equal[i] == cuda::std::isgreaterequal(lhs[i], rhs[i]));
    assert(less[i] == cuda::std::isless(lhs[i], rhs[i]));
    assert(less_equal[i] == cuda::std::islessequal(lhs[i], rhs[i]));
    assert(less_greater[i] == cuda::std::islessgreater(lhs[i], rhs[i]));
    assert(unordered[i] == cuda::std::isunordered(lhs[i], rhs[i]));
  }

  Mask greater_vs = cuda::std::simd::isgreater(lhs, scalar);
  Mask less_vs    = cuda::std::simd::isless(lhs, scalar);
  Mask greater_sv = cuda::std::simd::isgreater(scalar, rhs);
  Mask less_sv    = cuda::std::simd::isless(scalar, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(greater_vs[i] == cuda::std::isgreater(lhs[i], scalar));
    assert(less_vs[i] == cuda::std::isless(lhs[i], scalar));
    assert(greater_sv[i] == cuda::std::isgreater(scalar, rhs[i]));
    assert(less_sv[i] == cuda::std::isless(scalar, rhs[i]));
  }
}

DEFINE_SIMD_MATH_FLOATING_TEST()

int main(int, char**)
{
  assert(test());
  return 0;
}
