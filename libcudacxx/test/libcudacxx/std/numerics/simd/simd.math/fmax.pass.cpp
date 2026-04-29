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

// [simd.math], fmax

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec lhs(positive_math_values<T>{});
  Vec rhs(T{0.5});
  T scalar{0.5};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmax(lhs, rhs)), Vec>);
  static_assert(noexcept(cuda::std::simd::fmax(lhs, rhs)));

  // [simd.math]: scalar broadcast variants
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmax(lhs, scalar)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmax(scalar, rhs)), Vec>);
  static_assert(noexcept(cuda::std::simd::fmax(scalar, rhs)));

  Vec fmax_result = cuda::std::simd::fmax(lhs, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(fmax_result[i] == cuda::std::fmax(lhs[i], rhs[i]));
  }

  Vec fmax_vs = cuda::std::simd::fmax(lhs, scalar);
  for (int i = 0; i < N; ++i)
  {
    assert(fmax_vs[i] == cuda::std::fmax(lhs[i], scalar));
  }
}

DEFINE_SIMD_MATH_FLOATING_TEST()
DEFINE_SIMD_MATH_FLOATING_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  assert(test_runtime());
  return 0;
}
