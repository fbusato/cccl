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

// [simd.math], fdim, fmax, fmin

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

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fdim(lhs, rhs)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmax(lhs, rhs)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmin(lhs, rhs)), Vec>);
  static_assert(noexcept(cuda::std::simd::fdim(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::fmax(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::fmin(lhs, rhs)));

  // [simd.math]: scalar broadcast variants
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fdim(lhs, scalar)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fdim(scalar, rhs)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmax(lhs, scalar)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmax(scalar, rhs)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmin(lhs, scalar)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmin(scalar, rhs)), Vec>);
  static_assert(noexcept(cuda::std::simd::fdim(lhs, scalar)));
  static_assert(noexcept(cuda::std::simd::fmax(scalar, rhs)));
  static_assert(noexcept(cuda::std::simd::fmin(lhs, scalar)));

  Vec fdim_result = cuda::std::simd::fdim(lhs, rhs);
  Vec fmax_result = cuda::std::simd::fmax(lhs, rhs);
  Vec fmin_result = cuda::std::simd::fmin(lhs, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(fdim_result[i] == cuda::std::fdim(lhs[i], rhs[i]));
    assert(fmax_result[i] == cuda::std::fmax(lhs[i], rhs[i]));
    assert(fmin_result[i] == cuda::std::fmin(lhs[i], rhs[i]));
  }

  Vec fdim_vs = cuda::std::simd::fdim(lhs, scalar);
  Vec fdim_sv = cuda::std::simd::fdim(scalar, rhs);
  Vec fmax_vs = cuda::std::simd::fmax(lhs, scalar);
  Vec fmin_sv = cuda::std::simd::fmin(scalar, rhs);
  for (int i = 0; i < N; ++i)
  {
    assert(fdim_vs[i] == cuda::std::fdim(lhs[i], scalar));
    assert(fdim_sv[i] == cuda::std::fdim(scalar, rhs[i]));
    assert(fmax_vs[i] == cuda::std::fmax(lhs[i], scalar));
    assert(fmin_sv[i] == cuda::std::fmin(scalar, rhs[i]));
  }
}

DEFINE_SIMD_MATH_FLOATING_TEST()

int main(int, char**)
{
  assert(test());
  return 0;
}
