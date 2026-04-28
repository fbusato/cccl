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

// [simd.math], modulo functions

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec    = simd::basic_vec<T, simd::fixed_size<N>>;
  using IntVec = simd::rebind_t<int, Vec>;

  Vec lhs(positive_math_values<T>{});
  Vec rhs(T{0.5});
  T scalar{0.5};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmod(lhs, rhs)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::remainder(lhs, rhs)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::remquo(lhs, rhs, cuda::std::declval<IntVec*>())), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::modf(lhs, cuda::std::declval<Vec*>())), Vec>);
  static_assert(noexcept(cuda::std::simd::fmod(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::remainder(lhs, rhs)));
  static_assert(noexcept(cuda::std::simd::remquo(lhs, rhs, cuda::std::declval<IntVec*>())));
  static_assert(noexcept(cuda::std::simd::modf(lhs, cuda::std::declval<Vec*>())));

  // [simd.math]: scalar broadcast and mixed arguments.
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmod(lhs, scalar)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fmod(scalar, rhs)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::remainder(lhs, scalar)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::remainder(scalar, rhs)), Vec>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::simd::remquo(lhs, scalar, cuda::std::declval<IntVec*>())), Vec>);
  static_assert(noexcept(cuda::std::simd::fmod(lhs, scalar)));
  static_assert(noexcept(cuda::std::simd::remainder(scalar, rhs)));

  IntVec quotients;
  Vec integrals;
  Vec fmod_result      = cuda::std::simd::fmod(lhs, rhs);
  Vec remainder_result = cuda::std::simd::remainder(lhs, rhs);
  Vec remquo_result    = cuda::std::simd::remquo(lhs, rhs, &quotients);
  Vec modf_result      = cuda::std::simd::modf(lhs, &integrals);
  for (int i = 0; i < N; ++i)
  {
    int quotient = 0;
    T integral   = 0;
    assert(fmod_result[i] == cuda::std::fmod(lhs[i], rhs[i]));
    assert(remainder_result[i] == cuda::std::remainder(lhs[i], rhs[i]));
    assert(remquo_result[i] == cuda::std::remquo(lhs[i], rhs[i], &quotient));
    assert(quotients[i] == quotient);
    assert(modf_result[i] == cuda::std::modf(lhs[i], &integral));
    assert(integrals[i] == integral);
  }

  Vec fmod_vs      = cuda::std::simd::fmod(lhs, scalar);
  Vec fmod_sv      = cuda::std::simd::fmod(scalar, rhs);
  Vec remainder_vs = cuda::std::simd::remainder(lhs, scalar);
  Vec remainder_sv = cuda::std::simd::remainder(scalar, rhs);
  IntVec mixed_quotients;
  Vec remquo_mixed = cuda::std::simd::remquo(lhs, scalar, &mixed_quotients);
  for (int i = 0; i < N; ++i)
  {
    int q = 0;
    assert(fmod_vs[i] == cuda::std::fmod(lhs[i], scalar));
    assert(fmod_sv[i] == cuda::std::fmod(scalar, rhs[i]));
    assert(remainder_vs[i] == cuda::std::remainder(lhs[i], scalar));
    assert(remainder_sv[i] == cuda::std::remainder(scalar, rhs[i]));
    assert(remquo_mixed[i] == cuda::std::remquo(lhs[i], scalar, &q));
    assert(mixed_quotients[i] == q);
  }
}

DEFINE_SIMD_MATH_FLOATING_TEST()

int main(int, char**)
{
  assert(test());
  return 0;
}
