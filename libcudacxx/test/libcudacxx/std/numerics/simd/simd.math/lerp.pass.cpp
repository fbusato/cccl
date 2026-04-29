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

// [simd.math], lerp

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec x(positive_math_values<T>{});
  Vec y(T{2});
  Vec z(T{0.25});
  T sy{2};
  T sz{0.25};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(x, y, z)), Vec>);
  static_assert(noexcept(cuda::std::simd::lerp(x, y, z)));

  // [simd.math]: each ternary overload accepts any combination of vec and scalar arguments where
  // at least one operand is a basic_vec. Verify return type and noexcept for each permutation.
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(sy, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(x, sy, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(x, y, sz)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(sy, sz, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(sy, y, sz)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::lerp(x, sy, sz)), Vec>);
  static_assert(noexcept(cuda::std::simd::lerp(sy, y, z)));
  static_assert(noexcept(cuda::std::simd::lerp(x, sy, z)));
  static_assert(noexcept(cuda::std::simd::lerp(x, y, sz)));

  Vec lerp_result = cuda::std::simd::lerp(x, y, z);
  for (int i = 0; i < N; ++i)
  {
    assert(lerp_result[i] == cuda::std::lerp(x[i], y[i], z[i]));
  }

  Vec lerp_mixed_a = cuda::std::simd::lerp(x, sy, z);
  Vec lerp_mixed_b = cuda::std::simd::lerp(x, y, sz);
  for (int i = 0; i < N; ++i)
  {
    assert(lerp_mixed_a[i] == cuda::std::lerp(x[i], sy, z[i]));
    assert(lerp_mixed_b[i] == cuda::std::lerp(x[i], y[i], sz));
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
