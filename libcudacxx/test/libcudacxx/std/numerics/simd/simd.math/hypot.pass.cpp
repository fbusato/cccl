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

// [simd.math], hypot

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec x(positive_math_values<T>{});
  Vec y(T{0.5});
  Vec z(T{0.25});
  T sy{0.5};
  T sz{0.25};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, y)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, y, z)), Vec>);
  static_assert(noexcept(cuda::std::simd::hypot(x, y)));
  static_assert(noexcept(cuda::std::simd::hypot(x, y, z)));

  // [simd.math]: scalar broadcast for binary and ternary hypot.
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, sy)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(sy, x)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, y, sz)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, sy, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(sy, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(sy, sz, x)), Vec>);
  static_assert(noexcept(cuda::std::simd::hypot(x, sy)));

  Vec hypot2_result = cuda::std::simd::hypot(x, y);
  Vec hypot3_result = cuda::std::simd::hypot(x, y, z);
  T tolerance       = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(hypot2_result[i], cuda::std::hypot(x[i], y[i]), tolerance));
    assert(almost_equal(hypot3_result[i], cuda::std::hypot(x[i], y[i], z[i]), tolerance));
  }

  Vec hypot_vs  = cuda::std::simd::hypot(x, sy);
  Vec hypot_sv  = cuda::std::simd::hypot(sy, x);
  Vec hypot_vvs = cuda::std::simd::hypot(x, y, sz);
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(hypot_vs[i], cuda::std::hypot(x[i], sy), tolerance));
    assert(almost_equal(hypot_sv[i], cuda::std::hypot(sy, x[i]), tolerance));
    assert(almost_equal(hypot_vvs[i], cuda::std::hypot(x[i], y[i], sz), tolerance));
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
