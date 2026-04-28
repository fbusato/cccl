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

// [simd.math], roots, hypot, pow

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

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::cbrt(x)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::sqrt(x)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, y)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::pow(x, y)), Vec>);
  static_assert(noexcept(cuda::std::simd::cbrt(x)));
  static_assert(noexcept(cuda::std::simd::sqrt(x)));
  static_assert(noexcept(cuda::std::simd::hypot(x, y)));
  static_assert(noexcept(cuda::std::simd::hypot(x, y, z)));
  static_assert(noexcept(cuda::std::simd::pow(x, y)));

  // [simd.math]: scalar broadcast for `hypot` (binary and ternary) and `pow`.
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, sy)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(sy, x)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, y, sz)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(x, sy, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(sy, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::hypot(sy, sz, x)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::pow(x, sy)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::pow(sy, x)), Vec>);
  static_assert(noexcept(cuda::std::simd::hypot(x, sy)));
  static_assert(noexcept(cuda::std::simd::pow(sy, x)));

  Vec cbrt_result   = cuda::std::simd::cbrt(x);
  Vec sqrt_result   = cuda::std::simd::sqrt(x);
  Vec hypot2_result = cuda::std::simd::hypot(x, y);
  Vec hypot3_result = cuda::std::simd::hypot(x, y, z);
  Vec pow_result    = cuda::std::simd::pow(x, y);
  T tolerance       = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(cbrt_result[i], cuda::std::cbrt(x[i]), tolerance));
    assert(almost_equal(sqrt_result[i], cuda::std::sqrt(x[i]), tolerance));
    assert(almost_equal(hypot2_result[i], cuda::std::hypot(x[i], y[i]), tolerance));
    assert(almost_equal(hypot3_result[i], cuda::std::hypot(x[i], y[i], z[i]), tolerance));
    assert(almost_equal(pow_result[i], cuda::std::pow(x[i], y[i]), tolerance));
  }

  Vec hypot_vs  = cuda::std::simd::hypot(x, sy);
  Vec hypot_sv  = cuda::std::simd::hypot(sy, x);
  Vec hypot_vvs = cuda::std::simd::hypot(x, y, sz);
  Vec pow_vs    = cuda::std::simd::pow(x, sy);
  Vec pow_sv    = cuda::std::simd::pow(sy, x);
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(hypot_vs[i], cuda::std::hypot(x[i], sy), tolerance));
    assert(almost_equal(hypot_sv[i], cuda::std::hypot(sy, x[i]), tolerance));
    assert(almost_equal(hypot_vvs[i], cuda::std::hypot(x[i], y[i], sz), tolerance));
    assert(almost_equal(pow_vs[i], cuda::std::pow(x[i], sy), tolerance));
    assert(almost_equal(pow_sv[i], cuda::std::pow(sy, x[i]), tolerance));
  }
}

DEFINE_SIMD_MATH_FLOATING_TEST()

int main(int, char**)
{
  assert(test());
  return 0;
}
