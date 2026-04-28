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

// [simd.math], trigonometric functions

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec vec(math_values<T>{});
  Vec other(T{0.5});
  T scalar{0.5};

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::acos(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::asin(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::atan(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::atan2(vec, vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::cos(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::sin(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::tan(vec)), Vec>);
  static_assert(noexcept(cuda::std::simd::acos(vec)));
  static_assert(noexcept(cuda::std::simd::asin(vec)));
  static_assert(noexcept(cuda::std::simd::atan(vec)));
  static_assert(noexcept(cuda::std::simd::atan2(vec, vec)));
  static_assert(noexcept(cuda::std::simd::cos(vec)));
  static_assert(noexcept(cuda::std::simd::sin(vec)));
  static_assert(noexcept(cuda::std::simd::tan(vec)));

  // [simd.math]: scalar broadcast for `atan2`.
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::atan2(vec, scalar)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::atan2(scalar, vec)), Vec>);
  static_assert(noexcept(cuda::std::simd::atan2(vec, scalar)));
  static_assert(noexcept(cuda::std::simd::atan2(scalar, vec)));

  Vec acos_result  = cuda::std::simd::acos(vec);
  Vec asin_result  = cuda::std::simd::asin(vec);
  Vec atan_result  = cuda::std::simd::atan(vec);
  Vec atan2_result = cuda::std::simd::atan2(vec, vec);
  Vec cos_result   = cuda::std::simd::cos(vec);
  Vec sin_result   = cuda::std::simd::sin(vec);
  Vec tan_result   = cuda::std::simd::tan(vec);
  T tolerance      = T{1e-5};
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(acos_result[i], cuda::std::acos(vec[i]), tolerance));
    assert(almost_equal(asin_result[i], cuda::std::asin(vec[i]), tolerance));
    assert(almost_equal(atan_result[i], cuda::std::atan(vec[i]), tolerance));
    assert(almost_equal(atan2_result[i], cuda::std::atan2(vec[i], vec[i]), tolerance));
    assert(almost_equal(cos_result[i], cuda::std::cos(vec[i]), tolerance));
    assert(almost_equal(sin_result[i], cuda::std::sin(vec[i]), tolerance));
    assert(almost_equal(tan_result[i], cuda::std::tan(vec[i]), tolerance));
  }

  Vec atan2_vs = cuda::std::simd::atan2(vec, scalar);
  Vec atan2_sv = cuda::std::simd::atan2(scalar, vec);
  for (int i = 0; i < N; ++i)
  {
    assert(almost_equal(atan2_vs[i], cuda::std::atan2(vec[i], scalar), tolerance));
    assert(almost_equal(atan2_sv[i], cuda::std::atan2(scalar, vec[i]), tolerance));
  }
}

DEFINE_SIMD_MATH_FLOATING_TEST()

int main(int, char**)
{
  assert(test());
  return 0;
}
