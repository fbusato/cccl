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

// [simd.math], fma

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

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(x, y, z)), Vec>);
  static_assert(noexcept(cuda::std::simd::fma(x, y, z)));

  // we need to check all permutations of vec and scalar arguments
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(sy, y, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(x, sy, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(x, y, sz)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(sy, sz, z)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(sy, y, sz)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::fma(x, sy, sz)), Vec>);
  static_assert(noexcept(cuda::std::simd::fma(sy, y, z)));
  static_assert(noexcept(cuda::std::simd::fma(x, sy, z)));
  static_assert(noexcept(cuda::std::simd::fma(x, y, sz)));

  Vec fma_result = cuda::std::simd::fma(x, y, z);
  for (int i = 0; i < N; ++i)
  {
    assert(fma_result[i] == cuda::std::fma(x[i], y[i], z[i]));
  }

  Vec fma_mixed_a = cuda::std::simd::fma(sy, y, z);
  Vec fma_mixed_b = cuda::std::simd::fma(x, sy, z);
  Vec fma_mixed_c = cuda::std::simd::fma(x, y, sz);
  for (int i = 0; i < N; ++i)
  {
    assert(fma_mixed_a[i] == cuda::std::fma(sy, y[i], z[i]));
    assert(fma_mixed_b[i] == cuda::std::fma(x[i], sy, z[i]));
    assert(fma_mixed_c[i] == cuda::std::fma(x[i], y[i], sz));
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
