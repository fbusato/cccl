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

// [simd.math], manipulation functions

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

template <typename T, int N>
TEST_FUNC void test_type()
{
  using Vec     = simd::basic_vec<T, simd::fixed_size<N>>;
  using IntVec  = simd::rebind_t<int, Vec>;
  using LongVec = simd::rebind_t<long, Vec>;

  Vec vec(positive_math_values<T>{});
  Vec next(T{2});
  IntVec exponents(1);
  LongVec long_exponents(1);
  T scalar_next{2};
  int scalar_exp       = 1;
  long scalar_long_exp = 1;

  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::frexp(vec, cuda::std::declval<IntVec*>())), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::ilogb(vec)), IntVec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::ldexp(vec, exponents)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::logb(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::scalbn(vec, exponents)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::scalbln(vec, long_exponents)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::nextafter(vec, next)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::copysign(vec, next)), Vec>);
  static_assert(noexcept(cuda::std::simd::frexp(vec, cuda::std::declval<IntVec*>())));
  static_assert(noexcept(cuda::std::simd::ilogb(vec)));
  static_assert(noexcept(cuda::std::simd::ldexp(vec, exponents)));
  static_assert(noexcept(cuda::std::simd::nextafter(vec, next)));
  static_assert(noexcept(cuda::std::simd::copysign(vec, next)));

  // [simd.math]: scalar broadcast variants. `nextafter` and `copysign` admit `(deduced-vec-t<V>, V)` and
  // `(V, deduced-vec-t<V>)` overloads via implicit conversion of the scalar to the deduced vec.
  // `ldexp`/`scalbn`/`scalbln` accept a scalar exponent that broadcasts to `rebind_t<int, V>` /
  // `rebind_t<long, V>`.
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::nextafter(vec, scalar_next)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::nextafter(scalar_next, next)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::copysign(vec, scalar_next)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::copysign(scalar_next, next)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::ldexp(vec, scalar_exp)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::scalbn(vec, scalar_exp)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::simd::scalbln(vec, scalar_long_exp)), Vec>);
  static_assert(noexcept(cuda::std::simd::nextafter(vec, scalar_next)));
  static_assert(noexcept(cuda::std::simd::ldexp(vec, scalar_exp)));

  IntVec frexp_exponents;
  Vec frexp_result     = cuda::std::simd::frexp(vec, &frexp_exponents);
  IntVec ilogb_result  = cuda::std::simd::ilogb(vec);
  Vec ldexp_result     = cuda::std::simd::ldexp(vec, exponents);
  Vec logb_result      = cuda::std::simd::logb(vec);
  Vec scalbn_result    = cuda::std::simd::scalbn(vec, exponents);
  Vec scalbln_result   = cuda::std::simd::scalbln(vec, long_exponents);
  Vec nextafter_result = cuda::std::simd::nextafter(vec, next);
  Vec copysign_result  = cuda::std::simd::copysign(vec, next);
  for (int i = 0; i < N; ++i)
  {
    int exponent = 0;
    assert(frexp_result[i] == cuda::std::frexp(vec[i], &exponent));
    assert(frexp_exponents[i] == exponent);
    assert(ilogb_result[i] == cuda::std::ilogb(vec[i]));
    assert(ldexp_result[i] == cuda::std::ldexp(vec[i], exponents[i]));
    assert(logb_result[i] == cuda::std::logb(vec[i]));
    assert(scalbn_result[i] == cuda::std::scalbn(vec[i], exponents[i]));
    assert(scalbln_result[i] == cuda::std::scalbln(vec[i], long_exponents[i]));
    assert(nextafter_result[i] == cuda::std::nextafter(vec[i], next[i]));
    assert(copysign_result[i] == cuda::std::copysign(vec[i], next[i]));
  }

  Vec ldexp_scalar   = cuda::std::simd::ldexp(vec, scalar_exp);
  Vec scalbn_scalar  = cuda::std::simd::scalbn(vec, scalar_exp);
  Vec scalbln_scalar = cuda::std::simd::scalbln(vec, scalar_long_exp);
  Vec nextafter_vs   = cuda::std::simd::nextafter(vec, scalar_next);
  Vec nextafter_sv   = cuda::std::simd::nextafter(scalar_next, next);
  Vec copysign_vs    = cuda::std::simd::copysign(vec, scalar_next);
  Vec copysign_sv    = cuda::std::simd::copysign(scalar_next, next);
  for (int i = 0; i < N; ++i)
  {
    assert(ldexp_scalar[i] == cuda::std::ldexp(vec[i], scalar_exp));
    assert(scalbn_scalar[i] == cuda::std::scalbn(vec[i], scalar_exp));
    assert(scalbln_scalar[i] == cuda::std::scalbln(vec[i], scalar_long_exp));
    assert(nextafter_vs[i] == cuda::std::nextafter(vec[i], scalar_next));
    assert(nextafter_sv[i] == cuda::std::nextafter(scalar_next, next[i]));
    assert(copysign_vs[i] == cuda::std::copysign(vec[i], scalar_next));
    assert(copysign_sv[i] == cuda::std::copysign(scalar_next, next[i]));
  }
}

DEFINE_SIMD_MATH_FLOATING_TEST()

int main(int, char**)
{
  assert(test());
  return 0;
}
