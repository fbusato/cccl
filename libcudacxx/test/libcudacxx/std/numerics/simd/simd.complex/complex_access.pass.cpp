//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA Complex++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.ctor] complex constructor, [simd.complex.access] complex accessors: real(), imag()

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"
#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// complex constructor

template <typename T, int N>
__host__ __device__ constexpr void test_complex_ctor()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  RealVec reals(offset_generator<T, 1>{});
  RealVec imags(offset_generator<T, 10>{});

  static_assert(noexcept(ComplexVec(reals, imags)));
  static_assert(noexcept(ComplexVec(reals)));

  ComplexVec vec(reals, imags);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i].real() == static_cast<T>(i + 1));
    assert(vec[i].imag() == static_cast<T>(i + 10));
  }

  ComplexVec vec_real_only(reals);
  for (int i = 0; i < N; ++i)
  {
    assert(vec_real_only[i].real() == static_cast<T>(i + 1));
    assert(vec_real_only[i].imag() == T(0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// real() / imag() getters

template <typename T, int N>
__host__ __device__ constexpr void test_getters()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(complex_generator<T, 0, 10>{});

  auto reals = vec.real();
  auto imags = vec.imag();

  static_assert(cuda::std::is_same_v<decltype(reals), simd::basic_vec<T, simd::fixed_size<N>>>);
  static_assert(cuda::std::is_same_v<decltype(imags), simd::basic_vec<T, simd::fixed_size<N>>>);
  static_assert(noexcept(vec.real()));
  static_assert(noexcept(vec.imag()));

  for (int i = 0; i < N; ++i)
  {
    assert(reals[i] == static_cast<T>(i));
    assert(imags[i] == static_cast<T>(i + 10));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// real(vec) / imag(vec) setters

template <typename T, int N>
__host__ __device__ constexpr void test_setters()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(1), T(2)));

  static_assert(noexcept(vec.real(cuda::std::declval<const RealVec&>())));
  static_assert(noexcept(vec.imag(cuda::std::declval<const RealVec&>())));

  RealVec new_reals(offset_generator<T, 100>{});
  vec.real(new_reals);

  for (int i = 0; i < N; ++i)
  {
    assert(vec[i].real() == static_cast<T>(i + 100));
    assert(vec[i].imag() == T(2));
  }

  RealVec new_imags(offset_generator<T, 200>{});
  vec.imag(new_imags);

  for (int i = 0; i < N; ++i)
  {
    assert(vec[i].real() == static_cast<T>(i + 100));
    assert(vec[i].imag() == static_cast<T>(i + 200));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
__host__ __device__ constexpr void test_type()
{
  test_complex_ctor<T, N>();
  test_getters<T, N>();
  test_setters<T, N>();
}

DEFINE_COMPLEX_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
