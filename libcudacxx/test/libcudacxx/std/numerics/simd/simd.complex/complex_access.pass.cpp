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

// [simd.ctor] complex constructor, [simd.complex.access] complex accessors: real(), imag()

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// complex constructor

template <typename T, int N>
__host__ __device__ constexpr void test_complex_ctor()
{
  using C    = cuda::std::complex<T>;
  using Vec  = simd::basic_vec<C, simd::fixed_size<N>>;
  using FVec = simd::basic_vec<T, simd::fixed_size<N>>;

  FVec reals([](auto i) { return static_cast<T>(i + 1); });
  FVec imags([](auto i) { return static_cast<T>(i + 10); });

  Vec v(reals, imags);
  for (int i = 0; i < N; ++i)
  {
    assert(v[i].real() == static_cast<T>(i + 1));
    assert(v[i].imag() == static_cast<T>(i + 10));
  }

  Vec v_real_only(reals);
  for (int i = 0; i < N; ++i)
  {
    assert(v_real_only[i].real() == static_cast<T>(i + 1));
    assert(v_real_only[i].imag() == T(0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// real() / imag() getters

template <typename T, int N>
__host__ __device__ constexpr void test_getters()
{
  using C   = cuda::std::complex<T>;
  using Vec = simd::basic_vec<C, simd::fixed_size<N>>;

  Vec v([](auto i) { return C(static_cast<T>(i), static_cast<T>(i * 10)); });

  auto reals = v.real();
  auto imags = v.imag();

  static_assert(cuda::std::is_same_v<decltype(reals), simd::basic_vec<T, simd::fixed_size<N>>>);
  static_assert(cuda::std::is_same_v<decltype(imags), simd::basic_vec<T, simd::fixed_size<N>>>);

  static_assert(noexcept(v.real()));
  static_assert(noexcept(v.imag()));

  for (int i = 0; i < N; ++i)
  {
    assert(reals[i] == static_cast<T>(i));
    assert(imags[i] == static_cast<T>(i * 10));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// real(v) / imag(v) setters

template <typename T, int N>
__host__ __device__ constexpr void test_setters()
{
  using C   = cuda::std::complex<T>;
  using Vec = simd::basic_vec<C, simd::fixed_size<N>>;
  using FVec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec v(C(T(1), T(2)));

  FVec new_reals([](auto i) { return static_cast<T>(i + 100); });
  v.real(new_reals);

  for (int i = 0; i < N; ++i)
  {
    assert(v[i].real() == static_cast<T>(i + 100));
    assert(v[i].imag() == T(2));
  }

  FVec new_imags([](auto i) { return static_cast<T>(i + 200); });
  v.imag(new_imags);

  for (int i = 0; i < N; ++i)
  {
    assert(v[i].real() == static_cast<T>(i + 100));
    assert(v[i].imag() == static_cast<T>(i + 200));
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

__host__ __device__ constexpr bool test()
{
  test_type<float, 1>();
  test_type<float, 4>();
  test_type<double, 1>();
  test_type<double, 4>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
