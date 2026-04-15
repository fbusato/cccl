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

// [simd.complex.math] free functions: real, imag, abs, arg, norm, conj, proj,
// exp, log, log10, sqrt, sin, asin, cos, acos, tan, atan, sinh, asinh, cosh,
// acosh, tanh, atanh, polar, pow

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/__simd/complex_math.h>

#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// real() / imag() free functions

template <typename T, int N>
__host__ __device__ constexpr void test_real_imag_free()
{
  using C   = cuda::std::complex<T>;
  using Vec = simd::basic_vec<C, simd::fixed_size<N>>;
  using FVec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec v([](auto i) { return C(static_cast<T>(i + 1), static_cast<T>(i + 10)); });

  FVec reals = simd::real(v);
  FVec imags = simd::imag(v);

  static_assert(cuda::std::is_same_v<decltype(reals), FVec>);
  static_assert(cuda::std::is_same_v<decltype(imags), FVec>);

  static_assert(noexcept(simd::real(v)));
  static_assert(noexcept(simd::imag(v)));

  for (int i = 0; i < N; ++i)
  {
    assert(reals[i] == static_cast<T>(i + 1));
    assert(imags[i] == static_cast<T>(i + 10));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// conj() / norm()

template <typename T, int N>
__host__ __device__ constexpr void test_conj_norm()
{
  using C   = cuda::std::complex<T>;
  using Vec = simd::basic_vec<C, simd::fixed_size<N>>;
  using FVec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec v(C(T(3), T(4)));

  Vec vc = simd::conj(v);
  for (int i = 0; i < N; ++i)
  {
    assert(vc[i].real() == T(3));
    assert(vc[i].imag() == T(-4));
  }

  FVec vn = simd::norm(v);
  for (int i = 0; i < N; ++i)
  {
    assert(vn[i] == T(25));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// abs()

template <typename T, int N>
__host__ __device__ constexpr void test_abs()
{
  using C   = cuda::std::complex<T>;
  using Vec = simd::basic_vec<C, simd::fixed_size<N>>;
  using FVec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec v(C(T(3), T(4)));
  FVec va = simd::abs(v);
  for (int i = 0; i < N; ++i)
  {
    assert(va[i] == T(5));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// exp, log round-trip

template <typename T, int N>
__host__ __device__ void test_exp_log()
{
  using C   = cuda::std::complex<T>;
  using Vec = simd::basic_vec<C, simd::fixed_size<N>>;

  Vec v(C(T(1), T(0)));
  Vec ve = simd::exp(v);
  Vec vl = simd::log(ve);

  for (int i = 0; i < N; ++i)
  {
    T re_diff = vl[i].real() - T(1);
    T im_diff = vl[i].imag();
    assert(re_diff * re_diff < T(1e-6));
    assert(im_diff * im_diff < T(1e-6));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// sin, cos identity: sin^2 + cos^2 = 1

template <typename T, int N>
__host__ __device__ void test_sin_cos()
{
  using C   = cuda::std::complex<T>;
  using Vec = simd::basic_vec<C, simd::fixed_size<N>>;

  Vec v(C(T(0.5), T(0)));
  Vec vs = simd::sin(v);
  Vec vc = simd::cos(v);

  for (int i = 0; i < N; ++i)
  {
    C sum = vs[i] * vs[i] + vc[i] * vc[i];
    T diff_re = sum.real() - T(1);
    T diff_im = sum.imag();
    assert(diff_re * diff_re < T(1e-5));
    assert(diff_im * diff_im < T(1e-5));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// polar

template <typename T, int N>
__host__ __device__ void test_polar()
{
  using C    = cuda::std::complex<T>;
  using FVec = simd::basic_vec<T, simd::fixed_size<N>>;
  using CVec = simd::basic_vec<C, simd::fixed_size<N>>;

  FVec rho(T(5));
  FVec theta(T(0));
  CVec result = simd::polar(rho, theta);

  for (int i = 0; i < N; ++i)
  {
    T diff_re = result[i].real() - T(5);
    T diff_im = result[i].imag();
    assert(diff_re * diff_re < T(1e-6));
    assert(diff_im * diff_im < T(1e-6));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// pow

template <typename T, int N>
__host__ __device__ constexpr void test_pow()
{
  using C   = cuda::std::complex<T>;
  using Vec = simd::basic_vec<C, simd::fixed_size<N>>;

  Vec base(C(T(2), T(0)));
  Vec expo(C(T(3), T(0)));
  Vec result = simd::pow(base, expo);

  for (int i = 0; i < N; ++i)
  {
    T diff_re = result[i].real() - T(8);
    T diff_im = result[i].imag();
    assert(diff_re * diff_re < T(1e-4));
    assert(diff_im * diff_im < T(1e-4));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
__host__ __device__ constexpr void test_constexpr()
{
  test_real_imag_free<T, N>();
  test_conj_norm<T, N>();
}

template <typename T, int N>
__host__ __device__ void test_runtime()
{
  test_abs<T, N>();
  test_exp_log<T, N>();
  test_sin_cos<T, N>();
  test_polar<T, N>();
  test_pow<T, N>();
}

__host__ __device__ constexpr bool test()
{
  test_constexpr<float, 1>();
  test_constexpr<float, 4>();
  test_constexpr<double, 1>();
  test_constexpr<double, 4>();
  return true;
}

__host__ __device__ bool test_rt()
{
  test_runtime<float, 1>();
  test_runtime<float, 4>();
  test_runtime<double, 1>();
  test_runtime<double, 4>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_rt());
  return 0;
}
