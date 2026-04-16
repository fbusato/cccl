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

// [simd.complex.math] free functions: real, imag, abs, arg, norm, conj, proj,
// exp, log, log10, sqrt, polar, pow

#include <cuda/std/complex>

#include "../simd_test_utils.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// real() / imag() free functions

template <typename T, int N>
__host__ __device__ constexpr void test_real_imag_free()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(complex_generator<T, 1, 10>{});

  RealVec reals = simd::real(vec);
  RealVec imags = simd::imag(vec);

  static_assert(cuda::std::is_same_v<decltype(reals), RealVec>);
  static_assert(cuda::std::is_same_v<decltype(imags), RealVec>);
  static_assert(noexcept(simd::real(vec)));
  static_assert(noexcept(simd::imag(vec)));

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
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(3), T(4)));

  static_assert(cuda::std::is_same_v<decltype(simd::conj(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::norm(vec)), RealVec>);
  static_assert(!noexcept(simd::conj(vec)));
  static_assert(!noexcept(simd::norm(vec)));

  ComplexVec vec_cos = simd::conj(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(vec_cos[i].real() == T(3));
    assert(vec_cos[i].imag() == T(-4));
  }

  RealVec vec_norm = simd::norm(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(vec_norm[i] == T(25));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// arg()

template <typename T, int N>
__host__ __device__ void test_arg()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(1), T(0)));

  static_assert(cuda::std::is_same_v<decltype(simd::arg(vec)), RealVec>);
  static_assert(!noexcept(simd::arg(vec)));

  RealVec vec_arg = simd::arg(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(vec_arg[i] == T(0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// abs()

template <typename T, int N>
__host__ __device__ void test_abs()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(3), T(4)));

  static_assert(cuda::std::is_same_v<decltype(simd::abs(vec)), RealVec>);
  static_assert(!noexcept(simd::abs(vec)));

  RealVec vec_abs = simd::abs(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(vec_abs[i] == T(5));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// proj()

template <typename T, int N>
__host__ __device__ void test_proj()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(3), T(4)));

  static_assert(cuda::std::is_same_v<decltype(simd::proj(vec)), ComplexVec>);
  static_assert(!noexcept(simd::proj(vec)));

  ComplexVec vec_proj = simd::proj(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(vec_proj[i].real() == T(3));
    assert(vec_proj[i].imag() == T(4));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// exp, log, log10 round-trips

template <typename T, int N>
__host__ __device__ void test_exp_log()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(1), T(0)));

  static_assert(cuda::std::is_same_v<decltype(simd::exp(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::log(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::log10(vec)), ComplexVec>);
  static_assert(!noexcept(simd::exp(vec)));
  static_assert(!noexcept(simd::log(vec)));
  static_assert(!noexcept(simd::log10(vec)));

  ComplexVec vec_exp = simd::exp(vec);
  ComplexVec vec_log = simd::log(vec_exp);

  for (int i = 0; i < N; ++i)
  {
    T re_diff = vec_log[i].real() - T(1);
    T im_diff = vec_log[i].imag();
    assert(re_diff * re_diff < T(1e-6));
    assert(im_diff * im_diff < T(1e-6));
  }

  ComplexVec vec10(Complex(T(100), T(0)));
  ComplexVec vec_log10 = simd::log10(vec10);
  for (int i = 0; i < N; ++i)
  {
    T diff_re = vec_log10[i].real() - T(2);
    T diff_im = vec_log10[i].imag();
    assert(diff_re * diff_re < T(1e-6));
    assert(diff_im * diff_im < T(1e-6));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// sqrt()

template <typename T, int N>
__host__ __device__ void test_sqrt()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(4), T(0)));

  static_assert(cuda::std::is_same_v<decltype(simd::sqrt(vec)), ComplexVec>);
  static_assert(!noexcept(simd::sqrt(vec)));

  ComplexVec vec_sqrt = simd::sqrt(vec);
  for (int i = 0; i < N; ++i)
  {
    T diff_re = vec_sqrt[i].real() - T(2);
    T diff_im = vec_sqrt[i].imag();
    assert(diff_re * diff_re < T(1e-6));
    assert(diff_im * diff_im < T(1e-6));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// polar

template <typename T, int N>
__host__ __device__ void test_polar()
{
  using Complex    = cuda::std::complex<T>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  RealVec rho(T(5));
  RealVec theta(T(0));

  static_assert(cuda::std::is_same_v<decltype(simd::polar(rho, theta)), ComplexVec>);
  static_assert(!noexcept(simd::polar(rho, theta)));

  ComplexVec result = simd::polar(rho, theta);

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
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec base(Complex(T(2), T(0)));
  ComplexVec expo(Complex(T(3), T(0)));

  static_assert(cuda::std::is_same_v<decltype(simd::pow(base, expo)), ComplexVec>);
  static_assert(!noexcept(simd::pow(base, expo)));

  ComplexVec result = simd::pow(base, expo);

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
  test_arg<T, N>();
  test_abs<T, N>();
  test_proj<T, N>();
  test_exp_log<T, N>();
  test_sqrt<T, N>();
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

template <typename T, int N>
__host__ __device__ void test_type()
{
  test_constexpr<T, N>();
  test_runtime<T, N>();
}

__host__ __device__ bool test_rt()
{
  test_runtime<float, 1>();
  test_runtime<float, 4>();
  test_runtime<double, 1>();
  test_runtime<double, 4>();
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half, 1>();
  test_type<__half, 4>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16, 1>();
  test_type<__nv_bfloat16, 4>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_rt());
  return 0;
}
