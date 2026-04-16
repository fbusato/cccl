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

// [simd.complex.math] trigonometric and hyperbolic functions:
// sin, asin, cos, acos, tan, atan, sinh, asinh, cosh, acosh, tanh, atanh

#include <cuda/std/complex>

#include "../simd_test_utils.h"

namespace simd = cuda::std::simd;

template <typename T>
__host__ __device__ T trig_tol()
{
  return T(1e-5);
}

#if _LIBCUDACXX_HAS_NVFP16()
template <>
__host__ __device__ __half trig_tol<__half>()
{
  return __half(1e-2);
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
template <>
__host__ __device__ __nv_bfloat16 trig_tol<__nv_bfloat16>()
{
  return __nv_bfloat16(1e-1);
}
#endif // _LIBCUDACXX_HAS_NVBF16()

//----------------------------------------------------------------------------------------------------------------------
// sin, cos, tan, asin, acos, atan

template <typename T, int N>
__host__ __device__ void test_trig()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(0.5), T(0)));

  static_assert(cuda::std::is_same_v<decltype(simd::sin(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::cos(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::tan(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::asin(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::acos(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::atan(vec)), ComplexVec>);
  static_assert(!noexcept(simd::sin(vec)));
  static_assert(!noexcept(simd::cos(vec)));
  static_assert(!noexcept(simd::tan(vec)));
  static_assert(!noexcept(simd::asin(vec)));
  static_assert(!noexcept(simd::acos(vec)));
  static_assert(!noexcept(simd::atan(vec)));

  ComplexVec vec_sin = simd::sin(vec);
  ComplexVec vec_cos = simd::cos(vec);
  ComplexVec vec_tan = simd::tan(vec);

  for (int i = 0; i < N; ++i)
  {
    Complex sum = vec_sin[i] * vec_sin[i] + vec_cos[i] * vec_cos[i];
    T diff_re   = sum.real() - T(1);
    T diff_im   = sum.imag();
    assert(diff_re * diff_re < trig_tol<T>());
    assert(diff_im * diff_im < trig_tol<T>());

    Complex ratio = vec_sin[i] / vec_cos[i];
    T tan_diff_re = ratio.real() - vec_tan[i].real();
    T tan_diff_im = ratio.imag() - vec_tan[i].imag();
    assert(tan_diff_re * tan_diff_re < trig_tol<T>());
    assert(tan_diff_im * tan_diff_im < trig_tol<T>());
  }

  ComplexVec vec_asin = simd::asin(vec_sin);
  ComplexVec vec_acos = simd::acos(vec_cos);
  ComplexVec vec_atan = simd::atan(vec_tan);
  for (int i = 0; i < N; ++i)
  {
    T asin_diff = vec_asin[i].real() - T(0.5);
    T acos_diff = vec_acos[i].real() - T(0.5);
    T atan_diff = vec_atan[i].real() - T(0.5);
    assert(asin_diff * asin_diff < trig_tol<T>());
    assert(acos_diff * acos_diff < trig_tol<T>());
    assert(atan_diff * atan_diff < trig_tol<T>());
  }
}

//----------------------------------------------------------------------------------------------------------------------
// sinh, cosh, tanh, asinh, acosh, atanh

template <typename T, int N>
__host__ __device__ void test_hyperbolic()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(Complex(T(0.5), T(0)));

  static_assert(cuda::std::is_same_v<decltype(simd::sinh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::cosh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::tanh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::asinh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::acosh(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::atanh(vec)), ComplexVec>);
  static_assert(!noexcept(simd::sinh(vec)));
  static_assert(!noexcept(simd::cosh(vec)));
  static_assert(!noexcept(simd::tanh(vec)));
  static_assert(!noexcept(simd::asinh(vec)));
  static_assert(!noexcept(simd::acosh(vec)));
  static_assert(!noexcept(simd::atanh(vec)));

  ComplexVec vec_sinh = simd::sinh(vec);
  ComplexVec vec_cosh = simd::cosh(vec);
  ComplexVec vec_tanh = simd::tanh(vec);

  for (int i = 0; i < N; ++i)
  {
    Complex diff = vec_cosh[i] * vec_cosh[i] - vec_sinh[i] * vec_sinh[i];
    T diff_re    = diff.real() - T(1);
    T diff_im    = diff.imag();
    assert(diff_re * diff_re < trig_tol<T>());
    assert(diff_im * diff_im < trig_tol<T>());

    Complex ratio  = vec_sinh[i] / vec_cosh[i];
    T tanh_diff_re = ratio.real() - vec_tanh[i].real();
    T tanh_diff_im = ratio.imag() - vec_tanh[i].imag();
    assert(tanh_diff_re * tanh_diff_re < trig_tol<T>());
    assert(tanh_diff_im * tanh_diff_im < trig_tol<T>());
  }

  ComplexVec vec_asinh = simd::asinh(vec_sinh);
  ComplexVec vec_atanh = simd::atanh(vec_tanh);
  for (int i = 0; i < N; ++i)
  {
    T asinh_diff = vec_asinh[i].real() - T(0.5);
    T atanh_diff = vec_atanh[i].real() - T(0.5);
    assert(asinh_diff * asinh_diff < trig_tol<T>());
    assert(atanh_diff * atanh_diff < trig_tol<T>());
  }

  ComplexVec vec2(Complex(T(2), T(0)));
  ComplexVec vec_acosh = simd::acosh(vec2);
  ComplexVec vec_cosh2 = simd::cosh(vec_acosh);
  for (int i = 0; i < N; ++i)
  {
    T diff_re = vec_cosh2[i].real() - T(2);
    T diff_im = vec_cosh2[i].imag();
    assert(diff_re * diff_re < trig_tol<T>());
    assert(diff_im * diff_im < trig_tol<T>());
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
__host__ __device__ void test_type()
{
  test_trig<T, N>();
  test_hyperbolic<T, N>();
}

__host__ __device__ bool test()
{
  test_type<float, 1>();
  test_type<float, 4>();
  test_type<double, 1>();
  test_type<double, 4>();
  return true;
}

__host__ __device__ bool test_runtime()
{
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
  assert(test_runtime());
  return 0;
}
