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

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

namespace simd = cuda::std::simd;

// Meaningful inputs for trigonometric/hyperbolic tests: four complex values spanning all quadrants with mixed
// magnitudes
template <typename T>
struct trig_input_generator
{
  template <typename I>
  __host__ __device__ constexpr cuda::std::complex<T> operator()(I i) const noexcept
  {
    switch (static_cast<int>(i) & 3)
    {
      case 0:
        return cuda::std::complex<T>(T(0.5), T(0.3));
      case 1:
        return cuda::std::complex<T>(T(-0.7), T(0.6));
      case 2:
        return cuda::std::complex<T>(T(1.1), T(-0.9));
      default:
        return cuda::std::complex<T>(T(-0.3), T(-0.5));
    }
  }
};

//----------------------------------------------------------------------------------------------------------------------
// sin, cos, tan, asin, acos, atan

template <typename T, int N>
__host__ __device__ void test_trig()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(trig_input_generator<T>{});

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

  ComplexVec vec_sin  = simd::sin(vec);
  ComplexVec vec_cos  = simd::cos(vec);
  ComplexVec vec_tan  = simd::tan(vec);
  ComplexVec vec_asin = simd::asin(vec);
  ComplexVec vec_acos = simd::acos(vec);
  ComplexVec vec_atan = simd::atan(vec);
  for (int i = 0; i < N; ++i)
  {
    is_about(vec_sin[i], cuda::std::sin(vec[i]));
    is_about(vec_cos[i], cuda::std::cos(vec[i]));
    is_about(vec_tan[i], cuda::std::tan(vec[i]));
    is_about(vec_asin[i], cuda::std::asin(vec[i]));
    is_about(vec_acos[i], cuda::std::acos(vec[i]));
    is_about(vec_atan[i], cuda::std::atan(vec[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// sinh, cosh, tanh, asinh, acosh, atanh

template <typename T, int N>
__host__ __device__ void test_hyperbolic()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(trig_input_generator<T>{});

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

  ComplexVec vec_sinh  = simd::sinh(vec);
  ComplexVec vec_cosh  = simd::cosh(vec);
  ComplexVec vec_tanh  = simd::tanh(vec);
  ComplexVec vec_asinh = simd::asinh(vec);
  ComplexVec vec_acosh = simd::acosh(vec);
  ComplexVec vec_atanh = simd::atanh(vec);
  for (int i = 0; i < N; ++i)
  {
    is_about(vec_sinh[i], cuda::std::sinh(vec[i]));
    is_about(vec_cosh[i], cuda::std::cosh(vec[i]));
    is_about(vec_tanh[i], cuda::std::tanh(vec[i]));
    is_about(vec_asinh[i], cuda::std::asinh(vec[i]));
    is_about(vec_acosh[i], cuda::std::acosh(vec[i]));
    is_about(vec_atanh[i], cuda::std::atanh(vec[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
__host__ __device__ void test_type()
{
  test_trig<T, N>();
  test_hyperbolic<T, N>();
}

DEFINE_COMPLEX_TEST_NONCONSTEXPR()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  assert(test_runtime());
  return 0;
}
