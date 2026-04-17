//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef SIMD_TEST_UTILS_H
#define SIMD_TEST_UTILS_H

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "fp_compare.h"
#include "test_macros.h"

namespace simd = cuda::std::simd;

//----------------------------------------------------------------------------------------------------------------------
// common utilities

struct wrong_generator
{};

template <typename>
struct is_const_member_function : cuda::std::false_type
{};

template <typename R, typename C, typename... Args>
struct is_const_member_function<R (C::*)(Args...) const> : cuda::std::true_type
{};

template <typename R, typename C, typename... Args>
struct is_const_member_function<R (C::*)(Args...) const noexcept> : cuda::std::true_type
{};

template <typename T>
constexpr bool is_const_member_function_v = is_const_member_function<T>::value;

//----------------------------------------------------------------------------------------------------------------------
// mask utilities

struct is_even
{
  template <typename I>
  __host__ __device__ constexpr bool operator()(I i) const noexcept
  {
    return i % 2 == 0;
  }
};

struct is_first_half
{
  template <typename I>
  __host__ __device__ constexpr bool operator()(I i) const noexcept
  {
    return i < 2;
  }
};

template <int Bytes>
using integer_from_t = cuda::std::__make_nbit_int_t<Bytes * 8, true>;

//----------------------------------------------------------------------------------------------------------------------
// Approximate floating-point comparison for SIMD complex math tests
//
// Scalar and SIMD code paths produce bit-identical results under nvcc with gcc
// or clang as the host compiler, but nvc++ and clang-cuda legitimately diverge
// in the last few bits due to different FMA contraction and intrinsic choices.
// `is_about` delegates to the shared `fptest_close` helper in fp_compare.h on
// those two toolchains, with per-type tolerances matching the long-standing
// `is_about` overloads in libcudacxx/test/.../complex.number/cases.h, and
// falls back to bit-exact `==` everywhere else so regressions on the
// nvcc+host-compiler path are still caught.

// Compiler gate: bit-exact on nvcc+gcc and nvrtc, relaxed on every toolchain
// where FMA contraction or intrinsic selection legitimately diverges
// (nvc++ and every compilation whose host compiler is clang, which covers
// nvcc+clang and clang-cuda).
//
// Constant-evaluated arithmetic is IEEE-754-deterministic on every toolchain,
// so bit-exact equality is always safe at compile time. The `fptest_close`
// path is reserved for runtime evaluation. Reduced-precision types are
// widened to `float` before delegating because `__half(int)` and
// `__nv_bfloat16(int)` are not constexpr constructors, so instantiating
// `fptest_close<T>` (which contains `constexpr T zero = T(0)`) is ill-formed
// for them.
#if _CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(CLANG)
__host__ __device__ constexpr void is_about(float a, float b)
{
  if (cuda::std::is_constant_evaluated())
  {
    assert(a == b);
  }
  else
  {
    assert(fptest_close(a, b, 1.e-6f));
  }
}

__host__ __device__ constexpr void is_about(double a, double b)
{
  if (cuda::std::is_constant_evaluated())
  {
    assert(a == b);
  }
  else
  {
    assert(fptest_close(a, b, 1.e-14));
  }
}

#  if _LIBCUDACXX_HAS_NVFP16()
__host__ __device__ inline void is_about(__half a, __half b)
{
  assert(fptest_close(static_cast<float>(a), static_cast<float>(b), 1.e-3f));
}
#  endif // _LIBCUDACXX_HAS_NVFP16()

#  if _LIBCUDACXX_HAS_NVBF16()
__host__ __device__ inline void is_about(__nv_bfloat16 a, __nv_bfloat16 b)
{
  assert(fptest_close(static_cast<float>(a), static_cast<float>(b), 5.e-3f));
}
#  endif // _LIBCUDACXX_HAS_NVBF16()

template <typename T>
__host__ __device__ constexpr void is_about(const cuda::std::complex<T>& a, const cuda::std::complex<T>& b)
{
  if (cuda::std::is_constant_evaluated())
  {
    assert(a == b);
  }
  else
  {
    is_about(a.real(), b.real());
    is_about(a.imag(), b.imag());
  }
}
#else // nvcc+gcc and nvrtc: enforce bit-exact equality.
template <typename T>
__host__ __device__ constexpr void is_about(T a, T b)
{
  assert(a == b);
}
template <typename T>
__host__ __device__ constexpr void is_about(const cuda::std::complex<T>& a, const cuda::std::complex<T>& b)
{
  assert(a == b);
}
#endif // _CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(CLANG)

//----------------------------------------------------------------------------------------------------------------------
// vec utilities

template <typename T>
struct iota_generator
{
  template <typename I>
  __host__ __device__ constexpr T operator()(I i) const noexcept
  {
    return static_cast<T>(i + 1);
  }
};

template <typename T, int Offset>
struct offset_generator
{
  template <typename I>
  __host__ __device__ constexpr T operator()(I i) const noexcept
  {
    return static_cast<T>(i + Offset);
  }
};

template <typename T, int RealOffset, int ImagOffset>
struct complex_generator
{
  template <typename I>
  __host__ __device__ constexpr cuda::std::complex<T> operator()(I i) const noexcept
  {
    return cuda::std::complex<T>(static_cast<T>(i + RealOffset), static_cast<T>(i + ImagOffset));
  }
};

template <typename T, int N>
__host__ __device__ constexpr simd::basic_vec<T, simd::fixed_size<N>> make_iota_vec()
{
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i);
  }
  return simd::basic_vec<T, simd::fixed_size<N>>(arr);
}

// Each vec test file must define test_type<T, N>() and then define test() using this macro.
// clang-format off
#if defined(__cccl_lib_char8_t)
#  define _SIMD_TEST_CHAR8_T()                                    \
    test_type<char8_t, 1>();                                      \
    test_type<char8_t, 4>();
#else
#  define _SIMD_TEST_CHAR8_T()
#endif

#if _CCCL_HAS_INT128()
#  define _SIMD_TEST_INT128()                                     \
    test_type<__int128_t, 1>();                                   \
    test_type<__int128_t, 4>();
#else
#  define _SIMD_TEST_INT128()
#endif

#if _LIBCUDACXX_HAS_NVFP16()
#  define _SIMD_TEST_FP16()                                       \
    test_type<__half, 1>();                                       \
    test_type<__half, 4>();
#else
#  define _SIMD_TEST_FP16()
#endif

#if _LIBCUDACXX_HAS_NVBF16()
#  define _SIMD_TEST_BF16()                                       \
    test_type<__nv_bfloat16, 1>();                                \
    test_type<__nv_bfloat16, 4>();
#else
#  define _SIMD_TEST_BF16()
#endif

// __half and __nv_bfloat16 constructors are not constexpr (CUDA toolkit limitation),
// so they are tested only at runtime via test_runtime().
#define DEFINE_BASIC_VEC_TEST_RUNTIME()                           \
  __host__ __device__ bool test_runtime()                         \
  {                                                               \
    _SIMD_TEST_FP16()                                             \
    _SIMD_TEST_BF16()                                             \
    return true;                                                  \
  }

#define DEFINE_BASIC_VEC_TEST()                                   \
  __host__ __device__ constexpr bool test()                       \
  {                                                               \
    test_type<int8_t, 1>();                                       \
    test_type<int8_t, 4>();                                       \
    test_type<int16_t, 1>();                                      \
    test_type<int16_t, 4>();                                      \
    test_type<int32_t, 1>();                                      \
    test_type<int32_t, 4>();                                      \
    test_type<int64_t, 1>();                                      \
    test_type<int64_t, 4>();                                      \
    test_type<uint8_t, 1>();                                      \
    test_type<uint8_t, 4>();                                      \
    test_type<uint16_t, 1>();                                     \
    test_type<uint16_t, 4>();                                     \
    test_type<uint32_t, 1>();                                     \
    test_type<uint32_t, 4>();                                     \
    test_type<uint64_t, 1>();                                     \
    test_type<uint64_t, 4>();                                     \
    test_type<char16_t, 1>();                                     \
    test_type<char16_t, 4>();                                     \
    test_type<char32_t, 1>();                                     \
    test_type<char32_t, 4>();                                     \
    test_type<wchar_t, 1>();                                      \
    test_type<wchar_t, 4>();                                      \
    _SIMD_TEST_CHAR8_T()                                          \
    test_type<float, 1>();                                        \
    test_type<float, 4>();                                        \
    test_type<double, 1>();                                       \
    test_type<double, 4>();                                       \
    _SIMD_TEST_INT128()                                           \
    return true;                                                  \
  }

// Complex types only use float and double for constexpr tests.
// __half and __nv_bfloat16 are tested via DEFINE_BASIC_VEC_TEST_RUNTIME().

#define DEFINE_COMPLEX_TEST()                                     \
  __host__ __device__ constexpr bool test()                       \
  {                                                               \
    test_type<float, 1>();                                        \
    test_type<float, 4>();                                        \
    test_type<double, 1>();                                       \
    test_type<double, 4>();                                       \
    return true;                                                  \
  }

#define DEFINE_COMPLEX_TEST_NONCONSTEXPR()                        \
  __host__ __device__ bool test()                                 \
  {                                                               \
    test_type<float, 1>();                                        \
    test_type<float, 4>();                                        \
    test_type<double, 1>();                                       \
    test_type<double, 4>();                                       \
    return true;                                                  \
  }
// clang-format on

#endif // SIMD_TEST_UTILS_H
