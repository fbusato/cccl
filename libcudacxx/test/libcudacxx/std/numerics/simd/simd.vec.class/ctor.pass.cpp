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

// [simd.ctor], basic_vec constructors
//
// constexpr explicit basic_vec(Up&&) noexcept;                                     // broadcast (explicit)
// constexpr basic_vec(Up&&) noexcept;                                              // broadcast (implicit)
// constexpr explicit basic_vec(const basic_vec<U,UAbi>&) noexcept;                 // converting (explicit)
// constexpr basic_vec(const basic_vec<U,UAbi>&) noexcept;                          // converting (implicit)
// constexpr basic_vec(Range&&, flags<> = {});                                      // range
// constexpr basic_vec(Range&&, const mask_type&, flags<> = {});                    // masked range

#include "../simd_test_utils.h"

//----------------------------------------------------------------------------------------------------------------------
// member types and size

template <typename T, int N>
__host__ __device__ constexpr void test_member_types()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<typename Vec::value_type, T>);
  static_assert(cuda::std::is_same_v<typename Vec::abi_type, simd::fixed_size<N>>);
  static_assert(Vec::size() == N);
}

//----------------------------------------------------------------------------------------------------------------------
// broadcast constructor

template <typename T, int N>
__host__ __device__ constexpr void test_broadcast()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  static_assert(noexcept(Vec(cuda::std::declval<T>()))); // declval<T>() is needed for __half and __nv_bfloat16

  Vec vec(T{42});
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == T{42});
  }

  Vec zero(T{0});
  for (int i = 0; i < N; ++i)
  {
    assert(zero[i] == T{0});
  }
}

//----------------------------------------------------------------------------------------------------------------------
// converting constructor

template <typename T, typename U, int N>
__host__ __device__ constexpr void test_converting()
{
  using Src = simd::basic_vec<U, simd::fixed_size<N>>;
  using Dst = simd::basic_vec<T, simd::fixed_size<N>>;
  Src src(U{3});
  static_assert(noexcept(Dst(src)));

  Dst dst(src);
  for (int i = 0; i < N; ++i)
  {
    assert(dst[i] == static_cast<T>(U{3}));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// range constructor

template <typename T, int N>
__host__ __device__ constexpr void test_range()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i + 1);
  }

  static_assert(!noexcept(Vec(arr)));
  static_assert(!noexcept(Vec(arr, simd::flag_default)));

  Vec vec(arr);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<T>(i + 1));
  }

  Vec vec2(arr, simd::flag_default);
  for (int i = 0; i < N; ++i)
  {
    assert(vec2[i] == static_cast<T>(i + 1));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// masked range constructor

template <typename T, int N>
__host__ __device__ constexpr void test_masked_range()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(i + 1);
  }

  Mask even_mask(is_even{});
  static_assert(!noexcept(Vec(arr, even_mask)));
  static_assert(!noexcept(Vec(arr, even_mask, simd::flag_default)));

  Vec vec(arr, even_mask);
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      assert(vec[i] == static_cast<T>(i + 1));
    }
    else
    {
      assert(vec[i] == T{0});
    }
  }

  Vec vec2(arr, even_mask, simd::flag_default);
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      assert(vec2[i] == static_cast<T>(i + 1));
    }
    else
    {
      assert(vec2[i] == T{0});
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// range constructor with flag_convert
// constructs a basic_vec<T> from an array<U> with simd::flag_convert, where U is wider than T (not value-preserving)

template <typename T, typename U, int N>
__host__ __device__ constexpr void test_range_convert_lossy()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  cuda::std::array<U, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<U>(i + 1);
  }

  static_assert(!noexcept(Vec(arr, simd::flag_convert)));

  Vec vec(arr, simd::flag_convert);
  for (int i = 0; i < N; ++i)
  {
    assert(vec[i] == static_cast<T>(static_cast<U>(i + 1)));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// masked range constructor with flag_convert
// constructs a basic_vec<T> from an array<U> with simd::flag_convert, where U is wider than T (not value-preserving)

template <typename T, typename U, int N>
__host__ __device__ constexpr void test_masked_range_convert_lossy()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  cuda::std::array<U, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<U>(i + 1);
  }

  Mask even_mask(is_even{});
  static_assert(!noexcept(Vec(arr, even_mask, simd::flag_convert)));

  Vec vec(arr, even_mask, simd::flag_convert);
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      assert(vec[i] == static_cast<T>(static_cast<U>(i + 1)));
    }
    else
    {
      assert(vec[i] == T{0});
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// SFINAE constraints

template <typename T, int N>
__host__ __device__ constexpr void test_sfinae()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  static_assert(cuda::std::is_constructible_v<Vec, T>);

  using VecDifferentSize = simd::basic_vec<T, simd::fixed_size<N + 1>>;
  static_assert(!cuda::std::is_constructible_v<Vec, const VecDifferentSize&>);

  static_assert(!cuda::std::is_constructible_v<Vec, wrong_generator>);
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
__host__ __device__ constexpr void test_type()
{
  test_member_types<T, N>();
  test_broadcast<T, N>();
  test_range<T, N>();
  test_masked_range<T, N>();
  test_sfinae<T, N>();
  if constexpr (sizeof(T) >= 2 && cuda::std::is_integral_v<T>)
  {
    using Smaller = cuda::std::conditional_t<cuda::std::is_signed_v<T>, cuda::std::int8_t, cuda::std::uint8_t>;
    test_converting<T, Smaller, N>();
  }
  if constexpr (sizeof(T) < 8 && cuda::std::is_integral_v<T>)
  {
    using Wider = cuda::std::conditional_t<cuda::std::is_signed_v<T>, cuda::std::int64_t, cuda::std::uint64_t>;
    test_range_convert_lossy<T, Wider, N>();
    test_masked_range_convert_lossy<T, Wider, N>();
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    test_range_convert_lossy<T, double, N>();
    test_masked_range_convert_lossy<T, double, N>();
  }
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
