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

// [simd.bit], bit manipulation

#include <cuda/std/__simd_>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename T>
struct bit_values
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I) const noexcept
  {
    return static_cast<T>((I::value + 1) * 3);
  }
};

template <typename T>
struct shift_values
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I) const noexcept
  {
    return static_cast<T>(I::value + 1);
  }
};

template <typename T, int N>
TEST_FUNC constexpr void test_byteswap()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  const Vec vec(bit_values<T>{});
  static_assert(cuda::std::is_same_v<decltype(simd::byteswap(vec)), Vec>);
  static_assert(noexcept(simd::byteswap(vec)));

  const Vec result = simd::byteswap(vec);
  for (int i = 0; i < N; ++i)
  {
    assert(result[i] == cuda::std::byteswap(vec[i]));
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_unsigned_unary()
{
  using Vec       = simd::basic_vec<T, simd::fixed_size<N>>;
  using SignedVec = simd::rebind_t<cuda::std::make_signed_t<T>, Vec>;
  using Mask      = typename Vec::mask_type;

  const Vec vec(bit_values<T>{});
  static_assert(cuda::std::is_same_v<decltype(simd::bit_ceil(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::bit_floor(vec)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::has_single_bit(vec)), Mask>);
  static_assert(cuda::std::is_same_v<decltype(simd::bit_width(vec)), SignedVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::countl_zero(vec)), SignedVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::countl_one(vec)), SignedVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::countr_zero(vec)), SignedVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::countr_one(vec)), SignedVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::popcount(vec)), SignedVec>);

  static_assert(!noexcept(simd::bit_ceil(vec)));
  static_assert(noexcept(simd::bit_floor(vec)));
  static_assert(noexcept(simd::has_single_bit(vec)));
  static_assert(noexcept(simd::bit_width(vec)));
  static_assert(noexcept(simd::countl_zero(vec)));
  static_assert(noexcept(simd::countl_one(vec)));
  static_assert(noexcept(simd::countr_zero(vec)));
  static_assert(noexcept(simd::countr_one(vec)));
  static_assert(noexcept(simd::popcount(vec)));

  const Vec bit_ceil_result       = simd::bit_ceil(vec);
  const Vec bit_floor_result      = simd::bit_floor(vec);
  const Mask has_single_bit_result = simd::has_single_bit(vec);
  const SignedVec bit_width_result = simd::bit_width(vec);
  const SignedVec countl_zero_result = simd::countl_zero(vec);
  const SignedVec countl_one_result  = simd::countl_one(vec);
  const SignedVec countr_zero_result = simd::countr_zero(vec);
  const SignedVec countr_one_result  = simd::countr_one(vec);
  const SignedVec popcount_result    = simd::popcount(vec);

  for (int i = 0; i < N; ++i)
  {
    assert(bit_ceil_result[i] == cuda::std::bit_ceil(vec[i]));
    assert(bit_floor_result[i] == cuda::std::bit_floor(vec[i]));
    assert(has_single_bit_result[i] == cuda::std::has_single_bit(vec[i]));
    assert(bit_width_result[i] == static_cast<typename SignedVec::value_type>(cuda::std::bit_width(vec[i])));
    assert(countl_zero_result[i] == static_cast<typename SignedVec::value_type>(cuda::std::countl_zero(vec[i])));
    assert(countl_one_result[i] == static_cast<typename SignedVec::value_type>(cuda::std::countl_one(vec[i])));
    assert(countr_zero_result[i] == static_cast<typename SignedVec::value_type>(cuda::std::countr_zero(vec[i])));
    assert(countr_one_result[i] == static_cast<typename SignedVec::value_type>(cuda::std::countr_one(vec[i])));
    assert(popcount_result[i] == static_cast<typename SignedVec::value_type>(cuda::std::popcount(vec[i])));
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_rotates()
{
  using Vec      = simd::basic_vec<T, simd::fixed_size<N>>;
  using ShiftVec = simd::rebind_t<cuda::std::make_signed_t<T>, Vec>;

  const Vec vec(bit_values<T>{});
  const ShiftVec shifts(shift_values<typename ShiftVec::value_type>{});
  static_assert(cuda::std::is_same_v<decltype(simd::rotl(vec, shifts)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::rotr(vec, shifts)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::rotl(vec, 1)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::rotr(vec, 1)), Vec>);
  static_assert(noexcept(simd::rotl(vec, shifts)));
  static_assert(noexcept(simd::rotr(vec, shifts)));
  static_assert(noexcept(simd::rotl(vec, 1)));
  static_assert(noexcept(simd::rotr(vec, 1)));

  const Vec rotl_vec_result   = simd::rotl(vec, shifts);
  const Vec rotr_vec_result   = simd::rotr(vec, shifts);
  const Vec rotl_scalar_result = simd::rotl(vec, -1);
  const Vec rotr_scalar_result = simd::rotr(vec, -1);

  for (int i = 0; i < N; ++i)
  {
    assert(rotl_vec_result[i] == cuda::std::rotl(vec[i], static_cast<int>(shifts[i])));
    assert(rotr_vec_result[i] == cuda::std::rotr(vec[i], static_cast<int>(shifts[i])));
    assert(rotl_scalar_result[i] == cuda::std::rotl(vec[i], -1));
    assert(rotr_scalar_result[i] == cuda::std::rotr(vec[i], -1));
  }
}

template <typename T, int N>
TEST_FUNC constexpr void test_unsigned_type()
{
  test_byteswap<T, N>();
  test_unsigned_unary<T, N>();
  test_rotates<T, N>();
}

template <typename V, typename = void>
struct has_simd_byteswap : cuda::std::false_type
{};

template <typename V>
struct has_simd_byteswap<V, cuda::std::void_t<decltype(simd::byteswap(cuda::std::declval<V>()))>>
    : cuda::std::true_type
{};

template <typename V, typename = void>
struct has_simd_bit_floor : cuda::std::false_type
{};

template <typename V>
struct has_simd_bit_floor<V, cuda::std::void_t<decltype(simd::bit_floor(cuda::std::declval<V>()))>>
    : cuda::std::true_type
{};

template <typename V, typename = void>
struct has_simd_has_single_bit : cuda::std::false_type
{};

template <typename V>
struct has_simd_has_single_bit<V, cuda::std::void_t<decltype(simd::has_single_bit(cuda::std::declval<V>()))>>
    : cuda::std::true_type
{};

template <typename V0, typename V1, typename = void>
struct has_simd_rotl_vec : cuda::std::false_type
{};

template <typename V0, typename V1>
struct has_simd_rotl_vec<V0, V1, cuda::std::void_t<decltype(simd::rotl(cuda::std::declval<V0>(), cuda::std::declval<V1>()))>>
    : cuda::std::true_type
{};

template <typename V, typename = void>
struct has_simd_rotl_scalar : cuda::std::false_type
{};

template <typename V>
struct has_simd_rotl_scalar<V, cuda::std::void_t<decltype(simd::rotl(cuda::std::declval<V>(), 1))>>
    : cuda::std::true_type
{};

TEST_FUNC constexpr void test_constraints()
{
  using IntVec     = simd::basic_vec<cuda::std::int32_t, simd::fixed_size<4>>;
  using FloatVec   = simd::basic_vec<float, simd::fixed_size<4>>;
  using Uint16Vec  = simd::basic_vec<cuda::std::uint16_t, simd::fixed_size<4>>;
  using Uint32Vec  = simd::basic_vec<cuda::std::uint32_t, simd::fixed_size<4>>;
  using Int32Vec2  = simd::basic_vec<cuda::std::int32_t, simd::fixed_size<2>>;
  using Shift32Vec = simd::basic_vec<cuda::std::int32_t, simd::fixed_size<4>>;

  static_assert(has_simd_byteswap<IntVec>::value);
  static_assert(!has_simd_byteswap<FloatVec>::value);

  static_assert(!has_simd_bit_floor<IntVec>::value);
  static_assert(!has_simd_bit_floor<FloatVec>::value);
  static_assert(!has_simd_has_single_bit<IntVec>::value);
  static_assert(!has_simd_has_single_bit<FloatVec>::value);

  static_assert(has_simd_rotl_vec<Uint32Vec, Shift32Vec>::value);
  static_assert(!has_simd_rotl_vec<IntVec, Shift32Vec>::value);
  static_assert(!has_simd_rotl_vec<Uint32Vec, Uint16Vec>::value);
  static_assert(!has_simd_rotl_vec<Uint32Vec, Int32Vec2>::value);

  static_assert(has_simd_rotl_scalar<Uint32Vec>::value);
  static_assert(!has_simd_rotl_scalar<IntVec>::value);
}

TEST_FUNC constexpr bool test()
{
  test_byteswap<cuda::std::int8_t, 1>();
  test_byteswap<cuda::std::int8_t, 4>();
  test_byteswap<cuda::std::int16_t, 1>();
  test_byteswap<cuda::std::int16_t, 4>();
  test_byteswap<cuda::std::int32_t, 1>();
  test_byteswap<cuda::std::int32_t, 4>();
  test_byteswap<cuda::std::int64_t, 1>();
  test_byteswap<cuda::std::int64_t, 4>();

  test_unsigned_type<cuda::std::uint8_t, 1>();
  test_unsigned_type<cuda::std::uint8_t, 4>();
  test_unsigned_type<cuda::std::uint16_t, 1>();
  test_unsigned_type<cuda::std::uint16_t, 4>();
  test_unsigned_type<cuda::std::uint32_t, 1>();
  test_unsigned_type<cuda::std::uint32_t, 4>();
  test_unsigned_type<cuda::std::uint64_t, 1>();
  test_unsigned_type<cuda::std::uint64_t, 4>();

  test_constraints();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
