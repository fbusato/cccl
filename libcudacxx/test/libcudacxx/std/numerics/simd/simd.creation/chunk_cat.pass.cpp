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

// [simd.creation]
//
// template<class T, class Abi> constexpr auto chunk(const basic_vec<typename T::value_type, Abi>&);
// template<class T, class Abi> constexpr auto chunk(const basic_mask<mask-element-size<T>, Abi>&);
// template<simd-size-type N, class T, class Abi> constexpr auto chunk(const basic_vec<T, Abi>&);
// template<simd-size-type N, size_t Bytes, class Abi> constexpr auto chunk(const basic_mask<Bytes, Abi>&);
// template<class T, class A0, class... Abis> constexpr auto cat(const basic_vec<T,A0>&, const basic_vec<T,Abis>&...);
// template<size_t Bytes, class A0, class... Abis> constexpr auto cat(const basic_mask<Bytes,A0>&,
//                                                                    const basic_mask<Bytes,Abis>&...);

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// generators

template <typename T, int Offset>
struct offset_iota_generator
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I i) const noexcept
  {
    return static_cast<T>(i + Offset);
  }
};

//----------------------------------------------------------------------------------------------------------------------
// chunk<T, Abi>(basic_vec) - exact divisor

template <typename T>
TEST_FUNC constexpr void test_chunk_exact_vec()
{
  using SrcAbi                         = simd::fixed_size<8>;
  using SubVec                         = simd::basic_vec<T, simd::fixed_size<4>>;
  const simd::basic_vec<T, SrcAbi> src = make_iota_vec<T, 8>();

  const auto chunks = simd::chunk<SubVec, SrcAbi>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), const cuda::std::array<SubVec, 2>>);
  for (int i = 0; i < 4; ++i)
  {
    assert(chunks[0][i] == static_cast<T>(i));
    assert(chunks[1][i] == static_cast<T>(i + 4));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<T, Abi>(basic_vec) - remainder

template <typename T>
TEST_FUNC constexpr void test_chunk_remainder_vec()
{
  using SrcAbi                         = simd::fixed_size<6>;
  using SubVec                         = simd::basic_vec<T, simd::fixed_size<4>>;
  using TailVec                        = simd::resize_t<2, SubVec>;
  const simd::basic_vec<T, SrcAbi> src = make_iota_vec<T, 6>();

  const auto chunks = simd::chunk<SubVec, SrcAbi>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), const cuda::std::tuple<SubVec, TailVec>>);

  const auto& head = cuda::std::get<0>(chunks);
  const auto& tail = cuda::std::get<1>(chunks);
  for (int i = 0; i < 4; ++i)
  {
    assert(head[i] == static_cast<T>(i));
  }
  for (int i = 0; i < 2; ++i)
  {
    assert(tail[i] == static_cast<T>(i + 4));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<N>(basic_vec) - N-based delegation

template <typename T>
TEST_FUNC constexpr void test_chunk_by_n_vec()
{
  using SrcAbi                         = simd::fixed_size<8>;
  using SubVec                         = simd::basic_vec<T, simd::fixed_size<2>>;
  const simd::basic_vec<T, SrcAbi> src = make_iota_vec<T, 8>();

  const auto chunks = simd::chunk<2>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), const cuda::std::array<SubVec, 4>>);
  for (int k = 0; k < 4; ++k)
  {
    for (int i = 0; i < 2; ++i)
    {
      assert(chunks[k][i] == static_cast<T>(k * 2 + i));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// cat(basic_vec, basic_vec)

template <typename T>
TEST_FUNC constexpr void test_cat_two_vec()
{
  using Vec4 = simd::basic_vec<T, simd::fixed_size<4>>;
  using Vec2 = simd::basic_vec<T, simd::fixed_size<2>>;
  using Vec6 = simd::basic_vec<T, simd::fixed_size<6>>;

  const Vec4 a(offset_iota_generator<T, 0>{});
  const Vec2 b(offset_iota_generator<T, 100>{});

  const auto result = simd::cat(a, b);
  static_assert(cuda::std::is_same_v<decltype(result), const Vec6>);
  for (int i = 0; i < 4; ++i)
  {
    assert(result[i] == static_cast<T>(i));
  }
  for (int i = 0; i < 2; ++i)
  {
    assert(result[4 + i] == static_cast<T>(i + 100));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// cat(basic_vec, basic_vec, basic_vec)

template <typename T>
TEST_FUNC constexpr void test_cat_three_vec()
{
  using Vec2 = simd::basic_vec<T, simd::fixed_size<2>>;
  using Vec6 = simd::basic_vec<T, simd::fixed_size<6>>;

  const Vec2 a(offset_iota_generator<T, 0>{});
  const Vec2 b(offset_iota_generator<T, 10>{});
  const Vec2 c(offset_iota_generator<T, 20>{});

  const auto result = simd::cat(a, b, c);
  static_assert(cuda::std::is_same_v<decltype(result), const Vec6>);
  for (int i = 0; i < 2; ++i)
  {
    assert(result[i] == static_cast<T>(i));
    assert(result[2 + i] == static_cast<T>(i + 10));
    assert(result[4 + i] == static_cast<T>(i + 20));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<T, Abi>(basic_mask) - exact divisor

template <typename T>
TEST_FUNC constexpr void test_chunk_exact_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using SrcAbi                      = simd::fixed_size<8>;
  using SubMask                     = simd::basic_mask<Bytes, simd::fixed_size<4>>;
  using SrcMask                     = simd::basic_mask<Bytes, SrcAbi>;

  const SrcMask src(is_even{});
  const auto chunks = simd::chunk<SubMask, SrcAbi>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), const cuda::std::array<SubMask, 2>>);
  for (int i = 0; i < 4; ++i)
  {
    assert(chunks[0][i] == (i % 2 == 0));
    assert(chunks[1][i] == ((i + 4) % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<T, Abi>(basic_mask) - remainder

template <typename T>
TEST_FUNC constexpr void test_chunk_remainder_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using SrcAbi                      = simd::fixed_size<6>;
  using SubMask                     = simd::basic_mask<Bytes, simd::fixed_size<4>>;
  using TailMask                    = simd::resize_t<2, SubMask>;
  using SrcMask                     = simd::basic_mask<Bytes, SrcAbi>;

  const SrcMask src(is_even{});
  const auto chunks = simd::chunk<SubMask, SrcAbi>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), const cuda::std::tuple<SubMask, TailMask>>);

  const auto& head = cuda::std::get<0>(chunks);
  const auto& tail = cuda::std::get<1>(chunks);
  for (int i = 0; i < 4; ++i)
  {
    assert(head[i] == (i % 2 == 0));
  }
  for (int i = 0; i < 2; ++i)
  {
    assert(tail[i] == ((i + 4) % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<N>(basic_mask) - N-based delegation

template <typename T>
TEST_FUNC constexpr void test_chunk_by_n_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using SrcAbi                      = simd::fixed_size<8>;
  using SubMask                     = simd::basic_mask<Bytes, simd::fixed_size<2>>;
  using SrcMask                     = simd::basic_mask<Bytes, SrcAbi>;

  const SrcMask src(is_even{});
  const auto chunks = simd::chunk<2>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), const cuda::std::array<SubMask, 4>>);
  for (int k = 0; k < 4; ++k)
  {
    for (int i = 0; i < 2; ++i)
    {
      assert(chunks[k][i] == ((k * 2 + i) % 2 == 0));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// cat(basic_mask, basic_mask) and cat(basic_mask, basic_mask, basic_mask)

template <typename T>
TEST_FUNC constexpr void test_cat_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using Mask4                       = simd::basic_mask<Bytes, simd::fixed_size<4>>;
  using Mask2                       = simd::basic_mask<Bytes, simd::fixed_size<2>>;
  using Mask6                       = simd::basic_mask<Bytes, simd::fixed_size<6>>;

  const Mask4 a(is_even{}); // T,F,T,F
  const Mask2 b(true);
  const auto two = simd::cat(a, b);
  static_assert(cuda::std::is_same_v<decltype(two), const Mask6>);
  for (int i = 0; i < 4; ++i)
  {
    assert(two[i] == (i % 2 == 0));
  }
  for (int i = 0; i < 2; ++i)
  {
    assert(two[4 + i] == true);
  }

  const Mask2 c(false);
  const Mask2 d(true);
  const Mask2 e(false);
  const auto three = simd::cat(c, d, e);
  static_assert(cuda::std::is_same_v<decltype(three), const Mask6>);
  for (int i = 0; i < 2; ++i)
  {
    assert(three[i] == false);
    assert(three[2 + i] == true);
    assert(three[4 + i] == false);
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
TEST_FUNC constexpr void test_type()
{
  test_chunk_exact_vec<T>();
  test_chunk_remainder_vec<T>();
  test_chunk_by_n_vec<T>();
  test_cat_two_vec<T>();
  test_cat_three_vec<T>();
  test_chunk_exact_mask<T>();
  test_chunk_remainder_mask<T>();
  test_chunk_by_n_mask<T>();
  test_cat_mask<T>();
}

TEST_FUNC constexpr bool test()
{
  test_type<cuda::std::int8_t>();
  test_type<cuda::std::int16_t>();
  test_type<cuda::std::int32_t>();
  test_type<cuda::std::int64_t>();
  test_type<cuda::std::uint8_t>();
  test_type<cuda::std::uint32_t>();
  test_type<float>();
  test_type<double>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
