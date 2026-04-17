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

// [simd.mask.reductions], mask reductions
//
// template<size_t Bytes, class Abi>
//   constexpr bool all_of(const basic_mask<Bytes, Abi>&) noexcept;
// template<size_t Bytes, class Abi>
//   constexpr bool any_of(const basic_mask<Bytes, Abi>&) noexcept;
// template<size_t Bytes, class Abi>
//   constexpr bool none_of(const basic_mask<Bytes, Abi>&) noexcept;
// template<size_t Bytes, class Abi>
//   constexpr simd-size-type reduce_count(const basic_mask<Bytes, Abi>&) noexcept;
// template<size_t Bytes, class Abi>
//   constexpr simd-size-type reduce_min_index(const basic_mask<Bytes, Abi>&);
// template<size_t Bytes, class Abi>
//   constexpr simd-size-type reduce_max_index(const basic_mask<Bytes, Abi>&);
//
// constexpr bool all_of(same_as<bool> auto) noexcept;
// constexpr bool any_of(same_as<bool> auto) noexcept;
// constexpr bool none_of(same_as<bool> auto) noexcept;
// constexpr simd-size-type reduce_count(same_as<bool> auto) noexcept;
// constexpr simd-size-type reduce_min_index(same_as<bool> auto);
// constexpr simd-size-type reduce_max_index(same_as<bool> auto);

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// all_of

template <int Bytes, int N>
TEST_FUNC constexpr void test_all_of()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<decltype(simd::all_of(Mask(true))), bool>);
  static_assert(noexcept(simd::all_of(Mask(true))));

  assert(simd::all_of(Mask(true)) == true);
  assert(simd::all_of(Mask(false)) == false);

  if constexpr (N > 1)
  {
    Mask even(is_even{});
    assert(simd::all_of(even) == false);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// any_of

template <int Bytes, int N>
TEST_FUNC constexpr void test_any_of()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<decltype(simd::any_of(Mask(true))), bool>);
  static_assert(noexcept(simd::any_of(Mask(true))));

  assert(simd::any_of(Mask(true)) == true);
  assert(simd::any_of(Mask(false)) == false);

  if constexpr (N > 1)
  {
    Mask even(is_even{});
    assert(simd::any_of(even) == true);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// none_of

template <int Bytes, int N>
TEST_FUNC constexpr void test_none_of()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<decltype(simd::none_of(Mask(true))), bool>);
  static_assert(noexcept(simd::none_of(Mask(true))));

  assert(simd::none_of(Mask(true)) == false);
  assert(simd::none_of(Mask(false)) == true);

  if constexpr (N > 1)
  {
    Mask even(is_even{});
    assert(simd::none_of(even) == false);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_count

template <int Bytes, int N>
TEST_FUNC constexpr void test_reduce_count()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  static_assert(noexcept(simd::reduce_count(Mask(true))));

  assert(simd::reduce_count(Mask(true)) == N);
  assert(simd::reduce_count(Mask(false)) == 0);

  if constexpr (N > 1)
  {
    Mask even(is_even{});
    int expected = N / 2;
    assert(simd::reduce_count(even) == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_min_index

template <int Bytes, int N>
TEST_FUNC constexpr void test_reduce_min_index()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  assert(simd::reduce_min_index(Mask(true)) == 0);

  if constexpr (N > 1)
  {
    Mask last_only(is_index<N - 1>{});
    assert(simd::reduce_min_index(last_only) == N - 1);

    Mask even(is_even{});
    assert(simd::reduce_min_index(even) == 0);
  }
  if constexpr (N >= 4)
  {
    Mask upper_half(is_greater_equal_than_index<N / 2>{});
    assert(simd::reduce_min_index(upper_half) == N / 2);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_max_index

template <int Bytes, int N>
TEST_FUNC constexpr void test_reduce_max_index()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  assert(simd::reduce_max_index(Mask(true)) == N - 1);

  if constexpr (N > 1)
  {
    Mask first_only(is_index<0>{});
    assert(simd::reduce_max_index(first_only) == 0);

    Mask even(is_even{});
    assert(simd::reduce_max_index(even) == (N - 1) / 2 * 2);
  }

  if constexpr (N >= 4)
  {
    Mask lower_half(is_less_than_index<N / 2>{});
    assert(simd::reduce_max_index(lower_half) == N / 2 - 1);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// all_of / any_of / none_of consistency

template <int Bytes, int N>
TEST_FUNC constexpr void test_consistency()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  Mask all_true(true);
  Mask all_false(false);
  assert(simd::none_of(all_true) == !simd::any_of(all_true));
  assert(simd::none_of(all_false) == !simd::any_of(all_false));

  if constexpr (N > 1)
  {
    Mask even(is_even{});
    assert(simd::none_of(even) == !simd::any_of(even));
    assert(simd::all_of(even) == false);
    assert(simd::any_of(even) == true);
    assert(simd::none_of(even) == false);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// scalar bool overloads

TEST_FUNC constexpr void test_scalar_bool()
{
  static_assert(cuda::std::is_same_v<decltype(simd::all_of(true)), bool>);
  static_assert(cuda::std::is_same_v<decltype(simd::any_of(true)), bool>);
  static_assert(cuda::std::is_same_v<decltype(simd::none_of(true)), bool>);
  static_assert(noexcept(simd::all_of(true)));
  static_assert(noexcept(simd::any_of(true)));
  static_assert(noexcept(simd::none_of(true)));
  static_assert(noexcept(simd::reduce_count(true)));

  assert(simd::all_of(true) == true);
  assert(simd::all_of(false) == false);
  assert(simd::any_of(true) == true);
  assert(simd::any_of(false) == false);
  assert(simd::none_of(true) == false);
  assert(simd::none_of(false) == true);
  assert(simd::reduce_count(true) == 1);
  assert(simd::reduce_count(false) == 0);
  assert(simd::reduce_min_index(true) == 0);
  assert(simd::reduce_max_index(true) == 0);
}

//----------------------------------------------------------------------------------------------------------------------

template <int Bytes, int N>
TEST_FUNC constexpr void test_size()
{
  test_all_of<Bytes, N>();
  test_any_of<Bytes, N>();
  test_none_of<Bytes, N>();
  test_reduce_count<Bytes, N>();
  test_reduce_min_index<Bytes, N>();
  test_reduce_max_index<Bytes, N>();
  test_consistency<Bytes, N>();
}

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_size<Bytes, 1>();
  test_size<Bytes, 4>();
  test_size<Bytes, 8>();
}

TEST_FUNC constexpr bool test()
{
  test_bytes<1>();
  test_bytes<2>();
  test_bytes<4>();
  test_bytes<8>();
#if _CCCL_HAS_INT128()
  test_bytes<16>();
#endif
  test_scalar_bool();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
