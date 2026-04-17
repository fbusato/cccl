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

// [simd.iterator], begin/end on basic_vec and basic_mask
//
// constexpr iterator begin() noexcept;
// constexpr const_iterator begin() const noexcept;
// constexpr const_iterator cbegin() const noexcept;
// constexpr default_sentinel_t end() const noexcept;
// constexpr default_sentinel_t cend() const noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// type aliases and begin/end on basic_vec

template <typename T, int N>
TEST_FUNC constexpr void test_vec_types()
{
  using Vec       = simd::basic_vec<T, simd::fixed_size<N>>;
  using Iter      = typename Vec::iterator;
  using ConstIter = typename Vec::const_iterator;

  static_assert(cuda::std::is_same_v<typename Iter::value_type, T>);
  static_assert(cuda::std::is_same_v<typename Iter::iterator_category, cuda::std::input_iterator_tag>);
  static_assert(cuda::std::is_same_v<typename Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
  static_assert(cuda::std::is_same_v<typename Iter::difference_type, cuda::std::ptrdiff_t>);
  static_assert(cuda::std::is_same_v<typename ConstIter::value_type, T>);

  Vec vec{};
  const Vec const_vec{};
  unused(vec, const_vec);

  static_assert(cuda::std::is_same_v<decltype(vec.begin()), Iter>);
  static_assert(cuda::std::is_same_v<decltype(vec.end()), cuda::std::default_sentinel_t>);
  static_assert(cuda::std::is_same_v<decltype(const_vec.begin()), ConstIter>);
  static_assert(cuda::std::is_same_v<decltype(const_vec.cbegin()), ConstIter>);
  static_assert(cuda::std::is_same_v<decltype(const_vec.end()), cuda::std::default_sentinel_t>);
  static_assert(cuda::std::is_same_v<decltype(const_vec.cend()), cuda::std::default_sentinel_t>);

  static_assert(noexcept(vec.begin()));
  static_assert(noexcept(const_vec.begin()));
  static_assert(noexcept(const_vec.cbegin()));
  static_assert(noexcept(const_vec.end()));
  static_assert(noexcept(const_vec.cend()));
}

//----------------------------------------------------------------------------------------------------------------------
// type aliases and begin/end on basic_mask

template <typename T, int N>
TEST_FUNC constexpr void test_mask_types()
{
  using Vec       = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask      = typename Vec::mask_type;
  using Iter      = typename Mask::iterator;
  using ConstIter = typename Mask::const_iterator;

  static_assert(cuda::std::is_same_v<typename Iter::value_type, bool>);
  static_assert(cuda::std::is_same_v<typename Iter::iterator_category, cuda::std::input_iterator_tag>);
  static_assert(cuda::std::is_same_v<typename Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
  static_assert(cuda::std::is_same_v<typename Iter::difference_type, cuda::std::ptrdiff_t>);
  static_assert(cuda::std::is_same_v<typename ConstIter::value_type, bool>);

  Mask mask(true);
  const Mask const_mask(true);
  unused(mask, const_mask);

  static_assert(cuda::std::is_same_v<decltype(mask.begin()), Iter>);
  static_assert(cuda::std::is_same_v<decltype(mask.end()), cuda::std::default_sentinel_t>);
  static_assert(cuda::std::is_same_v<decltype(const_mask.begin()), ConstIter>);
  static_assert(cuda::std::is_same_v<decltype(const_mask.cbegin()), ConstIter>);
  static_assert(cuda::std::is_same_v<decltype(const_mask.end()), cuda::std::default_sentinel_t>);
  static_assert(cuda::std::is_same_v<decltype(const_mask.cend()), cuda::std::default_sentinel_t>);

  static_assert(noexcept(mask.begin()));
  static_assert(noexcept(const_mask.begin()));
  static_assert(noexcept(const_mask.cbegin()));
  static_assert(noexcept(const_mask.end()));
  static_assert(noexcept(const_mask.cend()));
}

//----------------------------------------------------------------------------------------------------------------------
// const iterator conversion

template <typename T, int N>
TEST_FUNC constexpr void test_const_conversion()
{
  using Vec       = simd::basic_vec<T, simd::fixed_size<N>>;
  using ConstIter = typename Vec::const_iterator;

  Vec vec = make_iota_vec<T, N>();

  auto it            = vec.begin();
  ConstIter const_it = it;
  assert(*const_it == *it);
  assert(*const_it == static_cast<T>(0));

  const Vec const_vec{};
  ConstIter const_it2 = const_vec.begin();
  ConstIter const_it3 = const_vec.cbegin();
  assert(const_it2 == const_it3);
}

//----------------------------------------------------------------------------------------------------------------------
// basic_vec iteration

template <typename T, int N>
TEST_FUNC constexpr void test_vec_iteration()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec   = make_iota_vec<T, N>();

  int count = 0;
  for (auto it = vec.begin(); it != cuda::std::default_sentinel; ++it)
  {
    assert(*it == static_cast<T>(count));
    ++count;
  }
  assert(count == N);
}

//----------------------------------------------------------------------------------------------------------------------
// basic_mask begin/end and iteration

template <typename T, int N>
TEST_FUNC constexpr void test_mask_iteration()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  Mask mask(true);
  int count = 0;
  for (auto it = mask.begin(); it != cuda::std::default_sentinel; ++it)
  {
    assert(*it == true);
    ++count;
  }
  assert(count == N);

  Mask mask_false(false);
  for (auto it = mask_false.begin(); it != cuda::std::default_sentinel; ++it)
  {
    assert(*it == false);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// range-based for loop

template <typename T, int N>
TEST_FUNC constexpr void test_range_based_for()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;

  Vec vec   = make_iota_vec<T, N>();
  int count = 0;
  for (auto val : vec)
  {
    assert(val == static_cast<T>(count));
    ++count;
  }
  assert(count == N);

  Mask mask(true);
  count = 0;
  for (auto val : mask)
  {
    assert(val == true);
    ++count;
  }
  assert(count == N);
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_vec_types<T, N>();
  test_mask_types<T, N>();
  test_const_conversion<T, N>();
  test_vec_iteration<T, N>();
  test_mask_iteration<T, N>();
  test_range_based_for<T, N>();
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
