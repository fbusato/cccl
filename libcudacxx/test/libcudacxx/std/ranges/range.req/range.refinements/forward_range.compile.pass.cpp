//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// template<class R>
// concept forward_range;

#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_range.h"

template <template <class...> class I>
__host__ __device__ constexpr bool check_forward_range()
{
  constexpr bool result = cuda::std::ranges::forward_range<test_range<I>>;
  static_assert(cuda::std::ranges::forward_range<test_range<I> const> == result, "");
  static_assert(cuda::std::ranges::forward_range<test_non_const_common_range<I>> == result, "");
  static_assert(cuda::std::ranges::forward_range<test_non_const_range<I>> == result, "");
  static_assert(cuda::std::ranges::forward_range<test_common_range<I>> == result, "");
  static_assert(cuda::std::ranges::forward_range<test_common_range<I> const> == result, "");
  static_assert(!cuda::std::ranges::forward_range<test_non_const_common_range<I> const>, "");
  static_assert(!cuda::std::ranges::forward_range<test_non_const_range<I> const>, "");
  return result;
}

static_assert(!check_forward_range<cpp17_input_iterator>(), "");
static_assert(!check_forward_range<cpp20_input_iterator>(), "");
static_assert(check_forward_range<forward_iterator>(), "");
static_assert(check_forward_range<bidirectional_iterator>(), "");
static_assert(check_forward_range<random_access_iterator>(), "");
static_assert(check_forward_range<contiguous_iterator>(), "");

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

static_assert(!cuda::std::ranges::forward_range<Holder<Incomplete>*>, "");
static_assert(!cuda::std::ranges::forward_range<Holder<Incomplete>*&>, "");
static_assert(!cuda::std::ranges::forward_range<Holder<Incomplete>*&&>, "");
static_assert(!cuda::std::ranges::forward_range<Holder<Incomplete>* const>, "");
static_assert(!cuda::std::ranges::forward_range<Holder<Incomplete>* const&>, "");
static_assert(!cuda::std::ranges::forward_range<Holder<Incomplete>* const&&>, "");

static_assert(cuda::std::ranges::forward_range<Holder<Incomplete>* [10]>, "");
static_assert(cuda::std::ranges::forward_range<Holder<Incomplete>* (&) [10]>, "");
static_assert(cuda::std::ranges::forward_range<Holder<Incomplete>* (&&) [10]>, "");
static_assert(cuda::std::ranges::forward_range<Holder<Incomplete>* const[10]>, "");
static_assert(cuda::std::ranges::forward_range<Holder<Incomplete>* const (&)[10]>, "");
static_assert(cuda::std::ranges::forward_range<Holder<Incomplete>* const (&&)[10]>, "");
#endif

int main(int, char**)
{
  return 0;
}
