//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr const_iterator cbegin() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_cbegin()
{
  using CharT = typename SV::value_type;
  using It    = typename SV::const_iterator;

  static_assert(cuda::std::is_same_v<It, decltype(SV{}.cbegin())>);
  static_assert(noexcept(SV{}.cbegin()));

  {
    const CharT* str = TEST_STRLIT(CharT, "");
    SV sv{str};
    assert(cuda::std::addressof(*sv.cbegin()) == str);
  }
  {
    const CharT* str = TEST_STRLIT(CharT, "Hello world!");
    SV sv{str};
    assert(cuda::std::addressof(*sv.cbegin()) == str);
  }
}

__host__ __device__ constexpr bool test()
{
  test_cbegin<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_cbegin<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_cbegin<cuda::std::u16string_view>();
  test_cbegin<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_cbegin<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
