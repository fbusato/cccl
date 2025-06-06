//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr counted_iterator& operator--()
//  requires bidirectional_iterator<I>;
// constexpr counted_iterator operator--(int)
//  requires bidirectional_iterator<I>;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
_CCCL_CONCEPT MinusEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)((iter--), (--iter));

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using Counted = cuda::std::counted_iterator<bidirectional_iterator<int*>>;
    cuda::std::counted_iterator iter(bidirectional_iterator<int*>{buffer + 2}, 6);
    assert(iter-- == Counted(bidirectional_iterator<int*>{buffer + 2}, 6));
    assert(--iter == Counted(bidirectional_iterator<int*>{buffer}, 8));
    assert(iter.count() == 8);

    static_assert(cuda::std::is_same_v<decltype(iter--), Counted>);
    static_assert(cuda::std::is_same_v<decltype(--iter), Counted&>);
  }
  {
    using Counted = cuda::std::counted_iterator<random_access_iterator<int*>>;
    Counted iter(random_access_iterator<int*>{buffer + 2}, 6);
    assert(iter-- == Counted(random_access_iterator<int*>{buffer + 2}, 6));
    assert(--iter == Counted(random_access_iterator<int*>{buffer}, 8));
    assert(iter.count() == 8);

    static_assert(cuda::std::is_same_v<decltype(iter--), Counted>);
    static_assert(cuda::std::is_same_v<decltype(--iter), Counted&>);
  }
  {
    using Counted = cuda::std::counted_iterator<contiguous_iterator<int*>>;
    cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer + 2}, 6);
    assert(iter-- == Counted(contiguous_iterator<int*>{buffer + 2}, 6));
    assert(--iter == Counted(contiguous_iterator<int*>{buffer}, 8));
    assert(iter.count() == 8);

    static_assert(cuda::std::is_same_v<decltype(iter--), Counted>);
    static_assert(cuda::std::is_same_v<decltype(--iter), Counted&>);
  }

  {
    static_assert(MinusEnabled<cuda::std::counted_iterator<contiguous_iterator<int*>>>);
    static_assert(!MinusEnabled<const cuda::std::counted_iterator<contiguous_iterator<int*>>>);
    static_assert(!MinusEnabled<cuda::std::counted_iterator<forward_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
