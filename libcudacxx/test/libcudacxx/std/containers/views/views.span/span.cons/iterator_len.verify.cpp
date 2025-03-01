//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

// <span>

// template <class It>
// constexpr explicit(Extent != dynamic_extent) span(It first, size_type count);
//  If Extent is not equal to dynamic_extent, then count shall be equal to Extent.
//

#include <cuda/std/cstddef>
#include <cuda/std/span>

template <class T, size_t extent>
__host__ __device__ cuda::std::span<T, extent> createImplicitSpan(T* ptr, size_t len)
{
  return {ptr, len}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**)
{
  // explicit constructor necessary
  int arr[] = {1, 2, 3};
  createImplicitSpan<int, 1>(arr, 3);

  cuda::std::span<int> sp = {0, 0}; // expected-error {{no matching constructor for initialization of
                                    // 'cuda::std::span<int>'}}
  cuda::std::span<int, 2> sp2 = {0, 0}; // expected-error {{no matching constructor for initialization of
                                        // 'cuda::std::span<int, 2>'}}
  cuda::std::span<const int> csp = {0, 0}; // expected-error {{no matching constructor for initialization of
                                           // 'cuda::std::span<const int>'}}
  cuda::std::span<const int, 2> csp2 = {0, 0}; // expected-error {{no matching constructor for initialization of
                                               // 'cuda::std::span<const int, 2>'}}

  return 0;
}
