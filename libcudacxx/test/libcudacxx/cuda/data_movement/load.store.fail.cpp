//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/data_movement>
#include <cuda/ptx>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include "test_macros.h"

__device__ cuda::std::array<uint32_t, 7> pointer1;
__device__ uint32_t pointer2[16];

// non power of 2 and properties
__global__ void load_kernel()
{
  unused(cuda::device::load(&pointer1, cuda::device::read_only));
  unused(cuda::device::load(&pointer1, cuda::device::read_write, cuda::device::eviction_unchanged));
  unused(
    cuda::device::load(&pointer1, cuda::device::read_write, cuda::device::eviction_none, cuda::device::prefetch_64B));
}

__global__ void store_kernel()
{
  cuda::std::array<uint32_t, 7> array{};
  cuda::device::store(array, &pointer1, cuda::device::eviction_unchanged); // not a multiple of 16 with eviction

  cuda::std::array<const uint32_t, 1> array1{};
  cuda::device::store(array1, reinterpret_cast<const uint32_t*>(&pointer2)); // const pointer

  cuda::std::array<uint32_t, 0> array2;
  cuda::device::store(array2, pointer2); // 0 size
}

__global__ void load_array_kernel()
{
  unused(cuda::device::load<0>(pointer2, cuda::aligned_size_t<4>{4})); // 0 size
  unused(cuda::device::load<2>(pointer2, cuda::aligned_size_t<6>{6})); // non power of 2
  unused(cuda::device::load<2>(pointer2, cuda::aligned_size_t<2>{2})); // alignment too small
  unused(cuda::device::load<3>(pointer2, cuda::aligned_size_t<8>{8})); // sizeof(T) * N is not a multiple of alignment
}

__global__ void store_array_kernel()
{
  cuda::std::array<uint32_t, 2> array1{};
  cuda::std::array<uint32_t, 3> array2{};
  cuda::device::store(array1, pointer2, cuda::aligned_size_t<6>{6}); // non power of 2
  cuda::device::store(array1, pointer2, cuda::aligned_size_t<2>{2}); // alignment too small
  cuda::device::store(array2, pointer2, cuda::aligned_size_t<8>{8}); // sizeof(T) * N is not a multiple of alignment
}

// the compiler gets stuck on this one
//__global__ void store_span_kernel()
//{
//  cuda::std::array<uint32_t, 2> array{};
//  cuda::std::span<uint32_t, cuda::std::dynamic_extent> span{array};
//  cuda::device::store(span, pointer2); // dynamic_extent
//}

//----------------------------------------------------------------------------------------------------------------------

int main(int, char**)
{
  return 0;
}
