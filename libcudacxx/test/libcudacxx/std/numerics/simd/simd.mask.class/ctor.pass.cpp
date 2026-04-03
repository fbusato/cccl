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

// [simd.mask.ctor], basic_mask constructors
//
// constexpr explicit basic_mask(value_type) noexcept;                          // broadcast
// constexpr explicit basic_mask(const basic_mask<UBytes, UAbi>&) noexcept;     // converting
// constexpr explicit basic_mask(Generator&&);                                  // generator
// constexpr basic_mask(const bitset<size()>&) noexcept;                        // bitset
// constexpr explicit basic_mask(unsigned-integer) noexcept;                    // unsigned integer

#include <cuda/std/bitset>
#include <cuda/std/type_traits>

#include "mask_test_utils.h"

//----------------------------------------------------------------------------------------------------------------------
// member types and size

template <int Bytes, int N>
__host__ __device__ constexpr void test_member_types()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;

  static_assert(cuda::std::is_same_v<typename Mask::value_type, bool>);
  static_assert(cuda::std::is_same_v<typename Mask::abi_type, simd::fixed_size<N>>);
  static_assert(Mask::size() == N);
}

//----------------------------------------------------------------------------------------------------------------------
// broadcast constructor

template <int Bytes, int N>
__host__ __device__ constexpr void test_broadcast()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  static_assert(noexcept(Mask(true)));

  Mask all_true(true);
  Mask all_false(false);
  for (int i = 0; i < N; ++i)
  {
    assert(all_true[i] == true);
    assert(all_false[i] == false);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// converting constructor

template <int Bytes, int UBytes, int N>
__host__ __device__ constexpr void test_converting()
{
  using Src = simd::basic_mask<UBytes, simd::fixed_size<N>>;
  using Dst = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Src src(is_even{});
  static_assert(noexcept(Dst(src)));

  Dst dst(src);
  for (int i = 0; i < N; ++i)
  {
    assert(dst[i] == src[i]);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// generator constructor

template <int Bytes, int N>
__host__ __device__ constexpr void test_generator()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  static_assert(!noexcept(Mask(is_even{})));

  Mask mask(is_even{});
  for (int i = 0; i < N; ++i)
  {
    assert(mask[i] == (i % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// bitset constructor

template <int Bytes, int N>
__host__ __device__ constexpr void test_bitset()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  cuda::std::bitset<N> bitset;
  static_assert(noexcept(Mask(bitset)));

  for (int i = 0; i < N; ++i)
  {
    bitset.set(i, (i % 2 == 0));
  }
  Mask mask(bitset);
  for (int i = 0; i < N; ++i)
  {
    assert(mask[i] == (i % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// unsigned integer constructor

template <int Bytes, int N, typename U>
__host__ __device__ constexpr void test_unsigned_int()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  static_assert(noexcept(Mask(U{0})));

  Mask mask(U{0});
  for (int i = 0; i < N; ++i)
  {
    assert(mask[i] == false);
  }

  constexpr int num_bits  = cuda::std::__num_bits_v<U>;
  constexpr int mask_bits = cuda::std::min(N, num_bits);
  Mask all_one(static_cast<U>(~U{0}));
  for (int i = 0; i < mask_bits; ++i)
  {
    assert(all_one[i] == true);
  }

  if constexpr (N >= 3 && num_bits >= 3)
  {
    Mask m_pat(U{0b101});
    assert(m_pat[0] == true);
    assert(m_pat[1] == false);
    assert(m_pat[2] == true);
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <int Bytes, int N>
__host__ __device__ constexpr void test_size()
{
  test_member_types<Bytes, N>();
  test_broadcast<Bytes, N>();
  test_generator<Bytes, N>();
  test_bitset<Bytes, N>();
  test_unsigned_int<Bytes, N, cuda::std::uint8_t>();
  test_unsigned_int<Bytes, N, cuda::std::uint16_t>();
  test_unsigned_int<Bytes, N, cuda::std::uint32_t>();
  test_unsigned_int<Bytes, N, cuda::std::uint64_t>();
}

template <int Bytes>
__host__ __device__ constexpr void test_bytes()
{
  test_size<Bytes, 1>();
  test_size<Bytes, 4>();
}

__host__ __device__ constexpr bool test()
{
  test_bytes<1>();
  test_bytes<4>();

  // test_converting  N1: Destination type size, N2: Source type size, N3: Mask number of elements
  test_converting<4, 2, 4>(); // 4 -> 2, 4 elements
  test_converting<2, 4, 4>(); // 2 -> 4, 4 elements
  test_converting<1, 8, 4>(); // 1 -> 8, 4 elements
  test_converting<8, 1, 4>(); // 8 -> 1, 4 elements
  test_converting<4, 4, 4>(); // 4 -> 4, 4 elements

  test_converting<1, 2, 1>(); // 1 -> 2, 1 element
  test_converting<1, 2, 2>(); // 1 -> 2, 2 elements
  test_converting<1, 2, 8>(); // 1 -> 2, 8 elements
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
