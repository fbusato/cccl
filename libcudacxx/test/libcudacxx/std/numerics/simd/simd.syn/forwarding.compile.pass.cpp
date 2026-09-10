//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a host device function in tile mode

// <cuda/simd>

// Test that the required SIMD overloads are forwarded into cuda::std.

#include <cuda/simd>
#include <cuda/std/complex>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

using Vec        = cuda::std::simd::vec<int, 4>;
using Pair       = cuda::std::pair<Vec, Vec>;
using BitVec     = cuda::std::simd::vec<unsigned int, 4>;
using CountVec   = cuda::std::simd::rebind_t<int, BitVec>;
using Complex    = cuda::std::complex<float>;
using ComplexVec = cuda::std::simd::vec<Complex, 4>;
using RealVec    = cuda::std::simd::vec<float, 4>;

static_assert(cuda::std::is_same_v<decltype(cuda::std::min(cuda::std::declval<Vec>(), cuda::std::declval<Vec>())), Vec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::max(cuda::std::declval<Vec>(), cuda::std::declval<Vec>())), Vec>);
static_assert(
  cuda::std::is_same_v<decltype(cuda::std::minmax(cuda::std::declval<Vec>(), cuda::std::declval<Vec>())), Pair>);
static_assert(
  cuda::std::is_same_v<
    decltype(cuda::std::clamp(cuda::std::declval<Vec>(), cuda::std::declval<Vec>(), cuda::std::declval<Vec>())),
    Vec>);

static_assert(cuda::std::is_same_v<decltype(cuda::std::byteswap(cuda::std::declval<BitVec>())), BitVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::bit_reverse(cuda::std::declval<BitVec>())), BitVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::bit_ceil(cuda::std::declval<BitVec>())), BitVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::bit_floor(cuda::std::declval<BitVec>())), BitVec>);
static_assert(
  cuda::std::is_same_v<decltype(cuda::std::has_single_bit(cuda::std::declval<BitVec>())), BitVec::mask_type>);
static_assert(
  cuda::std::is_same_v<decltype(cuda::std::shl(cuda::std::declval<BitVec>(), cuda::std::declval<BitVec>())), BitVec>);
static_assert(
  cuda::std::is_same_v<decltype(cuda::std::shr(cuda::std::declval<BitVec>(), cuda::std::declval<BitVec>())), BitVec>);
static_assert(
  cuda::std::is_same_v<decltype(cuda::std::rotl(cuda::std::declval<BitVec>(), cuda::std::declval<BitVec>())), BitVec>);
static_assert(
  cuda::std::is_same_v<decltype(cuda::std::rotr(cuda::std::declval<BitVec>(), cuda::std::declval<BitVec>())), BitVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::bit_width(cuda::std::declval<BitVec>())), CountVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::countl_zero(cuda::std::declval<BitVec>())), CountVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::countl_one(cuda::std::declval<BitVec>())), CountVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::countr_zero(cuda::std::declval<BitVec>())), CountVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::countr_one(cuda::std::declval<BitVec>())), CountVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::popcount(cuda::std::declval<BitVec>())), CountVec>);

static_assert(cuda::std::is_same_v<decltype(cuda::std::real(cuda::std::declval<ComplexVec>())), RealVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::imag(cuda::std::declval<ComplexVec>())), RealVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::arg(cuda::std::declval<ComplexVec>())), RealVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::norm(cuda::std::declval<ComplexVec>())), RealVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::conj(cuda::std::declval<ComplexVec>())), ComplexVec>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::proj(cuda::std::declval<ComplexVec>())), ComplexVec>);
static_assert(
  cuda::std::is_same_v<decltype(cuda::std::polar(cuda::std::declval<RealVec>(), cuda::std::declval<RealVec>())),
                       ComplexVec>);

int main(int, char**)
{
  return 0;
}
