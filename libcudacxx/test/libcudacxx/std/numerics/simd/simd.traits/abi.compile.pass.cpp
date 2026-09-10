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

// <cuda/std/__simd_>

// template<class T, simd-size-type N> using deduce-abi-t = ...;
// template<class T> using native-abi = ...;

#include <cuda/std/__simd_>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

namespace simd = cuda::std::simd;

using ValidAbi = simd::__deduce_abi_t<int, 4>;
static_assert(cuda::std::is_same_v<ValidAbi, simd::fixed_size<4>>);
static_assert(simd::__is_enabled_abi_v<ValidAbi>);
static_assert(simd::__simd_size_v<int, ValidAbi> == 4);
static_assert(simd::__mask_size_v<sizeof(int), ValidAbi> == 4);

using InvalidTypeAbi = simd::__deduce_abi_t<long double, 4>;
using ZeroSizeAbi    = simd::__deduce_abi_t<int, 0>;
using NegativeAbi    = simd::__deduce_abi_t<int, -1>;
using OversizeAbi    = simd::__deduce_abi_t<int, 65>;

static_assert(!simd::__is_enabled_abi_v<InvalidTypeAbi>);
static_assert(!simd::__is_enabled_abi_v<ZeroSizeAbi>);
static_assert(!simd::__is_enabled_abi_v<NegativeAbi>);
static_assert(!simd::__is_enabled_abi_v<OversizeAbi>);

static_assert(simd::__simd_size_v<long double, simd::fixed_size<4>> == 0);
static_assert(simd::__simd_size_v<int, simd::fixed_size<0>> == 0);
static_assert(simd::__simd_size_v<int, simd::fixed_size<-1>> == 0);
static_assert(simd::__simd_size_v<int, simd::fixed_size<65>> == 0);
static_assert(simd::__mask_size_v<sizeof(int), simd::fixed_size<0>> == 0);
static_assert(simd::__mask_size_v<sizeof(int), simd::fixed_size<65>> == 0);

static_assert(cuda::std::is_same_v<simd::native<int>, simd::fixed_size<1>>);
static_assert(simd::__simd_size_v<int, simd::native<int>> == 1);
static_assert(cuda::std::is_same_v<simd::__simd_complex_value_type_t<simd::vec<cuda::std::complex<float>, 4>>, float>);

int main(int, char**)
{
  return 0;
}
