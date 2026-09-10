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

// template<simd-size-type N, class V> struct resize;
// template<simd-size-type N, class V> using resize_t = typename resize<N, V>::type;

#include <cuda/std/__simd_>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <complex>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <int N, typename V, typename = void>
inline constexpr bool has_resize = false;

template <int N, typename V>
inline constexpr bool has_resize<N, V, cuda::std::void_t<typename simd::resize<N, V>::type>> = true;

//----------------------------------------------------------------------------------------------------------------------
// resize with basic_vec

template <typename T, int OldN, int NewN>
TEST_HOST_DEVICE_FUNC void test_resize_vec()
{
  using OldVec   = simd::basic_vec<T, simd::fixed_size<OldN>>;
  using Result   = simd::resize_t<NewN, OldVec>;
  using Expected = simd::basic_vec<T, simd::fixed_size<NewN>>;
  static_assert(cuda::std::is_same_v<typename Result::value_type, T>);
  static_assert(Result::size() == NewN);
  static_assert(cuda::std::is_same_v<Result, Expected>);
}

template <typename T>
TEST_HOST_DEVICE_FUNC void test_resize_vec_all()
{
  test_resize_vec<T, 4, 4>();
  test_resize_vec<T, 4, 2>();
  test_resize_vec<T, 2, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// resize with basic_mask

template <typename T, int OldN, int NewN>
TEST_HOST_DEVICE_FUNC void test_resize_mask()
{
  using OldMask  = simd::mask<T, OldN>;
  using Result   = simd::resize_t<NewN, OldMask>;
  using Expected = simd::mask<T, NewN>;
  static_assert(Result::size() == NewN);
  static_assert(cuda::std::is_same_v<Result, Expected>);
}

template <typename T>
TEST_HOST_DEVICE_FUNC void test_resize_mask_all()
{
  test_resize_mask<T, 4, 4>();
  test_resize_mask<T, 4, 2>();
  test_resize_mask<T, 2, 8>();
}

//----------------------------------------------------------------------------------------------------------------------
// resize_t matches resize::type

template <int N, typename V>
TEST_HOST_DEVICE_FUNC void test_resize_t_alias()
{
  static_assert(cuda::std::is_same_v<simd::resize_t<N, V>, typename simd::resize<N, V>::type>);
}

TEST_HOST_DEVICE_FUNC void test()
{
  // resize basic_vec
  test_resize_vec_all<char>();
  test_resize_vec_all<short>();
  test_resize_vec_all<int>();
  test_resize_vec_all<long long>();
  test_resize_vec_all<float>();
  test_resize_vec_all<double>();

  // resize basic_mask
  test_resize_mask_all<char>();
  test_resize_mask_all<int>();
  test_resize_mask_all<double>();
  test_resize_mask_all<cuda::std::complex<float>>();
#if _CCCL_HAS_INT128()
  test_resize_mask_all<cuda::std::complex<double>>();
#endif // _CCCL_HAS_INT128()
#if _CCCL_HAS_HOST_STD_LIB()
  test_resize_mask_all<::std::complex<float>>();
#  if _CCCL_HAS_INT128()
  test_resize_mask_all<::std::complex<double>>();
#  endif // _CCCL_HAS_INT128()
#endif // _CCCL_HOST_LIKE()

  // resize_t alias matches resize::type
  test_resize_t_alias<8, simd::vec<int, 4>>();
  test_resize_t_alias<2, simd::vec<float, 8>>();

  static_assert(has_resize<4, simd::vec<int, 2>>);
  static_assert(has_resize<4, simd::mask<int, 2>>);
  static_assert(!has_resize<0, simd::vec<int, 2>>);
  static_assert(!has_resize<-1, simd::vec<int, 2>>);
  static_assert(!has_resize<65, simd::vec<int, 2>>);
  static_assert(!has_resize<0, simd::mask<int, 2>>);
  static_assert(!has_resize<65, simd::mask<int, 2>>);
  static_assert(!has_resize<4, int>);
  static_assert(!has_resize<4, simd::basic_vec<int, simd::fixed_size<0>>>);
  static_assert(!has_resize<4, simd::basic_mask<sizeof(int), simd::fixed_size<65>>>);
}

int main(int, char**)
{
  test();
  return 0;
}
