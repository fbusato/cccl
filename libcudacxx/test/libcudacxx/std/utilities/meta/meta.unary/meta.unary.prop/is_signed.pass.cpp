//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_signed

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_signed()
{
  static_assert(cuda::std::is_signed<T>::value, "");
  static_assert(cuda::std::is_signed<const T>::value, "");
  static_assert(cuda::std::is_signed<volatile T>::value, "");
  static_assert(cuda::std::is_signed<const volatile T>::value, "");
  static_assert(cuda::std::is_signed_v<T>, "");
  static_assert(cuda::std::is_signed_v<const T>, "");
  static_assert(cuda::std::is_signed_v<volatile T>, "");
  static_assert(cuda::std::is_signed_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_signed()
{
  static_assert(!cuda::std::is_signed<T>::value, "");
  static_assert(!cuda::std::is_signed<const T>::value, "");
  static_assert(!cuda::std::is_signed<volatile T>::value, "");
  static_assert(!cuda::std::is_signed<const volatile T>::value, "");
  static_assert(!cuda::std::is_signed_v<T>, "");
  static_assert(!cuda::std::is_signed_v<const T>, "");
  static_assert(!cuda::std::is_signed_v<volatile T>, "");
  static_assert(!cuda::std::is_signed_v<const volatile T>, "");
}

class Class
{
public:
  __host__ __device__ ~Class();
};

struct A; // incomplete

int main(int, char**)
{
  test_is_not_signed<void>();
  test_is_not_signed<int&>();
  test_is_not_signed<Class>();
  test_is_not_signed<int*>();
  test_is_not_signed<const int*>();
  test_is_not_signed<char[3]>();
  test_is_not_signed<char[]>();
  test_is_not_signed<bool>();
  test_is_not_signed<unsigned>();
  test_is_not_signed<A>();

  test_is_signed<int>();
  test_is_signed<double>();

#if _CCCL_HAS_INT128()
  test_is_signed<__int128_t>();
  test_is_not_signed<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return 0;
}
