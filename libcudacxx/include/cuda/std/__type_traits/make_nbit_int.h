//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_MAKE_NBIT_INT_H
#define _LIBCUDACXX___TYPE_TRAITS_MAKE_NBIT_INT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__utility/unreachable.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _NBits, bool _IsSigned>
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto __make_nbit_int_impl() noexcept
{
  if constexpr (_IsSigned)
  {
    if constexpr (_NBits == 8)
    {
      return int8_t{};
    }
    else if constexpr (_NBits == 16)
    {
      return int16_t{};
    }
    else if constexpr (_NBits == 32)
    {
      return int32_t{};
    }
    else if constexpr (_NBits == 64)
    {
      return int64_t{};
    }
#if _CCCL_HAS_INT128()
    else if constexpr (_NBits == 128)
    {
      return __int128_t{};
    }
#endif // _CCCL_HAS_INT128()
    else
    {
      static_assert(_CCCL_TRAIT(__always_false, decltype(_NBits)), "Unsupported signed integer size");
      _CUDA_VSTD::unreachable();
    }
  }
  else
  {
    if constexpr (_NBits == 8)
    {
      return uint8_t{};
    }
    else if constexpr (_NBits == 16)
    {
      return uint16_t{};
    }
    else if constexpr (_NBits == 32)
    {
      return uint32_t{};
    }
    else if constexpr (_NBits == 64)
    {
      return uint64_t{};
    }
#if _CCCL_HAS_INT128()
    else if constexpr (_NBits == 128)
    {
      return __uint128_t{};
    }
#endif // _CCCL_HAS_INT128()
    else
    {
      static_assert(_CCCL_TRAIT(__always_false, decltype(_NBits)), "Unsupported unsigned integer size");
      _CUDA_VSTD::unreachable();
    }
  }
}

template <size_t _NBytes, bool _IsSigned = true>
using __make_nbit_int_t = decltype(__make_nbit_int_impl<_NBytes, _IsSigned>());

template <size_t _NBytes>
using __make_nbit_uint_t = __make_nbit_int_t<_NBytes, false>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_MAKE_NBIT_INT_H
