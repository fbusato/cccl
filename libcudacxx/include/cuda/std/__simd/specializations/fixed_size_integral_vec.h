//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_INTEGRAL_VEC_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_INTEGRAL_VEC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__simd/specializations/fixed_size_vec.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// Simd operations for fixed_size ABI with small integral element types.
template <typename _Tp, __simd_size_type _Np>
struct __simd_operations<_Tp, __fixed_size<_Np>, enable_if_t<__is_fixed_size_small_integral_v<_Tp, _Np>>>
    : __fixed_size_operations<_Tp, _Np>
{
  using _SimdStorage = __simd_storage<_Tp, __fixed_size<_Np>>;
  using _MaskStorage = __mask_storage<sizeof(_Tp), __fixed_size<_Np>>;

  static constexpr auto __ratio         = sizeof(unsigned) / sizeof(_Tp);
  static constexpr auto __usize         = _Np / __ratio;
  static constexpr auto __storage_bytes = __usize * sizeof(unsigned);
  using __unsigned_storage_t            = ::cuda::std::array<unsigned, __usize>;

  static constexpr auto __storage_alignment = alignof(_Tp) * _Np;

  [[nodiscard]] _CCCL_API static __unsigned_storage_t __to_unsigned_storage(const _SimdStorage& __s) noexcept
  {
    alignas(__storage_alignment) __unsigned_storage_t __output_data{};
    const auto __input_data = ::cuda::std::assume_aligned<__storage_alignment>(&__s);
    ::cuda::std::memcpy(__output_data.data(), __input_data, __storage_bytes);
    return __output_data;
  }

  [[nodiscard]] _CCCL_API static _SimdStorage __copy_from_unsigned_storage(const __unsigned_storage_t& __udata) noexcept
  {
    alignas(__storage_alignment) _SimdStorage __result;
    ::cuda::std::memcpy(__result.__data, __udata.data(), __storage_bytes);
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __bitwise_not(const _SimdStorage& __s) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      auto __udata = __to_unsigned_storage(__s);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __usize; ++__i)
      {
        __udata[__i] = ~__udata[__i];
      }
      _SimdStorage __result = __copy_from_unsigned_storage(__udata);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (auto __i = __usize * __ratio; __i < _Np; ++__i)
      {
        __result.__data[__i] = ~__s.__data[__i];
      }
      return __result;
    }
    return __fixed_size_operations<_Tp, _Np>::__bitwise_not(__s);
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_and(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      __unsigned_storage_t __result_udata{};
      const auto __lhs_udata = __to_unsigned_storage(__lhs);
      const auto __rhs_udata = __to_unsigned_storage(__rhs);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __usize; ++__i)
      {
        __result_udata[__i] = __lhs_udata[__i] & __rhs_udata[__i];
      }
      _SimdStorage __result = __copy_from_unsigned_storage(__result_udata);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (auto __i = __usize * __ratio; __i < _Np; ++__i)
      {
        __result.__data[__i] = (__lhs.__data[__i] & __rhs.__data[__i]);
      }
      return __result;
    }
    return __fixed_size_operations<_Tp, _Np>::__bitwise_and(__lhs, __rhs);
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_or(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      __unsigned_storage_t __result_udata{};
      const auto __lhs_udata = __to_unsigned_storage(__lhs);
      const auto __rhs_udata = __to_unsigned_storage(__rhs);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __usize; ++__i)
      {
        __result_udata[__i] = __lhs_udata[__i] | __rhs_udata[__i];
      }
      _SimdStorage __result = __copy_from_unsigned_storage(__result_udata);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (auto __i = __usize * __ratio; __i < _Np; ++__i)
      {
        __result.__data[__i] = (__lhs.__data[__i] | __rhs.__data[__i]);
      }
      return __result;
    }
    return __fixed_size_operations<_Tp, _Np>::__bitwise_or(__lhs, __rhs);
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_xor(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      __unsigned_storage_t __result_udata{};
      const auto __lhs_udata = __to_unsigned_storage(__lhs);
      const auto __rhs_udata = __to_unsigned_storage(__rhs);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __usize; ++__i)
      {
        __result_udata[__i] = __lhs_udata[__i] ^ __rhs_udata[__i];
      }
      _SimdStorage __result = __copy_from_unsigned_storage(__result_udata);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (auto __i = __usize * __ratio; __i < _Np; ++__i)
      {
        __result.__data[__i] = (__lhs.__data[__i] ^ __rhs.__data[__i]);
      }
      return __result;
    }
    return __fixed_size_operations<_Tp, _Np>::__bitwise_xor(__lhs, __rhs);
  }

  // Binary arithmetic operations

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __plus(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      const auto __lhs_udata = __to_unsigned_storage(__lhs);
      const auto __rhs_udata = __to_unsigned_storage(__rhs);
      __unsigned_storage_t __result_udata{};
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __usize; ++__i)
      {
        if constexpr (is_same_v<_Tp, uint16_t> || is_same_v<_Tp, int16_t>)
        {
          __result_udata[__i] = ::__vadd2(__lhs_udata[__i], __rhs_udata[__i]);
        }
        else if constexpr (is_same_v<_Tp, uint8_t> || is_same_v<_Tp, int8_t>)
        {
          __result_udata[__i] = ::__vadd4(__lhs_udata[__i], __rhs_udata[__i]);
        }
      }
      _SimdStorage __result = __copy_from_unsigned_storage(__result_udata);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (auto __i = __usize * __ratio; __i < _Np; ++__i)
      {
        __result.__data[__i] = (__lhs.__data[__i] + __rhs.__data[__i]);
      }
      return __result;
    }
    return __fixed_size_operations<_Tp, _Np>::__plus(__lhs, __rhs);
  }
};

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_FIXED_SIZE_INTEGRAL_VEC_H
