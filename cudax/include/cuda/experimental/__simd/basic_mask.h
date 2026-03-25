//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_BASIC_MASK_H
#define _CUDAX___SIMD_BASIC_MASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/array>
#include <cuda/std/bitset>
#include <cuda/std/cstdint>

#include <cuda/experimental/__simd/declaration.h>
#include <cuda/experimental/__simd/utility.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::simd
{
// [simd.mask.class], class template basic_mask
template <::cuda::std::size_t _Bytes, typename _Abi>
class basic_mask : public __mask_operations<_Bytes, _Abi>
{
  static_assert(_Bytes >= 0, "basic_mask requires a positive number of bytes");
  static_assert(__is_abi_tag_v<_Abi>, "basic_mask requires a valid ABI tag");

  using _Impl    = __mask_operations<_Bytes, _Abi>;
  using _Storage = typename _Impl::_MaskStorage;

  _Storage __s_;

  struct __storage_tag_t
  {};
  static constexpr __storage_tag_t __storage_tag{};

  _CCCL_API constexpr basic_mask(_Storage __v, __storage_tag_t) noexcept
      : __s_{__v}
  {}

public:
  using value_type = bool;
  using abi_type   = _Abi;

  // TODO(fbusato): add simd-iterator
  // using iterator       = simd-iterator<basic_mask>;
  // using const_iterator = simd-iterator<const basic_mask>;

  // constexpr iterator begin() noexcept { return {*this, 0}; }
  // constexpr const_iterator begin() const noexcept { return {*this, 0}; }
  // constexpr const_iterator cbegin() const noexcept { return {*this, 0}; }
  // constexpr default_sentinel_t end() const noexcept { return {}; }
  // constexpr default_sentinel_t cend() const noexcept { return {}; }

  static constexpr ::cuda::std::integral_constant<__simd_size_type, __simd_size_v<__integer_from<_Bytes>, _Abi>> size{};

  static constexpr auto __usize = ::cuda::std::size_t{size};

  _CCCL_HIDE_FROM_ABI basic_mask() noexcept = default;

  // [simd.mask.ctor], basic_mask constructors

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::same_as<_Up, value_type>)
  _CCCL_API constexpr explicit basic_mask(_Up __v) noexcept
      : __s_{_Impl::__broadcast(__v)}
  {}

  _CCCL_TEMPLATE(::cuda::std::size_t _UBytes, typename _UAbi)
  _CCCL_REQUIRES((__simd_size_v<__integer_from<_UBytes>, _UAbi> == size()))
  _CCCL_API constexpr explicit basic_mask(const basic_mask<_UBytes, _UAbi>& __x) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < size; ++__i)
    {
      __s_.__set(__i, __x[__i]);
    }
  }

  _CCCL_TEMPLATE(typename _Generator)
  _CCCL_REQUIRES(__can_generate_v<bool, _Generator, size>)
  _CCCL_API constexpr explicit basic_mask(_Generator&& __g)
      : __s_{_Impl::__generate(__g)}
  {}

  _CCCL_TEMPLATE(typename _Tp)
  _CCCL_REQUIRES(::cuda::std::same_as<_Tp, ::cuda::std::bitset<__usize>>)
  _CCCL_API constexpr basic_mask(const _Tp& __b) noexcept
      : __s_{_Impl::__broadcast(false)}
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < size; ++__i)
    {
      __s_.__set(__i, static_cast<bool>(__b[__i]));
    }
  }

  _CCCL_TEMPLATE(typename _Tp)
  _CCCL_REQUIRES((::cuda::std::__cccl_is_unsigned_integer_v<_Tp> && !::cuda::std::same_as<_Tp, value_type>) )
  _CCCL_API constexpr explicit basic_mask(_Tp __val) noexcept
      : __s_{_Impl::__broadcast(false)}
  {
    constexpr auto __num_bits    = __simd_size_type{::cuda::std::__num_bits_v<_Tp>};
    constexpr auto __size_as_int = size();
    constexpr auto __m           = __size_as_int < __num_bits ? __size_as_int : __num_bits;
    using __uint8_array_t        = ::cuda::std::array<::cuda::std::uint8_t, sizeof(_Tp)>;
    const auto __val1            = ::cuda::std::bit_cast<__uint8_array_t>(__val);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < __m; ++__i)
    {
      const auto __byte = __val1[__i / CHAR_BIT];
      __s_.__set(__i, static_cast<bool>((__byte >> (__i % CHAR_BIT)) & _Tp{1}));
    }
  }

  // [simd.mask.subscr], basic_mask subscript operators

  [[nodiscard]] _CCCL_API constexpr value_type operator[](__simd_size_type __i) const noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__i, __simd_size_type{0}, __simd_size_type{size}), "Index is out of bounds");
    return static_cast<bool>(__s_.__get(__i));
  }

  // TODO(fbusato): subscript with integral indices, requires permute()
  // template<simd-integral I>
  // constexpr resize_t<I::size(), basic_mask> operator[](const I& indices) const;

  // [simd.mask.unary], basic_mask unary operators

  [[nodiscard]] _CCCL_API constexpr basic_mask operator!() const noexcept
  {
    return {_Impl::__bitwise_not(__s_), __storage_tag};
  }

  _CCCL_TEMPLATE(::cuda::std::size_t _B = _Bytes)
  _CCCL_REQUIRES(__has_integer_from_v<_B>)
  [[nodiscard]] _CCCL_API constexpr basic_vec<__integer_from<_B>, _Abi> operator+() const noexcept
  {
    return static_cast<basic_vec<__integer_from<_B>, _Abi>>(*this);
  }

  _CCCL_TEMPLATE(::cuda::std::size_t _B = _Bytes)
  _CCCL_REQUIRES((!__has_integer_from_v<_B>) )
  _CCCL_API void operator+() const noexcept = delete;

  _CCCL_TEMPLATE(::cuda::std::size_t _B = _Bytes)
  _CCCL_REQUIRES(__has_integer_from_v<_B>)
  [[nodiscard]] _CCCL_API constexpr basic_vec<__integer_from<_B>, _Abi> operator-() const noexcept
  {
    return -static_cast<basic_vec<__integer_from<_B>, _Abi>>(*this);
  }

  _CCCL_TEMPLATE(::cuda::std::size_t _B = _Bytes)
  _CCCL_REQUIRES((!__has_integer_from_v<_B>) )
  _CCCL_API void operator-() const noexcept = delete;

  _CCCL_TEMPLATE(::cuda::std::size_t _B = _Bytes)
  _CCCL_REQUIRES(__has_integer_from_v<_B>)
  [[nodiscard]] _CCCL_API constexpr basic_vec<__integer_from<_B>, _Abi> operator~() const noexcept
  {
    return ~static_cast<basic_vec<__integer_from<_B>, _Abi>>(*this);
  }

  _CCCL_TEMPLATE(::cuda::std::size_t _B = _Bytes)
  _CCCL_REQUIRES((!__has_integer_from_v<_B>) )
  _CCCL_API void operator~() const noexcept = delete;

  // [simd.mask.conv], basic_mask conversions

  _CCCL_TEMPLATE(typename _Up, typename _Ap)
  _CCCL_REQUIRES((sizeof(_Up) != _Bytes && __simd_size_v<_Up, _Ap> == size()))
  _CCCL_API constexpr explicit operator basic_vec<_Up, _Ap>() const noexcept
  {
    basic_vec<_Up, _Ap> __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < size; ++__i)
    {
      __result[__i] = static_cast<_Up>((*this)[__i]);
    }
    return __result;
  }

  _CCCL_TEMPLATE(typename _Up, typename _Ap)
  _CCCL_REQUIRES((sizeof(_Up) == _Bytes && __simd_size_v<_Up, _Ap> == size()))
  _CCCL_API constexpr operator basic_vec<_Up, _Ap>() const noexcept
  {
    basic_vec<_Up, _Ap> __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < size; ++__i)
    {
      __result[__i] = static_cast<_Up>((*this)[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API constexpr ::cuda::std::bitset<__usize> to_bitset() const noexcept
  {
    ::cuda::std::bitset<__usize> __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (__simd_size_type __i = 0; __i < size; ++__i)
    {
      __result.set(__i, (*this)[__i]);
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API constexpr unsigned long long to_ullong() const
  {
    constexpr __simd_size_type __nbits = ::cuda::std::__num_bits_v<unsigned long long>;
    if constexpr (size > __nbits)
    {
      for (auto __i = __nbits; __i < size; ++__i)
      {
        _CCCL_ASSERT(!(*this)[__i], "Bit above unsigned long long width is set");
      }
    }
    return to_bitset().to_ullong();
  }

  // [simd.mask.binary], basic_mask binary operators

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator&&(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__logic_and(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator||(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__logic_or(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator&(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__bitwise_and(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator|(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__bitwise_or(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator^(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return {_Impl::__bitwise_xor(__lhs.__s_, __rhs.__s_), __storage_tag};
  }

  // [simd.mask.cassign], basic_mask compound assignment

  _CCCL_API friend constexpr basic_mask& operator&=(basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs = __lhs & __rhs;
  }

  _CCCL_API friend constexpr basic_mask& operator|=(basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs = __lhs | __rhs;
  }

  _CCCL_API friend constexpr basic_mask& operator^=(basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs = __lhs ^ __rhs;
  }

  // [simd.mask.comparison], basic_mask comparisons (element-wise)

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator==(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return !(__lhs ^ __rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator!=(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs ^ __rhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator>=(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs || !__rhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator<=(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return !__lhs || __rhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator>(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return __lhs && !__rhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr basic_mask
  operator<(const basic_mask& __lhs, const basic_mask& __rhs) noexcept
  {
    return !__lhs && __rhs;
  }

  // TODO(fbusato): [simd.mask.cond], basic_mask exposition only conditional operators
  // friend constexpr basic_mask __simd_select_impl(
  //   const basic_mask&, const basic_mask&, const basic_mask&) noexcept;
  // friend constexpr basic_mask __simd_select_impl(
  //   const basic_mask&, same_as<bool> auto, same_as<bool> auto) noexcept;
  // template<class T0, class T1>
  //   friend constexpr vec<see below, size()> __simd_select_impl(
  //     const basic_mask&, const T0&, const T1&) noexcept;
};

// [simd.mask.reductions], reductions

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr bool all_of(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  using __mask_storage_t = typename __mask_operations<_Bytes, _Abi>::_MaskStorage;
  return __mask_operations<_Bytes, _Abi>::__all(static_cast<__mask_storage_t>(__k));
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr bool any_of(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  using __mask_storage_t = typename __mask_operations<_Bytes, _Abi>::_MaskStorage;
  return __mask_operations<_Bytes, _Abi>::__any(static_cast<__mask_storage_t>(__k));
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr bool none_of(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  return !::cuda::experimental::simd::any_of(__k);
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_count(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  using __mask_storage_t = typename __mask_operations<_Bytes, _Abi>::_MaskStorage;
  return __mask_operations<_Bytes, _Abi>::__count(static_cast<__mask_storage_t>(__k));
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_min_index(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  _CCCL_ASSERT(any_of(__k), "No bits are set");
  using __mask_storage_t = typename __mask_operations<_Bytes, _Abi>::_MaskStorage;
  return __mask_operations<_Bytes, _Abi>::__min_index(static_cast<__mask_storage_t>(__k));
}

template <::cuda::std::size_t _Bytes, typename _Abi>
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_max_index(const basic_mask<_Bytes, _Abi>& __k) noexcept
{
  _CCCL_ASSERT(any_of(__k), "No bits are set");
  using __mask_storage_t = typename __mask_operations<_Bytes, _Abi>::_MaskStorage;
  return __mask_operations<_Bytes, _Abi>::__max_index(static_cast<__mask_storage_t>(__k));
}

// Scalar bool overloads

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr bool all_of(_Tp __x) noexcept
{
  return __x;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr bool any_of(_Tp __x) noexcept
{
  return __x;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr bool none_of(_Tp __x) noexcept
{
  return !__x;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_count(_Tp __x) noexcept
{
  return __x;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)
[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_min_index(_Tp __x) noexcept
{
  _CCCL_ASSERT(__x, "No bits are set");
  return 0;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::same_as<_Tp, bool>)

[[nodiscard]] _CCCL_API constexpr __simd_size_type reduce_max_index(_Tp __x) noexcept
{
  _CCCL_ASSERT(__x, "No bits are set");
  return 0;
}
} // namespace cuda::experimental::simd

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_BASIC_MASK_H
