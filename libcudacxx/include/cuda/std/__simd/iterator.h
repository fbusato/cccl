//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_ITERATOR_H
#define _CUDA_STD___SIMD_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_const.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.iterator], class template __simd_iterator
template <typename _Vp>
class __simd_iterator
{
  _Vp* __data_               = nullptr;
  __simd_size_type __offset_ = 0;

  _CCCL_API constexpr __simd_iterator(_Vp& __d, __simd_size_type __off) noexcept
      : __data_{::cuda::std::addressof(__d)}
      , __offset_{__off}
  {}

  template <typename, typename>
  friend class basic_vec;

  template <size_t, typename>
  friend class basic_mask;

  template <typename>
  friend class __simd_iterator;

public:
  using value_type        = typename _Vp::value_type;
  using iterator_category = input_iterator_tag;
  using iterator_concept  = random_access_iterator_tag;
  using difference_type   = __simd_size_type;

  _CCCL_HIDE_FROM_ABI constexpr __simd_iterator() noexcept                                  = default;
  _CCCL_HIDE_FROM_ABI constexpr __simd_iterator(const __simd_iterator&) noexcept            = default;
  _CCCL_HIDE_FROM_ABI constexpr __simd_iterator& operator=(const __simd_iterator&) noexcept = default;

  // non-const to const converting constructor
  _CCCL_TEMPLATE(typename _Up = remove_const_t<_Vp>)
  _CCCL_REQUIRES(is_const_v<_Vp> _CCCL_AND is_same_v<_Up, remove_const_t<_Vp>>)
  _CCCL_API constexpr __simd_iterator(const __simd_iterator<_Up>& __i)
      : __data_{__i.__data_}
      , __offset_{__i.__offset_}
  {}

  [[nodiscard]] _CCCL_API constexpr value_type operator*() const
  {
    return (*__data_)[__offset_];
  }

  _CCCL_API constexpr __simd_iterator& operator++()
  {
    return *this += 1;
  }

  _CCCL_API constexpr __simd_iterator operator++(int)
  {
    __simd_iterator __tmp = *this;
    *this += 1;
    return __tmp;
  }

  _CCCL_API constexpr __simd_iterator& operator--()
  {
    return *this -= 1;
  }

  _CCCL_API constexpr __simd_iterator operator--(int)
  {
    __simd_iterator __tmp = *this;
    *this -= 1;
    return __tmp;
  }

  _CCCL_API constexpr __simd_iterator& operator+=(const difference_type __n)
  {
    __offset_ += __n;
    return *this;
  }

  _CCCL_API constexpr __simd_iterator& operator-=(const difference_type __n)
  {
    __offset_ -= __n;
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr value_type operator[](const difference_type __n) const
  {
    return (*__data_)[__offset_ + __n];
  }

  // [simd.iterator] comparisons

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __simd_iterator& __a, const __simd_iterator& __b)
  {
    return __a.__data_ == __b.__data_ && __a.__offset_ == __b.__offset_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __simd_iterator __i, const default_sentinel_t) noexcept
  {
    return __i.__offset_ == _Vp::__size;
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __simd_iterator& __a, const __simd_iterator& __b)
  {
    return !(__a == __b);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const __simd_iterator __i, const ::cuda::std::default_sentinel_t __s) noexcept
  {
    return !(__i == __s);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const ::cuda::std::default_sentinel_t __s, const __simd_iterator __i) noexcept
  {
    return !(__i == __s);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const ::cuda::std::default_sentinel_t __s, const __simd_iterator __i) noexcept
  {
    return __i == __s;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<(const __simd_iterator& __a, const __simd_iterator& __b)
  {
    return __a.__offset_ < __b.__offset_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const __simd_iterator& __a, const __simd_iterator& __b)
  {
    return __b < __a;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const __simd_iterator& __a, const __simd_iterator& __b)
  {
    return !(__b < __a);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const __simd_iterator& __a, const __simd_iterator& __b)
  {
    return !(__a < __b);
  }
#else // ^^^ C++17 ^^^ / vvv C++20+ vvv
  [[nodiscard]] _CCCL_API friend constexpr auto operator<=>(const __simd_iterator __a, const __simd_iterator __b)
  {
    return __a.__offset_ <=> __b.__offset_;
  }
#endif // C++20+

  // [simd.iterator] arithmetic

  [[nodiscard]] _CCCL_API friend constexpr __simd_iterator operator+(__simd_iterator __i, const difference_type __n)
  {
    return __i += __n;
  }

  [[nodiscard]] _CCCL_API friend constexpr __simd_iterator operator+(const difference_type __n, __simd_iterator __i)
  {
    return __i += __n;
  }

  [[nodiscard]] _CCCL_API friend constexpr __simd_iterator operator-(__simd_iterator __i, const difference_type __n)
  {
    return __i -= __n;
  }

  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const __simd_iterator __a, const __simd_iterator __b)
  {
    return __a.__offset_ - __b.__offset_;
  }

  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const __simd_iterator __i, default_sentinel_t) noexcept
  {
    return __i.__offset_ - _Vp::__size;
  }

  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(default_sentinel_t, const __simd_iterator __i) noexcept
  {
    return _Vp::__size - __i.__offset_;
  }
};

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_ITERATOR_H
