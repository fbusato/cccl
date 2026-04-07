//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_AGGREGATE_MEMBERS_H
#define _CUDA_STD___TYPE_TRAITS_AGGREGATE_MEMBERS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_aggregate.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-field-initializers")

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_STRUCTURED_BINDING_SIZE)

// Returns the number of aggregate members, or `-1` if the type is not an aggregate.
template <typename _Tp, ::cuda::std::enable_if_t<_CCCL_BUILTIN_STRUCTURED_BINDING_SIZE(_Tp) >= 0, int> = 0>
constexpr int __aggregate_arity_v = _CCCL_BUILTIN_STRUCTURED_BINDING_SIZE(_Tp);

#else // ^^^ _CCCL_BUILTIN_STRUCTURED_BINDING_SIZE ^^^ / !_CCCL_BUILTIN_STRUCTURED_BINDING_SIZE vvv

// provide a generic way to initialize an aggregate member
struct __any_aggregate_member
{
  template <typename _Tp>
  _CCCL_API constexpr operator _Tp&&() const;
};

template <typename _Tp, bool = is_aggregate_v<_Tp>>
struct __aggregate_arity_impl
{
  template <typename... _Args,
            typename _Up   = _Tp,
            typename       = decltype(_Up{_Args{}...}), // SFINAE on args number
            typename _Self = __aggregate_arity_impl>
  _CCCL_API auto operator()(_Args... __args) -> decltype(_Self{}(__args..., __any_aggregate_member{}));

  template <typename... _Args>
  _CCCL_API auto operator()(_Args...) const -> char (*)[sizeof...(_Args) + 1]; // return the number of members + 1
};

// T is not an aggregate, return 1
template <typename _Tp>
struct __aggregate_arity_impl<_Tp, false>
{
  _CCCL_API auto operator()() const -> char*;
};

// Returns the number of aggregate members, or `-1` if the type is not an aggregate.
template <typename _Tp>
constexpr int __aggregate_arity_v = int{sizeof(*__aggregate_arity_impl<_Tp>{}())} - 2;

#endif // ^^^ !_CCCL_BUILTIN_STRUCTURED_BINDING_SIZE ^^^

// Apply a Predicate to every aggregate member

// provide a generic way to initialize an aggregate member but only if the Predicate is true
template <template <typename> class _Predicate>
struct __aggregate_member_if
{
  template <typename _Tp, typename = enable_if_t<_Predicate<remove_cvref_t<_Tp>>::value>>
  _CCCL_API constexpr operator _Tp&&() const;
};

template <int _Arity>
struct __aggregate_all_of_fn;

// T has 0 members, return true
template <>
struct __aggregate_all_of_fn<0>
{
  template <template <typename> class _Predicate, typename _Tp>
  _CCCL_API static auto __call(int) -> true_type;
};

#define _CCCL_AGGR_PROBE(_POS) \
  , __aggregate_member_if<_Predicate> {}

// T has N members, return true if the Predicate is true for all members (recursively)
#define _CCCL_AGGREGATE_ALL_OF_CASE(_NP)                                                                               \
  template <>                                                                                                          \
  struct __aggregate_all_of_fn<1 + _NP>                                                                                \
  {                                                                                                                    \
    template <template <typename> class _Predicate,                                                                    \
              typename _Tp,                                                                                            \
              typename _Up = _Tp,                                                                                      \
              typename = decltype(_Up{__aggregate_member_if<_Predicate>{} _CCCL_PP_REPEAT(_NP, _CCCL_AGGR_PROBE, 0)})> \
    _CCCL_API static auto __call(int) -> true_type;                                                                    \
                                                                                                                       \
    template <template <typename> class _Predicate, typename _Tp>                                                      \
    _CCCL_API static auto __call(...) -> false_type;                                                                   \
  }

inline constexpr int __aggregate_max_arity = 16;

_CCCL_AGGREGATE_ALL_OF_CASE(0);
_CCCL_AGGREGATE_ALL_OF_CASE(1);
_CCCL_AGGREGATE_ALL_OF_CASE(2);
_CCCL_AGGREGATE_ALL_OF_CASE(3);
_CCCL_AGGREGATE_ALL_OF_CASE(4);
_CCCL_AGGREGATE_ALL_OF_CASE(5);
_CCCL_AGGREGATE_ALL_OF_CASE(6);
_CCCL_AGGREGATE_ALL_OF_CASE(7);
_CCCL_AGGREGATE_ALL_OF_CASE(8);
_CCCL_AGGREGATE_ALL_OF_CASE(9);
_CCCL_AGGREGATE_ALL_OF_CASE(10);
_CCCL_AGGREGATE_ALL_OF_CASE(11);
_CCCL_AGGREGATE_ALL_OF_CASE(12);
_CCCL_AGGREGATE_ALL_OF_CASE(13);
_CCCL_AGGREGATE_ALL_OF_CASE(14);
_CCCL_AGGREGATE_ALL_OF_CASE(15);

#undef _CCCL_AGGREGATE_ALL_OF_CASE
#undef _CCCL_AGGR_PROBE

// return true if
// - T is an aggregate
// - T has a number of members between 0 and __aggregate_max_arity
// - T is not empty
template <template <typename> class _Predicate,
          typename _Tp,
          bool = is_aggregate_v<_Tp> && (__aggregate_arity_v<_Tp> >= 0)
              && (__aggregate_arity_v<_Tp> <= __aggregate_max_arity)
              && ((__aggregate_arity_v<_Tp> != 0) || is_empty_v<_Tp>)>
struct __aggregate_all_of : false_type
{};

// Applies a Predicate to every member reachable by aggregate initialization
template <template <typename> class _Predicate, typename _Tp>
struct __aggregate_all_of<_Predicate, _Tp, true>
    : decltype(__aggregate_all_of_fn<__aggregate_arity_v<_Tp>>::template __call<_Predicate, _Tp>(0))
{};

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_AGGREGATE_MEMBERS_H
