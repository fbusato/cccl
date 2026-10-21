// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename T>
inline constexpr bool is_signed_32_bit_lookback_integer_v =
  ::cuda::std::is_integral_v<T> && ::cuda::std::is_signed_v<T> && sizeof(T) == sizeof(::cuda::std::uint32_t);

template <typename T>
inline constexpr bool is_64_bit_lookback_integer_v =
  ::cuda::std::is_integral_v<T> && sizeof(T) == sizeof(::cuda::std::uint64_t);

// These are performance preferences, not PTX capability restrictions. CUDA 13.4 integration measurements found
// approximately 3% H100 RLE-encode regressions with signed 32-bit look-back components; the ordinary path restored
// baseline performance. The 64-bit value fallback preserves the existing reduce-by-key/RLE look-back behavior;
// it has not been independently retuned for every integer alias or ABI.
// Keep these defaults separate from capability checks: isolated warp results differ from integrated kernels, and
// these choices are not claimed optimal for every caller, architecture, or compiler version.
struct default_lookback_reduce_policy
{
  template <typename T>
  static constexpr bool use_shfl_down_op_for_key = !is_signed_32_bit_lookback_integer_v<T>;

  template <typename T>
  static constexpr bool use_shfl_down_op_for_value =
    !is_signed_32_bit_lookback_integer_v<T> && !is_64_bit_lookback_integer_v<T>;
};

template <bool UseKeyShflDownOp, bool UseValueShflDownOp>
struct lookback_reduce_policy
{
  template <typename>
  static constexpr bool use_shfl_down_op_for_key = UseKeyShflDownOp;

  template <typename>
  static constexpr bool use_shfl_down_op_for_value = UseValueShflDownOp;
};

// Older/custom agent policies need not provide this optional internal tuning alias.
template <typename AgentPolicy, typename DefaultPolicy, typename = void>
struct get_lookback_reduce_policy
{
  using type = DefaultPolicy;
};

template <typename AgentPolicy, typename DefaultPolicy>
struct get_lookback_reduce_policy<AgentPolicy,
                                  DefaultPolicy,
                                  ::cuda::std::void_t<typename AgentPolicy::detail::lookback_reduce_policy>>
{
  using type = typename AgentPolicy::detail::lookback_reduce_policy;
};

template <typename AgentPolicy, typename DefaultPolicy = default_lookback_reduce_policy>
using get_lookback_reduce_policy_t = typename get_lookback_reduce_policy<AgentPolicy, DefaultPolicy>::type;
} // namespace detail

CUB_NAMESPACE_END
