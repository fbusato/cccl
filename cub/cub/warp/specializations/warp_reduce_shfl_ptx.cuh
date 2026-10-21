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

#include <cuda/__functional/operator_properties.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
// Integer inputs are converted to a 32-bit or 64-bit unsigned PTX operand before the operation.
// This describes capability; callers separately select whether to use the fused operation.
template <typename ReductionOp, typename T>
inline constexpr bool is_shfl_down_op_supported_v =
  ::cuda::__is_cuda_std_plus_v<ReductionOp, T>
  && (::cuda::std::is_same_v<T, float> || ::cuda::std::is_same_v<T, double>
      || (::cuda::std::__cccl_is_integer_v<T> && sizeof(T) <= sizeof(::cuda::std::uint64_t)));

#define _CUB_SHFL_DOWN_OP_32BIT(CPP_OPERATOR, TYPE, PTX_OP, CONSTRAINT)                                                \
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE TYPE shfl_down_op(                                                  \
    const TYPE input, const CPP_OPERATOR, const int offset, const int shfl_c, const ::cuda::std::uint32_t member_mask) \
  {                                                                                                                    \
    TYPE output;                                                                                                       \
    asm volatile(                                                                                                      \
      "{"                                                                                                              \
      "  .reg .b32 r0;"                                                                                                \
      "  .reg .pred p;"                                                                                                \
      "  shfl.sync.down.b32 r0|p, %1, %2, %3, %4;"                                                                     \
      "  @p " PTX_OP " r0, r0, %1;"                                                                                    \
      "  mov.b32 %0, r0;"                                                                                              \
      "}"                                                                                                              \
      : "=" CONSTRAINT(output)                                                                                         \
      : CONSTRAINT(input), "r"(offset), "r"(shfl_c), "r"(member_mask));                                                \
    return output;                                                                                                     \
  }

#define _CUB_SHFL_DOWN_OP_64BIT(CPP_OPERATOR, TYPE, PTX_OP, CONSTRAINT)                                                \
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE TYPE shfl_down_op(                                                  \
    const TYPE input, const CPP_OPERATOR, const int offset, const int shfl_c, const ::cuda::std::uint32_t member_mask) \
  {                                                                                                                    \
    TYPE output;                                                                                                       \
    asm volatile(                                                                                                      \
      "{"                                                                                                              \
      "  .reg .b32 lo;"                                                                                                \
      "  .reg .b32 hi;"                                                                                                \
      "  .reg .b64 r0;"                                                                                                \
      "  .reg .pred p;"                                                                                                \
      "  mov.b64 %0, %1;"                                                                                              \
      "  mov.b64 {lo, hi}, %1;"                                                                                        \
      "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"                                                                     \
      "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"                                                                     \
      "  mov.b64 r0, {lo, hi};"                                                                                        \
      "  @p " PTX_OP " %0, %0, r0;"                                                                                    \
      "}"                                                                                                              \
      : "=" CONSTRAINT(output)                                                                                         \
      : CONSTRAINT(input), "r"(offset), "r"(shfl_c), "r"(member_mask));                                                \
    return output;                                                                                                     \
  }

_CUB_SHFL_DOWN_OP_32BIT(::cuda::std::plus<>, ::cuda::std::uint32_t, "add.u32", "r")
_CUB_SHFL_DOWN_OP_32BIT(::cuda::std::plus<>, float, "add.f32", "f")
_CUB_SHFL_DOWN_OP_64BIT(::cuda::std::plus<>, ::cuda::std::uint64_t, "add.u64", "l")
_CUB_SHFL_DOWN_OP_64BIT(::cuda::std::plus<>, double, "add.f64", "d")

#undef _CUB_SHFL_DOWN_OP_32BIT
#undef _CUB_SHFL_DOWN_OP_64BIT
} // namespace detail

CUB_NAMESPACE_END
