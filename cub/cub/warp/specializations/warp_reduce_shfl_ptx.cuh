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
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename ReductionOp, typename T>
inline constexpr bool is_shfl_down_op_supported_v =
  ::cuda::__is_cuda_std_plus_v<ReductionOp, T>
  && ((::cuda::std::is_floating_point_v<T> && sizeof(T) <= sizeof(double))
      || (::cuda::std::__cccl_is_integer_v<T> && sizeof(T) <= sizeof(::cuda::std::uint64_t)));

#define _CUB_SHFL_DOWN_OP_32BIT(CPP_OPERATOR, TYPE, PTX_OP, CONSTRAINT)                  \
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE TYPE shfl_down_op(                    \
    TYPE input, CPP_OPERATOR, int offset, int shfl_c, ::cuda::std::uint32_t member_mask) \
  {                                                                                      \
    TYPE output;                                                                         \
    asm volatile(                                                                        \
      "{"                                                                                \
      "  .reg .b32 r0;"                                                                  \
      "  .reg .pred p;"                                                                  \
      "  shfl.sync.down.b32 r0|p, %1, %2, %3, %4;"                                       \
      "  @p " PTX_OP " r0, r0, %1;"                                                      \
      "  mov.b32 %0, r0;"                                                                \
      "}"                                                                                \
      : "=" CONSTRAINT(output)                                                           \
      : CONSTRAINT(input), "r"(offset), "r"(shfl_c), "r"(member_mask));                  \
    return output;                                                                       \
  }

#define _CUB_SHFL_DOWN_OP_64BIT(CPP_OPERATOR, TYPE, PTX_OP, CONSTRAINT)                  \
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE TYPE shfl_down_op(                    \
    TYPE input, CPP_OPERATOR, int offset, int shfl_c, ::cuda::std::uint32_t member_mask) \
  {                                                                                      \
    TYPE output;                                                                         \
    asm volatile(                                                                        \
      "{"                                                                                \
      "  .reg .b32 lo;"                                                                  \
      "  .reg .b32 hi;"                                                                  \
      "  .reg .pred p;"                                                                  \
      "  mov.b64 {lo, hi}, %1;"                                                          \
      "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"                                       \
      "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"                                       \
      "  mov.b64 %0, {lo, hi};"                                                          \
      "  @p " PTX_OP " %0, %0, %1;"                                                      \
      "}"                                                                                \
      : "=" CONSTRAINT(output)                                                           \
      : CONSTRAINT(input), "r"(offset), "r"(shfl_c), "r"(member_mask));                  \
    return output;                                                                       \
  }

_CUB_SHFL_DOWN_OP_32BIT(::cuda::std::plus<>, ::cuda::std::uint32_t, "add.u32", "r")
_CUB_SHFL_DOWN_OP_32BIT(::cuda::std::plus<>, float, "add.f32", "f")
_CUB_SHFL_DOWN_OP_64BIT(::cuda::std::plus<>, ::cuda::std::uint64_t, "add.u64", "l")

#undef _CUB_SHFL_DOWN_OP_32BIT
#undef _CUB_SHFL_DOWN_OP_64BIT

// The f64 sum keeps its own PTX ordering: sharing the 64-bit integer form changes the generated SASS
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE double
shfl_down_op(double input, ::cuda::std::plus<>, int offset, int shfl_c, ::cuda::std::uint32_t member_mask)
{
  double output;
  asm volatile(
    "{"
    "  .reg .u32 lo;"
    "  .reg .u32 hi;"
    "  .reg .pred p;"
    "  .reg .f64 r0;"
    "  mov.b64 %0, %1;"
    "  mov.b64 {lo, hi}, %1;"
    "  shfl.sync.down.b32 lo|p, lo, %2, %3, %4;"
    "  shfl.sync.down.b32 hi|p, hi, %2, %3, %4;"
    "  mov.b64 r0, {lo, hi};"
    "  @p add.f64 %0, %0, r0;"
    "}"
    : "=d"(output)
    : "d"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));
  return output;
}
} // namespace detail

CUB_NAMESPACE_END
