// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * CUB_NS_QUALIFIER::WarpReduceShfl provides SHFL-based variants of parallel reduction of items partitioned across a
 * CUDA thread warp.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_operators.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/specializations/warp_reduce_shfl_ptx.cuh>
#include <cub/warp/specializations/warp_redux.cuh>

#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__utility/static_for.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename ReductionOp, typename T>
inline constexpr bool is_warp_vote_any_op =
  ::cuda::std::is_same_v<T, bool> && ::cuda::__is_cuda_std_logical_or_v<ReductionOp, T>;

template <typename ReductionOp, typename T>
inline constexpr bool is_warp_vote_all_op =
  ::cuda::std::is_same_v<T, bool> && ::cuda::__is_cuda_std_logical_and_v<ReductionOp, T>;

/**
 * @brief WarpReduceShfl provides SHFL-based variants of parallel reduction of items partitioned
 *        across a CUDA thread warp.
 *
 * @tparam T
 *   Data type being reduced
 *
 * @tparam LogicalWarpThreads
 *   Number of threads per logical warp (must be a power-of-two)
 */
template <typename T, int LogicalWarpThreads>
struct WarpReduceShfl
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");

  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  /// Whether the logical warp size and the PTX warp size coincide
  static constexpr bool IsArchWarp = (LogicalWarpThreads == warp_threads);

  /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
  static constexpr unsigned ShflC = (warp_threads - LogicalWarpThreads) << 8;

  /// Shared memory storage layout type
  using TempStorage = NullType;

  //---------------------------------------------------------------------
  // Thread fields
  //---------------------------------------------------------------------

  /// Lane index in logical warp
  int lane_id;

  /// Logical warp index in 32-thread physical warp
  int warp_id;

  /// 32-thread physical warp member mask of logical warp
  ::cuda::std::uint32_t member_mask;

  //---------------------------------------------------------------------
  // Construction
  //---------------------------------------------------------------------

  /// Constructor
  _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpReduceShfl(TempStorage& /*temp_storage*/)
      : lane_id(static_cast<int>(::cuda::ptx::get_sreg_laneid()))
      , warp_id(IsArchWarp ? 0 : (lane_id / LogicalWarpThreads))
      , member_mask(CUB_NS_QUALIFIER::WarpMask<LogicalWarpThreads>(warp_id))
  {
    if constexpr (!IsArchWarp)
    {
      lane_id = lane_id % LogicalWarpThreads;
    }
  }

  //---------------------------------------------------------------------
  // Reduction steps
  //
  // Each step combines the calling thread's input with the input of the lane `offset` positions down, where
  // `last_lane` is the index of the last lane in the segment.
  //---------------------------------------------------------------------

  // Summation is routed to the PTX specializations. Integers use the unsigned ones: two's complement addition is the
  // same for signed and unsigned types, and the low bits of narrower integers are preserved.
  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
  ReduceStep(T input, ReductionOp reduction_op, int last_lane, int offset)
  {
    if constexpr (is_shfl_down_op_supported_v<ReductionOp, T>)
    {
      constexpr auto plus = ::cuda::std::plus<>{};
      using unsigned_t    = ::cuda::std::conditional_t<sizeof(T) == 8, ::cuda::std::uint64_t, ::cuda::std::uint32_t>;
      using cast_t        = ::cuda::std::conditional_t<::cuda::std::is_floating_point_v<T>, T, unsigned_t>;
      const int shfl_c    = last_lane | ShflC;
      const auto input1   = static_cast<cast_t>(input);
      return static_cast<T>(CUB_NS_QUALIFIER::detail::shfl_down_op(input1, plus, offset, shfl_c, member_mask));
    }
    else
    {
      const auto temp = CUB_NS_QUALIFIER::ShuffleDown<LogicalWarpThreads>(input, offset, last_lane, member_mask);
      return (offset + lane_id <= last_lane) ? reduction_op(input, temp) : input;
    }
  }

  // Applies the reduction steps for offsets 1, 2, ..., LogicalWarpThreads / 2
  template <typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void ShuffleReduce(T& input, ReductionOp reduction_op, int last_lane)
  {
    constexpr int steps = ::cuda::ilog2(LogicalWarpThreads);
    ::cuda::static_for<steps>([&](auto step) {
      input = ReduceStep(input, reduction_op, last_lane, 1u << step);
    });
  }

  //---------------------------------------------------------------------
  // Reduction operations
  //---------------------------------------------------------------------

  /**
   * @brief Reduction
   *
   * @tparam AllValidLanes
   *   Whether all lanes in each warp are contributing a valid fold of items
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] valid_items
   *   Total number of valid items across the logical warp
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   */
  template <bool AllValidLanes, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T input, int valid_items, ReductionOp reduction_op)
  {
    // Dispatch to more efficient intrinsics when applicable
    const int last_lane = (AllValidLanes) ? LogicalWarpThreads - 1 : valid_items - 1;

    if constexpr (is_warp_vote_any_op<ReductionOp, T>)
    {
      return __any_sync(member_mask, input && (AllValidLanes || lane_id <= last_lane));
    }
    else if constexpr (is_warp_vote_all_op<ReductionOp, T>)
    {
      return __all_sync(member_mask, input || (!AllValidLanes && lane_id > last_lane));
    }
    else
    {
      constexpr bool has_identity = ::cuda::has_identity_element_v<ReductionOp, T>;
      T reduce_value              = input;
      if constexpr (IsArchWarp && (AllValidLanes || has_identity) && is_warp_redux_op_supported<ReductionOp, T>)
      {
        if constexpr (!AllValidLanes)
        {
          reduce_value = lane_id <= last_lane ? input : ::cuda::identity_element<ReductionOp, T>();
        }
        if (const auto output = CUB_NS_QUALIFIER::detail::warp_redux(reduce_value, 0xFFFFFFFFu, reduction_op))
        {
          return *output;
        }
      }
      ShuffleReduce(reduce_value, reduction_op, last_lane);
      return reduce_value;
    }
  }

  /**
   * @brief Segmented reduction
   *
   * @tparam HeadSegmented
   *   Whether flags indicate a segment-head or a segment-tail
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] flag
   *   Whether or not the current lane is a segment head/tail
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   */
  template <bool HeadSegmented, typename FlagT, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T SegmentedReduce(T input, FlagT flag, ReductionOp reduction_op)
  {
    // Get the start flags for each thread in the warp.
    unsigned warp_flags = __ballot_sync(member_mask, flag);

    // Convert to tail-segmented
    if constexpr (HeadSegmented)
    {
      warp_flags >>= 1;
    }

    // Mask out the bits below the current thread
    warp_flags &= ::cuda::ptx::get_sreg_lanemask_ge();

    // Mask of physical lanes outside the logical warp and convert to logical lanemask
    if constexpr (!IsArchWarp)
    {
      warp_flags = (warp_flags & member_mask) >> (warp_id * LogicalWarpThreads);
    }

    // Mask in the last lane of logical warp
    warp_flags |= 1u << (LogicalWarpThreads - 1);

    // Find the next set flag
    const int last_lane = ::cuda::std::countr_zero(warp_flags);

    ShuffleReduce(input, reduction_op, last_lane);
    return input;
  }
};
} // namespace detail

CUB_NAMESPACE_END
