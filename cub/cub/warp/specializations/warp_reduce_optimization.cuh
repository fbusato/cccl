/***********************************************************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_arch.cuh>
#include <cub/warp/specializations/shfl_down_op.cuh>

#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/ptx>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/bit>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

CUB_NAMESPACE_BEGIN

/***********************************************************************************************************************
 * WarpReduce Configuration Enums
 **********************************************************************************************************************/

namespace detail
{

enum class ReduceLogicalMode
{
  SingleReduction,
  MultipleReductions
};

enum class WarpReduceResultMode
{
  AllLanes,
  SingleLane
};

template <ReduceLogicalMode LogicalMode>
using reduce_logical_mode_t = _CUDA_VSTD::integral_constant<ReduceLogicalMode, LogicalMode>;

template <WarpReduceResultMode Kind>
using reduce_result_mode_t = _CUDA_VSTD::integral_constant<WarpReduceResultMode, Kind>;

} // namespace detail

inline constexpr auto single_reduction = detail::reduce_logical_mode_t<detail::ReduceLogicalMode::SingleReduction>{};
inline constexpr auto multiple_reductions =
  detail::reduce_logical_mode_t<detail::ReduceLogicalMode::MultipleReductions>{};

inline constexpr auto all_lanes_result  = detail::reduce_result_mode_t<detail::WarpReduceResultMode::AllLanes>{};
inline constexpr auto first_lane_result = detail::reduce_result_mode_t<detail::WarpReduceResultMode::SingleLane>{};

/***********************************************************************************************************************
 * WarpReduce Base Step
 **********************************************************************************************************************/

namespace detail
{

template <unsigned LogicalWarpSize>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE unsigned shuffle_mask([[maybe_unused]] unsigned step)
{
  if constexpr (_CUDA_VSTD::has_single_bit(LogicalWarpSize))
  {
    const auto clamp   = 1u << step;
    const auto segmask = 0b11110u << (step + 8);
    return clamp | segmask;
  }
  else
  {
    return LogicalWarpSize;
  }
}

template <unsigned LogicalWarpSize, ReduceLogicalMode LogicalMode>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE constexpr unsigned
member_mask(reduce_logical_mode_t<LogicalMode> logical_mode)
{
  static_assert(logical_mode == single_reduction || _CUDA_VSTD::has_single_bit(LogicalWarpSize));
  return (logical_mode == single_reduction) ? (0xFFFFFFFF >> (warp_threads - LogicalWarpSize)) : 0xFFFFFFFF;
}

template <int LogicalWarpSize, typename Input, typename ReductionOp, ReduceLogicalMode LogicalMode, WarpReduceResultMode Kind>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_sm30(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  reduce_result_mode_t<Kind> result_mode)
{
  using namespace internal;
  constexpr bool is_supported_floating_point =
    _CUDA_VSTD::is_floating_point_v<Input>
#if _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 530
    || is_one_of<Input, __half, __half2>()
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 900
    || is_one_of<Input, __nv_bfloat16, __nv_bfloat162>()
#endif // _CCCL_HAS_NVBF16()
    ;
  constexpr auto mask = cub::detail::member_mask<LogicalWarpSize>(logical_mode);
  if constexpr (_CUDA_VSTD::is_same_v<Input, bool>)
  {
    if constexpr (is_cuda_std_plus_v<ReductionOp, Input>)
    {
      return _CUDA_VSTD::popcount(::__ballot_sync(mask, input));
    }
    else if constexpr (is_cuda_std_bit_and_v<ReductionOp, Input>)
    {
      return ::__all_sync(mask, input);
    }
    else if constexpr (is_cuda_std_bit_or_v<ReductionOp, Input>)
    {
      return ::__any_sync(mask, input);
    }
    else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, Input>)
    {
      return _CUDA_VSTD::popcount(::__ballot_sync(mask, input)) % 2u;
    }
    else
    {
      static_assert(_CUDA_VSTD::__always_false_v<Input>, "invalid reduction operator with bool input type");
      _CCCL_UNREACHABLE();
    }
  }
  else if constexpr (_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) < sizeof(uint32_t))
  {
    return warp_reduce_sm30<LogicalWarpSize>(static_cast<int>(input), reduction_op, logical_mode, result_mode);
  }
  else if constexpr ((_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) == sizeof(uint32_t))
                     || is_supported_floating_point)
  {
    constexpr auto Log2Size     = ::cuda::ilog2(LogicalWarpSize * 2 - 1);
    constexpr auto LogicalWidth = _CUDA_VSTD::integral_constant<int, LogicalWarpSize>{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int K = 0; K < Log2Size; K++)
    {
      input = shfl_down_op(reduction_op, input, 1u << K, cub::detail::shuffle_mask<LogicalWarpSize>(K), mask);
    }
    if constexpr (result_mode == all_lanes_result)
    {
      input = _CUDA_VDEV::warp_shuffle_idx(input, 0, mask, LogicalWidth);
    }
    return input;
  }
  else
  {
    static_assert(_CUDA_VSTD::__always_false_v<Input>, "invalid input type/reduction operator combination");
    _CCCL_UNREACHABLE();
  }
}

template <unsigned LogicalWarpSize, typename Input, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_sm80(Input input, ReductionOp)
{
  using namespace internal;
  static_assert(_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) <= sizeof(uint32_t));
  using cast_t        = _CUDA_VSTD::_If<_CUDA_VSTD::is_signed_v<Input>, int32_t, uint32_t>;
  constexpr auto mask = cub::detail::member_mask<LogicalWarpSize>(single_reduction);
  if constexpr (is_cuda_std_bit_and_v<ReductionOp, Input>)
  {
    return static_cast<Input>(::__reduce_and_sync(mask, static_cast<uint32_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_or_v<ReductionOp, Input>)
  {
    return static_cast<Input>(::__reduce_or_sync(mask, static_cast<uint32_t>(input)));
  }
  else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, Input>)
  {
    return static_cast<Input>(::__reduce_xor_sync(mask, static_cast<uint32_t>(input)));
  }
  else if constexpr (is_cuda_std_plus_v<ReductionOp, Input>)
  {
    return ::__reduce_add_sync(mask, static_cast<cast_t>(input));
  }
  else if constexpr (is_cuda_minimum_v<ReductionOp, Input>)
  {
    return ::__reduce_min_sync(mask, static_cast<cast_t>(input));
  }
  else if constexpr (is_cuda_maximum_v<ReductionOp, Input>)
  {
    return ::__reduce_max_sync(mask, static_cast<cast_t>(input));
  }
  else
  {
    static_assert(_CUDA_VSTD::__always_false_v<Input>, "invalid input type/reduction operator combination");
    _CCCL_UNREACHABLE();
  }
}

template <unsigned LogicalWarpSize,
          typename Input,
          typename ReductionOp,
          ReduceLogicalMode LogicalMode,
          WarpReduceResultMode Kind>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_generic(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> logical_mode,
  reduce_result_mode_t<Kind> result_mode)
{
  constexpr auto Log2Size         = ::cuda::ilog2(LogicalWarpSize * 2 - 1);
  constexpr auto mask             = cub::detail::member_mask<LogicalWarpSize>(logical_mode);
  constexpr auto LogicalWarpSize1 = _CUDA_VSTD::has_single_bit(LogicalWarpSize) ? LogicalWarpSize : warp_threads;
  constexpr auto LogicalWidth     = _CUDA_VSTD::integral_constant<int, LogicalWarpSize1>{};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int K = 0; K < Log2Size; K++)
  {
    if constexpr (_CUDA_VSTD::has_single_bit(LogicalWarpSize))
    {
      auto res = _CUDA_VDEV::warp_shuffle_down(input, 1u << K, mask, LogicalWidth);
      if (res.pred)
      {
        input = reduction_op(input, res.data);
      }
    }
    else
    {
      auto lane_id = _CUDA_VPTX::get_sreg_laneid();
      auto dest    = ::min(lane_id + (1u << K), LogicalWarpSize - 1);
      auto res     = _CUDA_VDEV::warp_shuffle_idx(input, dest, mask, LogicalWidth);
      if (lane_id + (1u << K) < LogicalWarpSize)
      {
        input = reduction_op(input, res.data);
      }
    }
  }
  if constexpr (result_mode == all_lanes_result)
  {
    input = _CUDA_VDEV::warp_shuffle_idx(input, 0, LogicalWidth);
  }
  return input;
}

/***********************************************************************************************************************
 * WarpReduce Recursive Step
 **********************************************************************************************************************/

template <int LogicalWarpSize,
          typename Input,
          typename ReductionOp,
          ReduceLogicalMode LogicalMode,
          WarpReduceResultMode ResultMode>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE
Input warp_reduce_dispatch(Input, ReductionOp, reduce_logical_mode_t<LogicalMode>, reduce_result_mode_t<ResultMode>);

template <typename Input>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto split_integers(Input input)
{
  static_assert(_CUDA_VSTD::is_integral_v<Input>);
  constexpr auto half_bits = _CUDA_VSTD::__num_bits_v<Input> / 2;
  using unsigned_t         = _CUDA_VSTD::make_unsigned_t<Input>;
  using half_size_t        = _CUDA_VSTD::__make_nbit_uint_t<half_bits>;
  using output_t           = _CUDA_VSTD::__make_nbit_int_t<half_bits, _CUDA_VSTD::is_signed_v<Input>>;
  auto input1              = static_cast<unsigned_t>(input);
  auto high                = static_cast<half_size_t>(input1 >> half_bits);
  auto low                 = static_cast<half_size_t>(input1);
  return _CUDA_VSTD::array<output_t, 2>{static_cast<output_t>(high), static_cast<output_t>(low)};
}

template <typename Input>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE auto merge_integers(Input inputA, Input inputB)
{
  static_assert(_CUDA_VSTD::is_integral_v<Input>);
  constexpr auto num_bits = _CUDA_VSTD::__num_bits_v<Input>;
  using unsigned_t        = _CUDA_VSTD::__make_nbit_uint_t<num_bits * 2>;
  using output_t          = _CUDA_VSTD::__make_nbit_int_t<num_bits * 2, _CUDA_VSTD::is_signed_v<Input>>;
  return static_cast<output_t>(static_cast<unsigned_t>(inputA) << num_bits | inputB);
}

_CCCL_TEMPLATE(int LogicalWarpSize,
               typename Input,
               typename ReductionOp,
               ReduceLogicalMode LogicalMode,
               WarpReduceResultMode ResultMode)
_CCCL_REQUIRES(cub::internal::is_cuda_std_bitwise_v<ReductionOp, Input> _CCCL_AND(_CUDA_VSTD::is_integral_v<Input>))
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static Input warp_reduce_recursive(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> warp_mode,
  reduce_result_mode_t<ResultMode> result_mode)
{
  using detail::merge_integers;
  using detail::split_integers;
  using detail::warp_reduce_dispatch;
  auto [high, low]    = split_integers(input);
  auto high_reduction = warp_reduce_dispatch<LogicalWarpSize>(high, reduction_op, warp_mode, result_mode);
  auto low_reduction  = warp_reduce_dispatch<LogicalWarpSize>(low, reduction_op, warp_mode, result_mode);
  return merge_integers(high_reduction, low_reduction);
}

_CCCL_TEMPLATE(int LogicalWarpSize,
               typename Input,
               typename ReductionOp,
               ReduceLogicalMode LogicalMode,
               WarpReduceResultMode ResultMode)
_CCCL_REQUIRES(cub::internal::is_cuda_std_min_max_v<ReductionOp, Input> _CCCL_AND(_CUDA_VSTD::is_integral_v<Input>))
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static Input warp_reduce_recursive(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> warp_mode,
  reduce_result_mode_t<ResultMode> result_mode)
{
  using detail::merge_integers;
  using detail::split_integers;
  using detail::warp_reduce_dispatch;
  using internal::identity_v;
  constexpr auto half_bits = _CUDA_VSTD::__num_bits_v<Input> / 2;
  auto [high, low]         = split_integers(input);
  auto high_result         = warp_reduce_dispatch<LogicalWarpSize>(high, reduction_op, warp_mode, result_mode);
  if (high_result == 0) // shortcut: input is in range [0, 2^N/2)
  {
    return warp_reduce_dispatch<LogicalWarpSize>(low, reduction_op, warp_mode, result_mode);
  }
  if (_CUDA_VSTD::is_unsigned_v<Input> || high_result > 0) // >= 2^N/2 -> perform the computation as unsigned
  {
    using half_size_unsigned_t = _CUDA_VSTD::__make_nbit_uint_t<half_bits>;
    constexpr auto identity    = identity_v<ReductionOp, half_size_unsigned_t>;
    auto low_unsigned          = static_cast<half_size_unsigned_t>(low);
    auto low_selected          = high_result == high ? low_unsigned : identity;
    auto low_result = warp_reduce_dispatch<LogicalWarpSize>(low_selected, reduction_op, warp_mode, result_mode);
    return static_cast<Input>(merge_integers(static_cast<half_size_unsigned_t>(high_result), low_result));
  }
  // signed type and < 0
  using half_size_signed_t = _CUDA_VSTD::__make_nbit_int_t<half_bits, true>;
  constexpr auto identity  = identity_v<ReductionOp, half_size_signed_t>;
  auto low_selected        = high_result == high ? static_cast<half_size_signed_t>(low) : identity;
  auto low_result          = warp_reduce_dispatch<LogicalWarpSize>(low_selected, reduction_op, warp_mode, result_mode);
  return merge_integers(static_cast<half_size_signed_t>(high_result), low_result);
}

_CCCL_TEMPLATE(int LogicalWarpSize,
               typename Input,
               typename ReductionOp,
               ReduceLogicalMode LogicalMode,
               WarpReduceResultMode ResultMode)
_CCCL_REQUIRES(cub::internal::is_cuda_std_plus_v<ReductionOp, Input>)
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_recursive(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> warp_mode,
  reduce_result_mode_t<ResultMode> result_mode)
{
  using detail::merge_integers;
  using detail::split_integers;
  using detail::warp_reduce_dispatch;
  using unsigned_t         = _CUDA_VSTD::make_unsigned_t<Input>;
  constexpr auto half_bits = _CUDA_VSTD::__num_bits_v<Input> / 2;
  auto [high, low]         = split_integers(static_cast<unsigned_t>(input));
  auto high_reduction      = warp_reduce_dispatch<LogicalWarpSize>(high, reduction_op, warp_mode, result_mode);
  auto low_digits          = static_cast<uint32_t>(low >> (half_bits - 5));
  auto carry_out           = warp_reduce_dispatch<LogicalWarpSize>(low_digits, reduction_op, warp_mode, result_mode);
  auto low_reduction       = warp_reduce_dispatch<LogicalWarpSize>(low, reduction_op, warp_mode, result_mode);
  auto result_high         = high_reduction + (carry_out >> 5);
  return merge_integers(result_high, low_reduction);
}

/***********************************************************************************************************************
 * WarpReduce Dispatch
 **********************************************************************************************************************/

template <typename T>
inline constexpr bool is_complex_v = _CUDA_VSTD::__is_complex<T>::value;

template <int LogicalWarpSize,
          typename Input,
          typename ReductionOp,
          ReduceLogicalMode LogicalMode,
          WarpReduceResultMode ResultMode>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Input warp_reduce_dispatch(
  Input input,
  ReductionOp reduction_op,
  reduce_logical_mode_t<LogicalMode> warp_mode,
  reduce_result_mode_t<ResultMode> result_mode)
{
  using internal::is_cuda_std_min_max_v;
  using internal::is_cuda_std_plus_v;
  constexpr bool is_small_integer = _CUDA_VSTD::is_integral_v<Input> && sizeof(Input) <= sizeof(uint32_t);
  constexpr bool is_any_floating_point =
    _CUDA_VSTD::is_floating_point_v<Input> || _CUDA_VSTD::__is_extended_floating_point_v<Input>;
  constexpr bool is_supported_floating_point =
    _CUDA_VSTD::is_floating_point_v<Input>
#if _CCCL_HAS_NVFP16() && CUB_PTX_ARCH >= 530
    || is_one_of<Input, __half, __half2>()
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16() && CUB_PTX_ARCH >= 900
    || is_one_of<Input, __nv_bfloat16, __nv_bfloat162>()
#endif // _CCCL_HAS_NVBF16()
    ;
  //
  if constexpr (is_any_floating_point && is_cuda_std_min_max_v<ReductionOp, Input>)
  {
    constexpr auto num_bits = _CUDA_VSTD::__num_bits_v<Input>;
    using signed_t          = _CUDA_VSTD::__make_nbit_int_t<num_bits, true>;
    auto result             = warp_reduce_dispatch<LogicalWarpSize>(
      _CUDA_VSTD::bit_cast<signed_t>(input), reduction_op, warp_mode, result_mode);
    return _CUDA_VSTD::bit_cast<Input>(result);
  }
  else if constexpr (is_complex_v<Input> && is_cuda_std_plus_v<ReductionOp, Input>)
  {
#if _CCCL_HAS_NVFP16()
    if constexpr (_CUDA_VSTD::is_same_v<typename Input::value_type, __half>)
    {
      auto half2_value = unsafe_bitcast<__half2>(input);
      auto ret         = warp_reduce_dispatch<LogicalWarpSize>(half2_value, reduction_op, warp_mode, result_mode);
      return unsafe_bitcast<Input>(ret);
    }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
    if constexpr (_CUDA_VSTD::is_same_v<typename Input::value_type, __nv_bfloat16>)
    {
      auto bfloat2_value = unsafe_bitcast<__nv_bfloat162>(input);
      auto ret           = warp_reduce_dispatch<LogicalWarpSize>(bfloat2_value, reduction_op, warp_mode, result_mode);
      return unsafe_bitcast<Input>(ret);
    }
#endif // _CCCL_HAS_NVBF16()
    {
      auto real = warp_reduce_dispatch<LogicalWarpSize>(input.real(), reduction_op, warp_mode, result_mode);
      auto img  = warp_reduce_dispatch<LogicalWarpSize>(input.imag(), reduction_op, warp_mode, result_mode);
      return Input{real, img};
    }
  }
  else if constexpr (is_small_integer && warp_mode == single_reduction)
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                      (return warp_reduce_sm80<LogicalWarpSize>(input, reduction_op);),
                      (return warp_reduce_sm30<LogicalWarpSize>(input, reduction_op, single_reduction, result_mode);));
  }
  else if constexpr (is_small_integer || is_supported_floating_point)
  {
    return warp_reduce_sm30<LogicalWarpSize>(input, reduction_op, warp_mode, result_mode);
  }
  else if constexpr (_CUDA_VSTD::is_integral_v<Input> && sizeof(Input) > sizeof(uint32_t))
  {
    return warp_reduce_recursive<LogicalWarpSize>(input, reduction_op, warp_mode, result_mode);
  }
  else // generic implementation
  {
    return warp_reduce_generic<LogicalWarpSize>(input, reduction_op, warp_mode, result_mode);
  }
}

} // namespace detail

CUB_NAMESPACE_END
