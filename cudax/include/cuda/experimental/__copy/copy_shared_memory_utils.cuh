//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_COPY_SHARED_MEMORY_UTILS_H
#define _CUDAX__COPY_COPY_SHARED_MEMORY_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/devices>
#  include <cuda/std/__algorithm/any_of.h>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/array>

#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Count the number of leading contiguous dimensions in a raw tensor.
//!
//! Starting from dimension 0, counts consecutive dimensions where `stride[0] == 1` and `stride[i] == stride[i-1] *
//! extent[i-1]` for each subsequent dimension.
//!
//! @param[in] __tensor Raw tensor descriptor
//! @return Number of leading contiguous dimensions (0 if stride[0] != 1 or rank is 0)
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t
__num_contiguous_dimensions(const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>;
  using __rank_t       = typename __raw_tensor_t::__rank_t;
  if (__tensor.__rank == 0 || __tensor.__strides[0] != 1)
  {
    return 0;
  }
  __rank_t __count       = 1;
  auto __expected_stride = static_cast<_StrideT>(__tensor.__extents[0]);
  for (__rank_t __i = 1; __i < __tensor.__rank; ++__i)
  {
    if (__tensor.__strides[__i] != __expected_stride)
    {
      break;
    }
    __expected_stride *= static_cast<_StrideT>(__tensor.__extents[__i]);
    ++__count;
  }
  return __count;
}

//! @brief Return a device_ref for the current CUDA device.
//!
//! @return Device reference for the active CUDA context's device
[[nodiscard]] _CCCL_HOST_API inline device_ref __current_device()
{
  const auto __dev_id = ::cuda::__driver::__cudevice_to_ordinal(::cuda::__driver::__ctxGetDevice());
  return ::cuda::devices[__dev_id];
}

inline constexpr ::cuda::std::size_t __max_tile_size = 32; // warp-size

//! @brief Decide whether the shared-memory tiled transpose kernel is profitable.
//!
//! Returns true when the destination has stride-1 in mode 0, the source does not, there are at least two contiguous
//! destination dimensions, the resulting tile is large enough to amortize the shared-memory overhead, and the total
//! number of tiles is sufficient to utilize the GPU (at least one full wave across all SMs).
//!
//! @param[in] __src Source raw tensor descriptor
//! @param[in] __dst Destination raw tensor descriptor
//! @return true if the shared-memory kernel should be used
template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API bool
__use_shared_mem_kernel(const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _MaxRank>& __src,
                        const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _MaxRank>& __dst)
{
  using ::cuda::std::size_t;
  using __rank_t                = typename __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _MaxRank>::__rank_t;
  const size_t __num_contiguous = ::cuda::experimental::__num_contiguous_dimensions(__dst);
  // * destination is contiguous (in dimension 0) -> coalesced destination writes
  // * source is not contiguous (already excluded by vectorized copy)
  // * there are at least two contiguous destination dimensions -> otherwise, direct copy is better
  if (__src.__strides[0] == 1 || __dst.__strides[0] != 1 || __num_contiguous < 2)
  {
    return false;
  }
  // * source has at least one dimension with extent not equal to 1 -> otherwise, shared memory makes no sense
  const auto __ext_begin          = __src.__extents.cbegin();
  const bool __has_non_one_extent = ::cuda::std::any_of(__ext_begin, __ext_begin + __src.__rank, [](auto __extent) {
    return __extent != 1;
  });
  if (!__has_non_one_extent)
  {
    return false;
  }
  // * there are at least two contiguous destination dimensions -> otherwise, direct copy is better
  // * the tile is large enough to benefits from coalesced memory accesses
  const auto __current_dev            = ::cuda::experimental::__current_device();
  const size_t __max_shared_mem_bytes = __current_dev.attribute<::cudaDevAttrMaxSharedMemoryPerBlock>();
  size_t __size_product               = 1;
  int __tile_rank                     = 0;
  for (size_t __r = 0; __r < __num_contiguous; ++__r, ++__tile_rank)
  {
    const auto __tile_size_r = ::cuda::std::min(static_cast<size_t>(__dst.__extents[__r]), __max_tile_size);
    if (__size_product * __tile_size_r * sizeof(_TpOut) > __max_shared_mem_bytes)
    {
      break;
    }
    __size_product *= __tile_size_r;
  }
  if (__tile_rank < 2 || __size_product < __max_tile_size * 8)
  {
    return false;
  }
  // * there are enough tiles to keep the GPU busy (at least one full wave across all SMs)
  size_t __num_tiles = 1;
  for (__rank_t __r = 0; __r < __dst.__rank; ++__r)
  {
    const auto __extent    = static_cast<size_t>(__dst.__extents[__r]);
    const auto __tile_size = (__r < __tile_rank) ? ::cuda::std::min(__extent, __max_tile_size) : size_t{1};
    __num_tiles *= ::cuda::ceil_div(__extent, __tile_size);
  }
  const size_t __num_sms = __current_dev.attribute<::cudaDevAttrMultiProcessorCount>();
  return __num_tiles >= __num_sms;
}

using __tile_extent_t = unsigned;

//! @brief Compute the shared-memory tile sizes for a destination tensor.
//!
//! Greedily expands the tile across contiguous dimensions up to the warp-size cap per dimension and the device
//! shared-memory limit. Dimensions beyond the tile rank are set to extent 1.
//!
//! @param[in]  __tensor         Destination raw tensor descriptor
//! @param[out] __tile_total_size Total number of elements in one tile (output)
//! @return Per-dimension tile sizes (unused dimensions are 1)
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::array<__tile_extent_t, _MaxRank> __find_shared_mem_tiling(
  const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor, ::cuda::std::size_t& __tile_total_size)
{
  using ::cuda::std::size_t;
  namespace cudax                     = ::cuda::experimental;
  const auto __current_dev            = cudax::__current_device();
  const size_t __max_shared_mem_bytes = __current_dev.attribute<::cudaDevAttrMaxSharedMemoryPerBlock>();
  const size_t __num_contiguous       = cudax::__num_contiguous_dimensions(__tensor);

  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_sizes{};
  __tile_total_size  = 1;
  size_t __tile_rank = 0;
  for (size_t __r = 0; __r < __num_contiguous; ++__r, ++__tile_rank)
  {
    const auto __tile_size_r = ::cuda::std::min(static_cast<size_t>(__tensor.__extents[__r]), __max_tile_size);
    if (__tile_total_size * __tile_size_r * sizeof(_Tp) > __max_shared_mem_bytes)
    {
      break;
    }
    __tile_sizes[__r] = static_cast<__tile_extent_t>(__tile_size_r);
    __tile_total_size *= __tile_size_r;
  }
  for (size_t __r = __tile_rank; __r < _MaxRank; ++__r)
  {
    __tile_sizes[__r] = 1;
  }
  return __tile_sizes;
}

//! @brief Compute the thread block size for the shared-memory kernel.
//!
//! Balances occupancy by dividing the SM thread budget across as many blocks as the shared memory allows, then caps at
//! the device maximum.
//!
//! @param[in] __tile_total_bytes Shared memory required for one tile in bytes
//! @return Thread block size
[[nodiscard]] _CCCL_HOST_API inline int __find_thread_block_size(::cuda::std::size_t __tile_total_bytes)
{
  using ::cuda::std::size_t;
  const auto __dev                      = ::cuda::experimental::__current_device();
  const size_t __total_sm_threads       = __dev.attribute<::cudaDevAttrMaxThreadsPerMultiProcessor>();
  const size_t __max_thread_block_size  = __dev.attribute<::cudaDevAttrMaxThreadsPerBlock>();
  const size_t __total_shared_mem_bytes = __dev.attribute<::cudaDevAttrMaxSharedMemoryPerBlock>();
  const auto __num_blocks               = ::cuda::ceil_div(__total_shared_mem_bytes, __tile_total_bytes);
  const auto __thread_block_size =
    ::cuda::std::min(::cuda::ceil_div(__total_sm_threads, __num_blocks), __max_thread_block_size);
  return static_cast<int>(__thread_block_size);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__COPY_COPY_SHARED_MEMORY_UTILS_H
