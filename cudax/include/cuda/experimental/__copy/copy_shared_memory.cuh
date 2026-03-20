//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_COPY_SHARED_MEMORY_H
#define _CUDAX__COPY_COPY_SHARED_MEMORY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/launch>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/stable_sort.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/array>

#include <cuda/experimental/__copy/tensor_copy_utils.cuh>
#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy_bytes/abs_integer.cuh>
#include <cuda/experimental/__copy_bytes/types.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Shared-memory tiled transpose kernel for arbitrary-rank tensors.
//!
//! Each block processes one tile. Threads cooperatively iterate over tile elements with a stride
//! loop. Full (interior) tiles use a two-phase shared-memory transpose: load source data into
//! shared memory using source-coalesced ordering, then store from shared memory to destination
//! using destination-natural ordering. Partial (boundary) tiles copy elements directly without
//! shared memory.
//!
//! @param[in]  __config                Kernel launch configuration
//! @param[in]  __src_ptr               Pointer to source data
//! @param[out] __dst_ptr               Pointer to destination data
//! @param[in]  __grid_iter             Coordinate iterator for grid tile decomposition
//! @param[in]  __grid_tile_src_strides Per-dimension source strides scaled by tile sizes
//! @param[in]  __grid_tile_dst_strides Per-dimension destination strides scaled by tile sizes
//! @param[in]  __src_perm_iter         Coordinate iterator for src-permuted tile decomposition
//! @param[in]  __src_perm_src_strides  Src-permuted source strides for loading
//! @param[in]  __src_perm_smem_strides Src-permuted shared memory strides for loading
//! @param[in]  __tile_iter             Coordinate iterator for dst-natural tile decomposition
//! @param[in]  __dst_strides           Per-dimension destination strides for storing
//! @param[in]  __tile_total_size       Total number of elements in one tile
//! @param[in]  __tile_sizes            Per-dimension tile extents
//! @param[in]  __extents               Per-dimension tensor extents (for partial-tile bounds)
//! @param[in]  __src_strides           Per-dimension source strides (for partial-tile access)
template <typename _Config,
          ::cuda::std::size_t _MaxRank,
          typename _Tp,
          typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut>
__global__ void __copy_shared_mem_kernel(
  _CCCL_GRID_CONSTANT const _Config __config,
  _CCCL_GRID_CONSTANT const _Tp* const _CCCL_RESTRICT __src_ptr,
  _CCCL_GRID_CONSTANT _Tp* const _CCCL_RESTRICT __dst_ptr,
  _CCCL_GRID_CONSTANT const __tensor_coord_iterator<_ExtentT, _MaxRank> __grid_iter,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTIn, _MaxRank> __grid_tile_src_strides,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTOut, _MaxRank> __grid_tile_dst_strides,
  _CCCL_GRID_CONSTANT const __tensor_coord_iterator<unsigned, _MaxRank> __src_perm_iter,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTIn, _MaxRank> __src_perm_src_strides,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<int, _MaxRank> __src_perm_smem_strides,
  _CCCL_GRID_CONSTANT const __tensor_coord_iterator<unsigned, _MaxRank> __tile_iter,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTOut, _MaxRank> __dst_strides,
  _CCCL_GRID_CONSTANT const int __tile_total_size,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<unsigned, _MaxRank> __tile_sizes,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<::cuda::std::make_unsigned_t<_ExtentT>, _MaxRank> __extents,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTIn, _MaxRank> __src_strides)
{
  extern __shared__ char __smem_bytes[];
  auto* __smem = reinterpret_cast<_Tp*>(__smem_bytes);

  const auto __tid          = ::cuda::gpu_thread.rank_as<int>(::cuda::block, __config);
  const auto __block_stride = ::cuda::gpu_thread.count_as<int>(::cuda::block, __config);
  const auto __grid_idx     = ::cuda::block.index_as<_ExtentT>(::cuda::grid).x;

  // Grid tile decomposition: map linearized block index to src/dst base offsets
  const auto __grid_coords = __grid_iter(static_cast<_ExtentT>(__grid_idx));
  _StrideTIn __src_base    = 0;
  _StrideTOut __dst_base   = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __k = 0; __k < int{_MaxRank}; ++__k)
  {
    __src_base += static_cast<_StrideTIn>(__grid_coords[__k]) * __grid_tile_src_strides[__k];
    __dst_base += static_cast<_StrideTOut>(__grid_coords[__k]) * __grid_tile_dst_strides[__k];
  }

  // Partial tile detection
  using _UEp          = ::cuda::std::make_unsigned_t<_ExtentT>;
  bool __is_full_tile = true;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __k = 0; __k < int{_MaxRank}; ++__k)
  {
    const auto __block_start = static_cast<_UEp>(__grid_coords[__k]) * __tile_sizes[__k];
    if (__block_start + __tile_sizes[__k] > __extents[__k])
    {
      __is_full_tile = false;
    }
  }

  if (__is_full_tile)
  {
    // === Full-tile shared-memory transpose ===
    // Phase 1: load source -> shared memory (src-coalesced ordering)
    const __partial_tensor __src_pt{
      __src_ptr + __src_base, __src_perm_src_strides, ::cuda::std::default_accessor<const _Tp>{}};
    const __partial_tensor __smem_pt{__smem, __src_perm_smem_strides, ::cuda::std::default_accessor<_Tp>{}};

    for (int __i = __tid; __i < __tile_total_size; __i += __block_stride)
    {
      const auto __coords = __src_perm_iter(static_cast<unsigned>(__i));
      __smem_pt(__coords) = __src_pt(__coords);
    }

    __syncthreads();

    // Phase 2: store shared memory -> destination (dst-natural ordering)
    const __partial_tensor __dst_pt{__dst_ptr + __dst_base, __dst_strides, ::cuda::std::default_accessor<_Tp>{}};

    for (int __i = __tid; __i < __tile_total_size; __i += __block_stride)
    {
      const auto __coords = __tile_iter(static_cast<unsigned>(__i));
      __dst_pt(__coords)  = __smem[__i];
    }
  }
#if 1
  else
  {
    // === Boundary direct-copy (no shared memory) ===
    const __partial_tensor __src_pt{__src_ptr + __src_base, __src_strides, ::cuda::std::default_accessor<const _Tp>{}};
    const __partial_tensor __dst_pt{__dst_ptr + __dst_base, __dst_strides, ::cuda::std::default_accessor<_Tp>{}};

    ::cuda::std::array<unsigned, _MaxRank> __actual_sizes{};
    int __actual_total = 1;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __k = 0; __k < int{_MaxRank}; ++__k)
    {
      const auto __block_start = static_cast<_UEp>(__grid_coords[__k]) * __tile_sizes[__k];
      __actual_sizes[__k] =
        static_cast<unsigned>(::cuda::std::min(static_cast<_UEp>(__tile_sizes[__k]), __extents[__k] - __block_start));
      __actual_total *= __actual_sizes[__k];
    }

    for (int __i = __tid; __i < __actual_total; __i += __block_stride)
    {
      ::cuda::std::array<unsigned, _MaxRank> __coords;
      int __linear = __i;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __k = 0; __k < int{_MaxRank}; ++__k)
      {
        __coords[__k] = __linear % __actual_sizes[__k];
        __linear /= __actual_sizes[__k];
      }
      __dst_pt(__coords) = __src_pt(__coords);
    }
  }
#endif
}

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/devices>

//! @brief Return a device_ref for the current CUDA device.
//!
//! @return Device reference for the active CUDA context's device
_CCCL_HOST_API inline device_ref __current_device()
{
  const auto __dev_id = ::cuda::__driver::__cudevice_to_ordinal(::cuda::__driver::__ctxGetDevice());
  return ::cuda::devices[__dev_id];
}

//! @brief Decide whether the shared-memory tiled transpose kernel is profitable.
//!
//! Returns true when the destination has stride-1 in mode 0, the source does not, there are at
//! least two contiguous destination dimensions, and the resulting tile is large enough to amortize
//! the shared-memory overhead.
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
  const size_t __num_contiguous_dst = ::cuda::experimental::__num_contiguous_dimensions(__dst);
  if (__src.__strides[0] == 1 || __dst.__strides[0] != 1 || __num_contiguous_dst < 2)
  {
    return false;
  }
  constexpr int __warp_size        = 32;
  constexpr size_t __max_tile_size = __warp_size;
  const size_t __max_shared_mem_bytes =
    ::cuda::experimental::__current_device().attribute<::cudaDevAttrMaxSharedMemoryPerBlockOptin>();
  size_t __size_product = 1;
  int __tile_rank       = 0;
  for (size_t __r = 0; __r < __num_contiguous_dst; ++__r, ++__tile_rank)
  {
    const auto __tile_size_r = ::cuda::std::min(static_cast<size_t>(__dst.__extents[__r]), __max_tile_size);
    if (__size_product * __tile_size_r * sizeof(_TpOut) > __max_shared_mem_bytes)
    {
      break;
    }
    __size_product *= __tile_size_r;
  }
  return (__tile_rank >= 2 && __size_product >= static_cast<size_t>(__warp_size) * 8);
}

//! @brief Compute the shared-memory tile geometry for a destination tensor.
//!
//! Greedily expands the tile across contiguous dimensions up to the warp-size cap per dimension
//! and the device shared-memory limit. Dimensions beyond the tile rank are set to extent 1.
//!
//! @param[in]  __tensor         Destination raw tensor descriptor
//! @param[out] __tile_total_size Total number of elements in one tile (output)
//! @return Raw tensor describing the tile extents and contiguous strides
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API __raw_tensor<int, int, _Tp, _MaxRank> __find_shared_mem_tiling(
  const __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor, ::cuda::std::size_t& __tile_total_size)
{
  using ::cuda::std::size_t;
  constexpr int __warp_size        = 32;
  constexpr size_t __max_tile_size = __warp_size;
  const size_t __max_shared_mem_bytes =
    ::cuda::experimental::__current_device().attribute<::cudaDevAttrMaxSharedMemoryPerBlockOptin>();
  __raw_tensor<int, int, _Tp, _MaxRank> __tiling{};
  auto& __tile_rank    = __tiling.__rank;
  auto& __tile_extents = __tiling.__extents;
  auto& __tile_strides = __tiling.__strides;

  const size_t __num_contiguous_dst = ::cuda::experimental::__num_contiguous_dimensions(__tensor);
  int __size_product                = 1;
  __tile_rank                       = 0;
  for (size_t __r = 0; __r < __num_contiguous_dst; ++__r, ++__tile_rank)
  {
    const auto __tile_size_r = ::cuda::std::min(static_cast<size_t>(__tensor.__extents[__r]), __max_tile_size);
    if (__size_product * __tile_size_r * sizeof(_Tp) > __max_shared_mem_bytes)
    {
      break;
    }
    __tile_extents[__r] = __tile_size_r;
    __tile_strides[__r] = __size_product;
    __size_product *= __tile_size_r;
  }
  for (size_t __r = __tile_rank; __r < _MaxRank; ++__r)
  {
    __tile_extents[__r] = 1;
    __tile_strides[__r] = __size_product;
  }
  __tile_total_size = __size_product;
  return __tiling;
}

//! @brief Compute the thread block size for the shared-memory kernel.
//!
//! Balances occupancy by dividing the SM thread budget across as many blocks as the shared memory
//! allows, then caps at the device maximum.
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

//! @brief Launch the shared-memory tiled transpose kernel.
//!
//! Precomputes the src-coalesced permutation and tile geometry, constructs coordinate iterators,
//! then launches one block per tile. Used for the "transpose case" where the destination has
//! stride-1 in mode 0 and the source has stride-1 elsewhere.
//!
//! @pre `__src.__rank >= 2`
//! @pre `__dst.__strides[0] == 1`
//! @pre `__src.__strides[0] != 1`
//!
//! @param[in]  __src    Source raw tensor descriptor
//! @param[out] __dst    Destination raw tensor descriptor
//! @param[in]  __stream CUDA stream for asynchronous execution
template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __launch_copy_shared_mem_kernel(
  const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _MaxRank>& __src,
  const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _MaxRank>& __dst,
  ::cuda::stream_ref __stream)
{
  namespace cudax = ::cuda::experimental;
  using ::cuda::std::size_t;
  using _UEp         = ::cuda::std::make_unsigned_t<_ExtentT>;
  using __value_type = ::cuda::std::remove_cv_t<_TpIn>;
  _CCCL_ASSERT(__src.__rank >= 2, "Rank must be at least 2 for shared memory transpose");
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  _CCCL_ASSERT(__src.__extents == __dst.__extents, "Source and destination extents must be identical");
  const size_t __rank = __src.__rank;

  size_t __tile_total_size      = 0;
  const auto __tiling           = cudax::__find_shared_mem_tiling(__dst, __tile_total_size);
  const int __thread_block_size = cudax::__find_thread_block_size(__tile_total_size * sizeof(__value_type));

  // Tile sizes (unused dimensions set to 1 for correct iterator behavior)
  ::cuda::std::array<unsigned, _MaxRank> __tile_sizes{};
  for (size_t __i = 0; __i < _MaxRank; ++__i)
  {
    __tile_sizes[__i] = (__i < __rank) ? static_cast<unsigned>(__tiling.__extents[__i]) : 1u;
  }

  // Tensor extents (unused dimensions set to 1 for correct partial-tile logic)
  ::cuda::std::array<_UEp, _MaxRank> __extents{};
  for (size_t __i = 0; __i < _MaxRank; ++__i)
  {
    __extents[__i] = (__i < __rank) ? static_cast<_UEp>(__dst.__extents[__i]) : _UEp{1};
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Grid sizes and strides for block index decomposition
  ::cuda::std::array<_ExtentT, _MaxRank> __grid_tile_sizes{};
  ::cuda::std::array<_StrideTIn, _MaxRank> __grid_tile_src_strides{};
  ::cuda::std::array<_StrideTOut, _MaxRank> __grid_tile_dst_strides{};
  size_t __grid_size = 1;
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    __grid_tile_sizes[__i]       = ::cuda::ceil_div(__src.__extents[__i], static_cast<_ExtentT>(__tile_sizes[__i]));
    __grid_tile_src_strides[__i] = static_cast<_StrideTIn>(__tile_sizes[__i]) * __src.__strides[__i];
    __grid_tile_dst_strides[__i] = static_cast<_StrideTOut>(__tile_sizes[__i]) * __dst.__strides[__i];
    __grid_size *= __grid_tile_sizes[__i];
  }
  for (size_t __i = __rank; __i < _MaxRank; ++__i)
  {
    __grid_tile_sizes[__i] = 1;
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Src-coalesced permutation: sort modes by ascending |src_stride|
  ::cuda::std::array<int, _MaxRank> __src_perm{};
  for (size_t __i = 0; __i < _MaxRank; ++__i)
  {
    __src_perm[__i] = __i;
  }
  ::cuda::std::stable_sort(__src_perm.begin(), __src_perm.begin() + __rank, [&](auto __a, auto __b) {
    return cudax::__abs_integer(__src.__strides[__a]) < cudax::__abs_integer(__src.__strides[__b]);
  });

  //--------------------------------------------------------------------------------------------------------------------
  // Reordered arrays for loading src to shared memory (src-coalesced thread decomposition)
  ::cuda::std::array<unsigned, _MaxRank> __src_perm_sizes{};
  ::cuda::std::array<_StrideTIn, _MaxRank> __src_perm_src_strides{};
  ::cuda::std::array<int, _MaxRank> __src_perm_smem_strides{};
  ::cuda::std::array<int, _MaxRank> __canonical_strides{};
  __canonical_strides[0] = 1;
  for (size_t __i = 1; __i < __rank; ++__i)
  {
    __canonical_strides[__i] = __canonical_strides[__i - 1] * __tile_sizes[__i - 1];
  }
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    const auto __p               = __src_perm[__i];
    __src_perm_sizes[__i]        = __tile_sizes[__p];
    __src_perm_src_strides[__i]  = __src.__strides[__p];
    __src_perm_smem_strides[__i] = __canonical_strides[__p];
  }
  for (size_t __i = __rank; __i < _MaxRank; ++__i)
  {
    __src_perm_sizes[__i] = 1;
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Construct coordinate iterators on the host (precomputed fast modulo/division)
  const __tensor_coord_iterator<_ExtentT, _MaxRank> __grid_iter(__grid_tile_sizes);
  const __tensor_coord_iterator<unsigned, _MaxRank> __src_perm_iter(__src_perm_sizes);
  const __tensor_coord_iterator<unsigned, _MaxRank> __tile_iter(__tile_sizes);

  //--------------------------------------------------------------------------------------------------------------------
  // Launch the kernel
  const auto __config = ::cuda::make_config(
    ::cuda::block_dims(static_cast<unsigned>(__thread_block_size)),
    ::cuda::grid_dims(__grid_size),
    ::cuda::dynamic_shared_memory<__value_type[]>(__tile_total_size, ::cuda::non_portable));
  const auto __kernel =
    cudax::__copy_shared_mem_kernel<decltype(__config), _MaxRank, __value_type, _ExtentT, _StrideTIn, _StrideTOut>;

  ::cuda::launch(
    __stream,
    __config,
    __kernel,
    static_cast<const __value_type*>(__src.__data),
    static_cast<__value_type*>(__dst.__data),
    __grid_iter,
    __grid_tile_src_strides,
    __grid_tile_dst_strides,
    __src_perm_iter,
    __src_perm_src_strides,
    __src_perm_smem_strides,
    __tile_iter,
    __dst.__strides,
    static_cast<int>(__tile_total_size),
    __tile_sizes,
    __extents,
    __src.__strides);
}

#endif // !_CCCL_COMPILER(NVRTC)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_COPY_SHARED_MEMORY_H
