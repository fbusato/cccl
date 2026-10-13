// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_sort.cuh>

#include <nvbench_helper.cuh>

#include "../segmented_row_lengths.cuh"

template <class T, typename OffsetT>
void row_lengths(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const thrust::device_vector<OffsetT> offsets =
    load_segment_offsets<OffsetT>(state.get_string("Matrix"), state.get_string("Layout"));
  const auto segments = offsets.size() - 1;
  const auto elements = static_cast<std::size_t>(offsets.back());

  thrust::device_vector<T> buffer_1 = generate(elements);
  thrust::device_vector<T> buffer_2(elements, thrust::no_init);

  T* d_buffer_1 = thrust::raw_pointer_cast(buffer_1.data());
  T* d_buffer_2 = thrust::raw_pointer_cast(buffer_2.data());

  const OffsetT* d_begin_offsets = thrust::raw_pointer_cast(offsets.data());
  const OffsetT* d_end_offsets   = d_begin_offsets + 1;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_reads<OffsetT>(segments + 1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               auto env = cub_bench_env(alloc, launch);
               _CCCL_TRY_RUNTIME_API(
                 cub::DeviceSegmentedSort::SortKeys,
                 "SortKeys failed",
                 d_buffer_1,
                 d_buffer_2,
                 static_cast<cuda::std::int64_t>(elements),
                 static_cast<cuda::std::int64_t>(segments),
                 d_begin_offsets,
                 d_end_offsets,
                 env);
             });
}

NVBENCH_BENCH_TYPES(row_lengths, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, double>, nvbench::type_list<int32_t>))
  .set_name("row_lengths")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_string_axis("Matrix", row_length_matrices())
  .add_string_axis("Layout", {"original", "shuffled"});
