// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_segmented_reduce.cuh>

#include <nvbench_helper.cuh>

#include "../segmented_row_lengths.cuh"

template <class T, typename OffsetT>
void row_lengths(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const thrust::device_vector<OffsetT> offsets =
    load_segment_offsets<OffsetT>(state.get_string("Matrix"), state.get_string("Layout"));
  const auto segments = offsets.size() - 1;
  const auto elements = static_cast<std::size_t>(offsets.back());

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(segments, thrust::no_init);

  const T* d_in                  = thrust::raw_pointer_cast(in.data());
  T* d_out                       = thrust::raw_pointer_cast(out.data());
  const OffsetT* d_begin_offsets = thrust::raw_pointer_cast(offsets.data());
  const OffsetT* d_end_offsets   = d_begin_offsets + 1;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(segments);
  state.add_global_memory_reads<OffsetT>(segments + 1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceSegmentedReduce::Sum,
      "Segmented sum failed",
      d_in,
      d_out,
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
