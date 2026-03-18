// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_reduce.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <array>
#include <cstdint>

#include <nvbench_helper.cuh>

#ifndef CCCL_BENCHMARK_HAS_NVCOMP
#  define CCCL_BENCHMARK_HAS_NVCOMP 0
#endif

#if CCCL_BENCHMARK_HAS_NVCOMP
#  include <stdexcept>
#  include <string>

#  include <nvcomp/crc32.h>
#endif

namespace
{
constexpr int crc32_table_size       = 256;
constexpr int crc32_shift_power_size = 32;

struct crc32_spec_t
{
  std::uint32_t poly;
  std::uint32_t init;
  bool ref_in;
  bool ref_out;
  std::uint32_t xorout;
};

constexpr crc32_spec_t crc32_spec{
  0x04C11DB7u, // CRC-32 / PKZIP polynomial
  0xFFFFFFFFu,
  true,
  true,
  0xFFFFFFFFu};

_CCCL_HOST_DEVICE constexpr std::uint32_t byte_swap32(std::uint32_t x)
{
  return ((x & 0x000000FFu) << 24) | ((x & 0x0000FF00u) << 8) | ((x & 0x00FF0000u) >> 8) | ((x & 0xFF000000u) >> 24);
}

_CCCL_HOST_DEVICE constexpr std::uint32_t bit_reverse32(std::uint32_t x)
{
  x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
  x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
  x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
  x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
  return (x << 16) | (x >> 16);
}

_CCCL_HOST_DEVICE constexpr std::uint32_t gf2_poly32_multiply(std::uint32_t x, std::uint32_t y, std::uint32_t mod)
{
  std::uint32_t prod = 0;
  for (int i = 0; i < 32; ++i)
  {
    prod ^= ((y & 1u) != 0u) ? x : 0u;
    x = (x << 1) ^ (((x & 0x80000000u) != 0u) ? mod : 0u);
    y >>= 1;
  }

  return prod;
}

constexpr std::array<std::array<std::uint32_t, crc32_table_size>, 4> make_crc32_tables()
{
  std::array<std::array<std::uint32_t, crc32_table_size>, 4> tables{};

  for (int i = 0; i < crc32_table_size; ++i)
  {
    std::uint32_t msb = static_cast<std::uint32_t>(i) << 24;

    for (int table_idx = 0; table_idx < 4; ++table_idx)
    {
      for (int bit = 0; bit < 8; ++bit)
      {
        msb = (msb << 1) ^ (((msb & 0x80000000u) != 0u) ? crc32_spec.poly : 0u);
      }
      tables[table_idx][i] = msb;
    }
  }

  return tables;
}

constexpr std::array<std::uint32_t, crc32_shift_power_size> make_crc32_shift_powers()
{
  std::array<std::uint32_t, crc32_shift_power_size> powers{};
  powers[0] = crc32_spec.poly;

  for (int i = 1; i < crc32_shift_power_size; ++i)
  {
    powers[i] = gf2_poly32_multiply(powers[i - 1], powers[i - 1], crc32_spec.poly);
  }

  return powers;
}

constexpr auto host_crc32_tables       = make_crc32_tables();
constexpr auto host_crc32_shift_powers = make_crc32_shift_powers();

__device__ __constant__ std::uint32_t crc32_table_0[crc32_table_size];
__device__ __constant__ std::uint32_t crc32_table_1[crc32_table_size];
__device__ __constant__ std::uint32_t crc32_table_2[crc32_table_size];
__device__ __constant__ std::uint32_t crc32_table_3[crc32_table_size];
__device__ __constant__ std::uint32_t crc32_shift_powers[crc32_shift_power_size];

_CCCL_HOST_DEVICE inline std::uint32_t get_crc_table_value(int table_idx, std::uint32_t byte)
{
#ifdef __CUDA_ARCH__
  switch (table_idx)
  {
    case 0:
      return crc32_table_0[byte];
    case 1:
      return crc32_table_1[byte];
    case 2:
      return crc32_table_2[byte];
    default:
      return crc32_table_3[byte];
  }
#else
  return host_crc32_tables[table_idx][byte];
#endif
}

_CCCL_HOST_DEVICE inline std::uint32_t get_crc_shift_power(int bit)
{
#ifdef __CUDA_ARCH__
  return crc32_shift_powers[bit];
#else
  return host_crc32_shift_powers[bit];
#endif
}

void ensure_crc32_constants()
{
  static bool initialized = false;

  if (initialized)
  {
    return;
  }

  NVBENCH_CUDA_CALL_NOEXCEPT(
    cudaMemcpyToSymbol(crc32_table_0, host_crc32_tables[0].data(), sizeof(host_crc32_tables[0])));
  NVBENCH_CUDA_CALL_NOEXCEPT(
    cudaMemcpyToSymbol(crc32_table_1, host_crc32_tables[1].data(), sizeof(host_crc32_tables[1])));
  NVBENCH_CUDA_CALL_NOEXCEPT(
    cudaMemcpyToSymbol(crc32_table_2, host_crc32_tables[2].data(), sizeof(host_crc32_tables[2])));
  NVBENCH_CUDA_CALL_NOEXCEPT(
    cudaMemcpyToSymbol(crc32_table_3, host_crc32_tables[3].data(), sizeof(host_crc32_tables[3])));
  NVBENCH_CUDA_CALL_NOEXCEPT(
    cudaMemcpyToSymbol(crc32_shift_powers, host_crc32_shift_powers.data(), sizeof(host_crc32_shift_powers)));

  initialized = true;
}

_CCCL_HOST_DEVICE std::uint32_t crc32_shift_words(std::uint32_t crc, std::uint32_t suffix_words)
{
  if (suffix_words == 0)
  {
    return crc;
  }

  crc ^= crc32_spec.init;

#pragma unroll
  for (int bit = 0; bit < crc32_shift_power_size; ++bit)
  {
    if ((suffix_words & (1u << bit)) != 0u)
    {
      crc = gf2_poly32_multiply(crc, get_crc_shift_power(bit), crc32_spec.poly);
    }
  }

  return crc;
}

struct crc32_word_transform_op
{
  const std::uint32_t* input;
  std::uint32_t total_words;
  std::uint32_t chunk_words;

  template <typename IndexT>
  _CCCL_HOST_DEVICE std::uint32_t operator()(IndexT index) const
  {
    const auto chunk_idx   = static_cast<std::uint32_t>(index);
    const auto chunk_begin = chunk_idx * chunk_words;
    auto chunk_end         = chunk_begin + chunk_words;

    if (chunk_end > total_words)
    {
      chunk_end = total_words;
    }

    std::uint32_t crc = crc32_spec.init;
    for (auto word_idx = chunk_begin; word_idx < chunk_end; ++word_idx)
    {
      const std::uint32_t raw  = input[word_idx];
      const std::uint32_t word = crc32_spec.ref_in ? bit_reverse32(raw) : byte_swap32(raw);

      crc = crc ^ word;
      crc = get_crc_table_value(3, (crc >> 24) & 0xFF) ^ get_crc_table_value(2, (crc >> 16) & 0xFF)
          ^ get_crc_table_value(1, (crc >> 8) & 0xFF) ^ get_crc_table_value(0, crc & 0xFF);
    }

    const std::uint32_t suffix_words = total_words - chunk_end;
    return crc32_shift_words(crc, suffix_words);
  }
};

__global__ void finalize_crc32_kernel(std::uint32_t* output)
{
  std::uint32_t crc = output[0];

  if (crc32_spec.ref_out)
  {
    crc = bit_reverse32(crc);
  }

  output[0] = crc ^ crc32_spec.xorout;
}

template <typename OffsetT>
void cub_crc32_transform_reduce(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  const auto elements = static_cast<offset_t>(state.get_int64("Elements{io}"));
  const auto chunk_words = static_cast<std::uint32_t>(state.get_int64("ChunkWords"));

  ensure_crc32_constants();

  thrust::device_vector<std::uint32_t> input = generate(elements);
  thrust::device_vector<std::uint32_t> output(1);

  auto* d_input  = thrust::raw_pointer_cast(input.data());
  auto* d_output = thrust::raw_pointer_cast(output.data());

  crc32_word_transform_op transform_op{d_input, static_cast<std::uint32_t>(elements), chunk_words};

  const auto chunks =
    static_cast<offset_t>((static_cast<std::uint32_t>(elements) + chunk_words - 1u) / chunk_words);
  auto indices = thrust::counting_iterator<offset_t>(0);

  state.add_element_count(elements);
  state.add_global_memory_reads<std::uint32_t>(elements, "Size");
  state.add_global_memory_writes<std::uint32_t>(1);

  std::size_t temp_storage_bytes = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cub::DeviceReduce::TransformReduce(
    nullptr, temp_storage_bytes, indices, d_output, chunks, ::cuda::std::bit_xor<>{}, transform_op, std::uint32_t{0}));

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  auto* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    NVBENCH_CUDA_CALL_NOEXCEPT(cub::DeviceReduce::TransformReduce(
      d_temp_storage,
      temp_storage_bytes,
      indices,
      d_output,
      chunks,
      ::cuda::std::bit_xor<>{},
      transform_op,
      std::uint32_t{0},
      launch.get_stream()));

    //   finalize_crc32_kernel<<<1, 1, 0, launch.get_stream()>>>(d_output);
    NVBENCH_CUDA_CALL_NOEXCEPT(cudaPeekAtLastError());
  });
}

#if CCCL_BENCHMARK_HAS_NVCOMP
nvcompCRC32Spec_t make_nvcomp_crc32_spec()
{
  nvcompCRC32Spec_t spec{};
  spec.poly    = crc32_spec.poly;
  spec.init    = crc32_spec.init;
  spec.ref_in  = crc32_spec.ref_in;
  spec.ref_out = crc32_spec.ref_out;
  spec.xorout  = crc32_spec.xorout;
  return spec;
}

void throw_if_nvcomp_error(nvcompStatus_t status, const char* operation)
{
  if (status != nvcompSuccess)
  {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

template <typename OffsetT>
void nvcomp_crc32(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  const auto elements    = static_cast<offset_t>(state.get_int64("Elements{io}"));
  const auto total_bytes = static_cast<std::size_t>(elements) * sizeof(std::uint32_t);
  auto bench_stream      = state.get_cuda_stream().get_stream();

  thrust::device_vector<std::uint32_t> input = generate(elements);
  thrust::device_vector<std::uint32_t> output(1);
  thrust::device_vector<const void*> input_ptrs(1);
  thrust::device_vector<std::size_t> input_sizes(1, total_bytes);

  auto* d_output     = thrust::raw_pointer_cast(output.data());
  auto* d_input_ptrs = thrust::raw_pointer_cast(input_ptrs.data());
  auto* d_input_size = thrust::raw_pointer_cast(input_sizes.data());

  const void* d_input = thrust::raw_pointer_cast(input.data());
  NVBENCH_CUDA_CALL_NOEXCEPT(
    cudaMemcpyAsync(d_input_ptrs, &d_input, sizeof(d_input), cudaMemcpyHostToDevice, bench_stream));
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaStreamSynchronize(bench_stream));

  nvcompCRC32KernelConf_t kernel_conf{};
  throw_if_nvcomp_error(nvcompBatchedCRC32GetHeuristicConf(nullptr, 1, &kernel_conf, total_bytes, bench_stream),
                        "nvcompBatchedCRC32GetHeuristicConf");

  nvcompBatchedCRC32Opts_t opts{};
  opts.spec        = make_nvcomp_crc32_spec();
  opts.kernel_conf = kernel_conf;

  state.add_element_count(elements);
  state.add_global_memory_reads<std::uint32_t>(elements, "Size");
  state.add_global_memory_writes<std::uint32_t>(1);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    throw_if_nvcomp_error(
      nvcompBatchedCRC32Async(
        d_input_ptrs, d_input_size, 1, d_output, opts, nvcompCRC32OnlySegment, nullptr, launch.get_stream()),
      "nvcompBatchedCRC32Async");
  });
}
#endif
} // namespace

NVBENCH_BENCH_TYPES(cub_crc32_transform_reduce, NVBENCH_TYPE_AXES(offset_types))
  .set_name("cub")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(28, 28, 4))
  .add_int64_axis("ChunkWords", {4, 8, 16, 32, 64, 128});

#if CCCL_BENCHMARK_HAS_NVCOMP
NVBENCH_BENCH_TYPES(nvcomp_crc32, NVBENCH_TYPE_AXES(offset_types))
  .set_name("nvcomp")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(28, 28, 4));
#endif
