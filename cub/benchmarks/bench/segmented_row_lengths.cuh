// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <cuda/iterator>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Segment layouts read from headerless little-endian uint32 row-length files located under
// $CCCL_BENCH_ROW_LENGTHS_DIR/<group>/<matrix>.row_lengths.u32le

inline constexpr const char* row_lengths_suffix = ".row_lengths.u32le";

inline std::filesystem::path row_lengths_dir()
{
  const char* dir = std::getenv("CCCL_BENCH_ROW_LENGTHS_DIR");
  return dir ? std::filesystem::path{dir} : std::filesystem::path{};
}

inline std::vector<std::string> row_length_matrices()
{
  std::vector<std::string> ids;
  const auto root = row_lengths_dir();
  if (root.empty())
  {
    std::fprintf(stderr, "warning: CCCL_BENCH_ROW_LENGTHS_DIR is not set, no row-length workloads registered\n");
    return ids;
  }
  const std::string suffix{row_lengths_suffix};
  for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
  {
    const auto rel = std::filesystem::relative(entry.path(), root).generic_string();
    if (entry.is_regular_file() && rel.size() > suffix.size()
        && rel.compare(rel.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
      ids.push_back(rel.substr(0, rel.size() - suffix.size()));
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

// Fitted distributions read from $CCCL_BENCH_DISTRIBUTIONS, one line per matrix and layout:
//   <matrix> <layout> <dist> <num_segments> <params...>
//   binomial t p | negative_binomial k p | geometric p | uniform_int a b | poisson mean | discrete K v1..vK w1..wK
struct distribution_spec
{
  std::string name;
  std::size_t num_segments;
  std::vector<double> params;
};

inline const std::map<std::string, distribution_spec>& distribution_specs()
{
  static const auto specs = [] {
    std::map<std::string, distribution_spec> result;
    const char* path = std::getenv("CCCL_BENCH_DISTRIBUTIONS");
    if (!path)
    {
      return result;
    }
    std::ifstream file{path};
    if (!file)
    {
      throw std::runtime_error(std::string{"Unable to open "} + path);
    }
    std::string line;
    while (std::getline(file, line))
    {
      std::istringstream fields{line};
      std::string matrix, layout;
      distribution_spec spec;
      fields >> matrix >> layout >> spec.name >> spec.num_segments;
      for (double value; fields >> value;)
      {
        spec.params.push_back(value);
      }
      result.emplace(matrix + ' ' + layout, std::move(spec));
    }
    return result;
  }();
  return specs;
}

inline std::vector<std::uint32_t>
generate_lengths(const std::string& matrix, const std::string& layout, std::size_t num_segments)
{
  const auto& specs = distribution_specs();
  const auto it     = specs.find(matrix + ' ' + layout);
  if (it == specs.end())
  {
    throw std::runtime_error("No " + layout + " distribution for " + matrix + " (check CCCL_BENCH_DISTRIBUTIONS)");
  }
  const auto& [name, spec_segments, p] = it->second;
  if (spec_segments != num_segments)
  {
    throw std::runtime_error("Segment count mismatch between distribution and row-length file: " + matrix);
  }

  std::vector<std::uint32_t> lengths(num_segments);
  std::mt19937_64 rng{42};
  const auto fill = [&](auto dist, auto to_length) {
    for (auto& length : lengths)
    {
      const auto value = static_cast<std::int64_t>(to_length(dist(rng)));
      if (value < 0 || value > std::numeric_limits<std::uint32_t>::max())
      {
        throw std::runtime_error("Generated segment length out of range: " + matrix);
      }
      length = static_cast<std::uint32_t>(value);
    }
  };
  const auto identity = [](auto value) {
    return value;
  };

  if (name == "binomial")
  {
    fill(std::binomial_distribution<std::int64_t>{static_cast<std::int64_t>(p.at(0)), p.at(1)}, identity);
  }
  else if (name == "negative_binomial")
  {
    fill(std::negative_binomial_distribution<std::int64_t>{static_cast<std::int64_t>(p.at(0)), p.at(1)}, identity);
  }
  else if (name == "geometric")
  {
    fill(std::geometric_distribution<std::int64_t>{p.at(0)}, identity);
  }
  else if (name == "uniform_int")
  {
    fill(std::uniform_int_distribution<std::int64_t>{static_cast<std::int64_t>(p.at(0)),
                                                     static_cast<std::int64_t>(p.at(1))},
         identity);
  }
  else if (name == "poisson")
  {
    fill(std::poisson_distribution<std::int64_t>{p.at(0)}, identity);
  }
  else if (name == "discrete")
  {
    const auto k = static_cast<std::size_t>(p.at(0));
    if (p.size() != 1 + 2 * k)
    {
      throw std::runtime_error("Malformed discrete distribution: " + matrix);
    }
    const auto values = p.begin() + 1;
    fill(std::discrete_distribution<std::size_t>{values + k, values + 2 * k}, [&](std::size_t i) {
      return values[i];
    });
  }
  else
  {
    throw std::runtime_error("Unknown distribution " + name + ": " + matrix);
  }
  return lengths;
}

// layout == "original": row lengths in file order
// layout == "shuffled": exact histogram of the row lengths, expanded back to one length per segment, then shuffled
// layout == "discrete" / "named": lengths sampled from the fitted distribution (same number of segments)
template <typename OffsetT>
thrust::device_vector<OffsetT> load_segment_offsets(const std::string& matrix, const std::string& layout)
{
  const auto path = row_lengths_dir() / (matrix + row_lengths_suffix);
  std::ifstream file{path, std::ios::binary | std::ios::ate};
  if (!file)
  {
    throw std::runtime_error("Unable to open " + path.string());
  }
  const auto bytes = static_cast<std::size_t>(file.tellg());
  if (bytes == 0 || bytes % sizeof(std::uint32_t) != 0)
  {
    throw std::runtime_error("Invalid row-length file size: " + path.string());
  }
  std::vector<std::uint32_t> h_lengths(bytes / sizeof(std::uint32_t));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(h_lengths.data()), static_cast<std::streamsize>(bytes));

  constexpr auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<OffsetT>::max());
  const auto num_segments   = h_lengths.size();
  if (num_segments > max_offset)
  {
    throw std::runtime_error("Too many segments for OffsetT: " + matrix);
  }

  const bool fitted = layout == "discrete" || layout == "named";
  if (fitted)
  {
    h_lengths = generate_lengths(matrix, layout, num_segments);
  }

  thrust::device_vector<std::uint32_t> lengths = h_lengths;
  const auto total = thrust::reduce(thrust::device, lengths.begin(), lengths.end(), std::uint64_t{0});
  if (total > max_offset)
  {
    throw std::runtime_error("Too many elements for OffsetT: " + matrix);
  }

  if (layout == "shuffled")
  {
    // histogram: distinct lengths and their counts
    thrust::device_vector<std::uint32_t> sorted = lengths;
    thrust::sort(thrust::device, sorted.begin(), sorted.end());
    thrust::device_vector<std::uint32_t> values(num_segments);
    thrust::device_vector<std::int64_t> ends(num_segments);
    const auto num_bins = static_cast<std::size_t>(
      thrust::reduce_by_key(
        thrust::device,
        sorted.begin(),
        sorted.end(),
        cuda::constant_iterator<std::int64_t>(1),
        values.begin(),
        ends.begin())
        .first
      - values.begin());

    // expansion: segment i takes the value of the first bin whose cumulative count exceeds i
    thrust::inclusive_scan(thrust::device, ends.begin(), ends.begin() + num_bins, ends.begin());
    thrust::device_vector<std::int64_t> bin_idx(num_segments);
    thrust::upper_bound(
      thrust::device,
      ends.begin(),
      ends.begin() + num_bins,
      cuda::counting_iterator<std::int64_t>(0),
      cuda::counting_iterator<std::int64_t>(static_cast<std::int64_t>(num_segments)),
      bin_idx.begin());
    thrust::gather(thrust::device, bin_idx.begin(), bin_idx.end(), values.begin(), lengths.begin());

    if (static_cast<std::size_t>(ends[num_bins - 1]) != num_segments
        || thrust::reduce(thrust::device, lengths.begin(), lengths.end(), std::uint64_t{0}) != total)
    {
      throw std::runtime_error("Histogram expansion mismatch: " + matrix);
    }

    thrust::shuffle(thrust::device, lengths.begin(), lengths.end(), thrust::default_random_engine{42});
  }
  else if (layout != "original" && !fitted)
  {
    throw std::runtime_error("Unknown Layout: " + layout);
  }

  thrust::device_vector<OffsetT> offsets(num_segments + 1, OffsetT{0});
  thrust::inclusive_scan(thrust::device, lengths.begin(), lengths.end(), offsets.begin() + 1);
  return offsets;
}
