/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <limits>

#include <nvbench_helper.cuh>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

using types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t,
#if NVBENCH_HELPER_HAS_I128
                     int128_t,
#endif
                     float,
                     double>;

TEMPLATE_LIST_TEST_CASE("Generators produce data within specified range", "[gen]", types)
{
  const auto min = static_cast<TestType>(GENERATE_COPY(take(3, random(-124, 0))));
  const auto max = static_cast<TestType>(GENERATE_COPY(take(3, random(0, 124))));

  const thrust::device_vector<TestType> data = generate(1 << 16, bit_entropy::_1_000, min, max);

  const TestType min_element = *thrust::min_element(data.begin(), data.end());
  const TestType max_element = *thrust::max_element(data.begin(), data.end());

  REQUIRE(min_element >= min);
  REQUIRE(max_element <= max);
}
