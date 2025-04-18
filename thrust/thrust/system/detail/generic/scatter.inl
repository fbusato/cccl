/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/system/detail/generic/scatter.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename RandomAccessIterator>
_CCCL_HOST_DEVICE void
scatter(thrust::execution_policy<DerivedPolicy>& exec,
        InputIterator1 first,
        InputIterator1 last,
        InputIterator2 map,
        RandomAccessIterator output)
{
  thrust::copy(exec, first, last, thrust::make_permutation_iterator(output, map));
} // end scatter()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator>
_CCCL_HOST_DEVICE void scatter_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 map,
  InputIterator3 stencil,
  RandomAccessIterator output)
{
  thrust::scatter_if(exec, first, last, map, stencil, output, ::cuda::std::identity{});
} // end scatter_if()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename RandomAccessIterator,
          typename Predicate>
_CCCL_HOST_DEVICE void scatter_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 map,
  InputIterator3 stencil,
  RandomAccessIterator output,
  Predicate pred)
{
  thrust::transform_if(
    exec, first, last, stencil, thrust::make_permutation_iterator(output, map), ::cuda::std::identity{}, pred);
} // end scatter_if()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
