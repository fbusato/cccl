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
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/system/detail/generic/gather.h>
#include <thrust/transform.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator gather(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator map_first,
  InputIterator map_last,
  RandomAccessIterator input_first,
  OutputIterator result)
{
  return thrust::transform(
    exec,
    thrust::make_permutation_iterator(input_first, map_first),
    thrust::make_permutation_iterator(input_first, map_last),
    result,
    ::cuda::std::identity{});
} // end gather()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename RandomAccessIterator,
          typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator gather_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 map_first,
  InputIterator1 map_last,
  InputIterator2 stencil,
  RandomAccessIterator input_first,
  OutputIterator result)
{
  return thrust::gather_if(exec, map_first, map_last, stencil, input_first, result, ::cuda::std::identity{});
} // end gather_if()

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename RandomAccessIterator,
          typename OutputIterator,
          typename Predicate>
_CCCL_HOST_DEVICE OutputIterator gather_if(
  thrust::execution_policy<DerivedPolicy>& exec,
  InputIterator1 map_first,
  InputIterator1 map_last,
  InputIterator2 stencil,
  RandomAccessIterator input_first,
  OutputIterator result,
  Predicate pred)
{
  return thrust::transform_if(
    exec,
    thrust::make_permutation_iterator(input_first, map_first),
    thrust::make_permutation_iterator(input_first, map_last),
    stencil,
    result,
    ::cuda::std::identity{},
    pred);
} // end gather_if()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
