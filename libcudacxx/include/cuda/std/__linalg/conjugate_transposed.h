// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___LINALG_CONJUGATE_TRANSPOSED_HPP
#define _LIBCUDACXX___LINALG_CONJUGATE_TRANSPOSED_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/version>

#if defined(__cccl_lib_mdspan) && _CCCL_STD_VER >= 2017

#  include <cuda/std/__linalg/conjugated.h>
#  include <cuda/std/__linalg/transposed.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace linalg
{

template <class _ElementType, class _Extents, class _Layout, class _Accessor>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI auto
conjugate_transposed(mdspan<_ElementType, _Extents, _Layout, _Accessor> __a)
{
  return conjugated(transposed(__a));
}

} // end namespace linalg

_LIBCUDACXX_END_NAMESPACE_STD

#endif // defined(__cccl_lib_mdspan) && _CCCL_STD_VER >= 2017
#endif // _LIBCUDACXX___LINALG_CONJUGATE_TRANSPOSED_HPP