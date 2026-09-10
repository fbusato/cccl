//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a host device function in tile mode

// <cuda/simd>

#include <cuda/simd>

#ifndef __cccl_lib_simd
#  error "__cccl_lib_simd is not defined"
#elif __cccl_lib_simd != 202411L
#  error "__cccl_lib_simd has an incorrect value"
#endif

#ifndef __cccl_lib_simd_complex
#  error "__cccl_lib_simd_complex is not defined"
#elif __cccl_lib_simd_complex != 202502L
#  error "__cccl_lib_simd_complex has an incorrect value"
#endif

#ifndef __cccl_lib_simd_permutations
#  error "__cccl_lib_simd_permutations is not defined"
#elif __cccl_lib_simd_permutations != 202506L
#  error "__cccl_lib_simd_permutations has an incorrect value"
#endif

int main(int, char**)
{
  return 0;
}
