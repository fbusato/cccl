# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from numba import cuda, types

NUMBA_TYPES_TO_NP = {
    types.int8: np.int8,
    types.int16: np.int16,
    types.int32: np.int32,
    types.int64: np.int64,
    types.uint8: np.uint8,
    types.uint16: np.uint16,
    types.uint32: np.uint32,
    types.uint64: np.uint64,
    types.float32: np.float32,
    types.float64: np.float64,
}


def random_int(shape, dtype):
    return np.random.randint(0, 128, size=shape).astype(dtype)


@cuda.jit(device=True)
def row_major_tid():
    dim = cuda.blockDim
    idx = cuda.threadIdx
    return (
        (0 if dim.z == 1 else idx.z * dim.x * dim.y)
        + (0 if dim.y == 1 else idx.y * dim.x)
        + idx.x
    )
