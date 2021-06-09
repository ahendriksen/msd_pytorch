#!/usr/bin/env python
from collections import OrderedDict
import os
import numpy as np
# import torch
import kernel_tuner


def strides_to_str(x):
    itemsize = x.dtype.itemsize
    strides = [str(s // itemsize) for s in x.strides ]
    return ", ".join(strides)

def tune():
    c_in = 1
    c_out = 1
    N = 10    # Create data

    x = np.ones((1, c_in, N, N), dtype=np.float32)
    w = np.ones((c_out, c_in, 3, 3), dtype=np.float32)
    bias = np.ones((c_out), dtype=np.float32)
    y = np.ones((1, c_out, N, N), dtype=np.float32)

    dilation = np.int32(1)
    args = [x, w, bias, y, dilation]

    # Size of shared memory
    smem_args = dict(size=w.size * 4)

    size = (N, N)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16, 8]
    tune_params["block_size_y"] = [16, 8]
    tune_params["INPUT_STRIDES"] = [strides_to_str(x)]

    result, env = kernel_tuner.tune_kernel(
        "conv_forward_tuner",
        "conv_kernel_tuner.cu",
        size,
        args,
        tune_params,
        smem_args=smem_args,
        compiler_options=["-I" + os.getcwd()]
    )


if __name__ == "__main__":

    tune()
