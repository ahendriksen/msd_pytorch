#!/usr/bin/env python
from collections import OrderedDict
import os
import numpy as np
# import torch
import kernel_tuner


def shape_to_str(x):
    shape = [str(s) for s in x.shape]
    return ", ".join(shape)


def strides_to_str(x):
    itemsize = x.dtype.itemsize
    strides = [str(s // itemsize) for s in x.strides ]
    return ", ".join(strides)

def tune():
    c_in = 5
    c_out = 1
    N = 128    # Create data

    x = np.ones((1, c_in, N, N), dtype=np.float32)
    w = np.zeros((c_out, c_in, 3, 3), dtype=np.float32)
    w[:, :, 2, 2] = 1.0
    bias = np.ones((c_out), dtype=np.float32)
    y = np.zeros((1, c_out, N, N), dtype=np.float32)

    dilation = np.int32(1)

    args = [x, w, bias, y, dilation]

    # Size of shared memory
    smem_args = dict(size=w.size * 4)

    size = (N, N)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [64, 32, 16, 8, 4, 2]
    tune_params["block_size_y"] = [64, 32, 16, 8, 4, 2]

    tune_params["INPUT_SHAPE"] = [shape_to_str(x)]
    tune_params["INPUT_STRIDES"] = [strides_to_str(x)]

    tune_params["KERNEL_SHAPE"] = [shape_to_str(w)]
    tune_params["KERNEL_STRIDES"] = [strides_to_str(w)]

    tune_params["BIAS_SHAPE"] = [shape_to_str(bias)]
    tune_params["BIAS_STRIDES"] = [strides_to_str(bias)]

    tune_params["OUTPUT_SHAPE"] = [shape_to_str(y)]
    tune_params["OUTPUT_STRIDES"] = [strides_to_str(y)]

    # Running the kernel does not yet work...
    # run_params = {k: v[0] for k, v in tune_params.items()}
    # test_res = kernel_tuner.run_kernel(
    #     "conv_forward_tuner",
    #     "conv_kernel_tuner.cu",
    #     size,
    #     args,
    #     run_params,
    #     smem_args=smem_args,
    #     compiler_options=["-I" + os.getcwd()]
    # )
    # print("mean output: ", test_res[3].mean())

    result, env = kernel_tuner.tune_kernel(
        "conv_forward_tuner",
        "conv_kernel_tuner.cu",
        size,
        args,
        tune_params,
        smem_args=smem_args,
        compiler_options=[
            "-I" + os.getcwd(),
            # Generate code that is compatible with Geforce 10XX and up.
            "-gencode=arch=compute_61,code=sm_61",
        ]
    )


if __name__ == "__main__":

    tune()
