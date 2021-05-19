from time import perf_counter as timer
import torch
import msd_custom_convolutions as cc
import numpy as np
from functools import partial
import argparse
import json
from tqdm import tqdm


def timeit(f, it):
    torch.cuda.synchronize()
    start = timer()
    f(it)
    torch.cuda.synchronize()
    return timer() - start


def measure_small_execution_time(f,
                                 num_iters=[0,1,2,3],
                                 num_trials=2,
                                 ci=True):
    """Measures the execution time of a function f

    `f` must take an int `it` and compute `it` iterations.

    This function estimates how long a single iteration works using the
    methodology proposed in

    Moreno, C., & Fischmeister, S. (2017). Accurate measurement of small
    execution times—getting around measurement errors. IEEE Embedded Systems
    Letters, 9(1), 17–20. http://dx.doi.org/10.1109/les.2017.2654160

    The function returns:
    1. an estimate for the execution time of a single iteration, and
    2. a 90% confidence interval for the estimate (if `ci=True`).

    """
    try:
        # Warmup
        f(max(num_iters))
        # Measure
        num_iters = np.array(list(num_iters) * num_trials)
        timings = np.array([timeit(f, it) for it in tqdm(num_iters, leave=False)])

        slope, intercept = np.polyfit(num_iters, timings, deg=1)

        if not ci:
            return slope

        # Follows exposition in:
        # https://en.wikipedia.org/wiki/Simple_linear_regression#Confidence_intervals
        n = len(timings)
        timings_hat = slope * num_iters + intercept
        error = timings_hat - timings
        s_beta_hat = np.sqrt(
            1 / (n - 2) * np.sum(error ** 2) /
            np.sum((num_iters - num_iters.mean()) ** 2)
        )
        # Sample a million elements form a standard_t distribution for 90%
        # confidence interval
        N = 1_000_000
        t = np.sort(np.random.standard_t(n - 2, N))
        ci_5, ci_95 = t[5 * N // 100], t[95 * N // 100]

        ci = (slope + ci_5 * s_beta_hat, slope + ci_95 * s_beta_hat)
        return slope, ci

    except RuntimeError:
        return np.inf, (np.inf, np.inf)


def conv3d_forward(input, weight, bias, output, grad_output, grad_input, grad_weight, dilation, block_size=(8, 8, 8)):
    cc.conv_forward(input, weight, bias, output, dilation, block_size=block_size)


def conv3d_backward_x(input, weight, bias, output, grad_output, grad_input, grad_weight, dilation, block_size=(8, 8, 8)):
    cc.conv_backward_x(grad_output, weight, grad_input, dilation, block_size=tuple(block_size))


def conv3d_backward_k(input, weight, bias, output, grad_output, grad_input, grad_weight, dilation, block_size=(8, 8, 8)):
    cc.conv_backward_k(grad_output, input, grad_weight, dilation, block_size=block_size)


def conv3d_relu_forward(input, weight, bias, output, grad_output, grad_input, grad_weight, dilation, block_size=(8, 8, 8)):
    cc.conv_relu_forward(input, weight, bias, output, dilation, block_size=block_size)


def conv3d_relu_backward_x(input, weight, bias, output, grad_output, grad_input, grad_weight, dilation, block_size=(8, 8, 8)):
    cc.conv_relu_backward_x(output, grad_output, weight, grad_input, dilation, block_size=tuple(block_size))


def conv3d_relu_backward_k(input, weight, bias, output, grad_output, grad_input, grad_weight, dilation, block_size=(8, 8, 8)):
    cc.conv_relu_backward_k(output, grad_output, input, grad_weight, dilation, block_size=block_size)


# def conv3d_backward_bias(grad_output, grad_bias, block_size=(8, 8, 8)):
#     cc.conv_backward_bias(grad_output, grad_bias, block_size=block_size)

# def conv3d_relu_backward_bias(output, grad_output, grad_bias, block_size=(8, 8, 8)):
#     cc.conv_relu_backward_bias(output, grad_output, grad_bias, block_size=block_size)

functions = {
    'conv3d_forward': (512, conv3d_forward),
    'conv3d_backward_x': (512, conv3d_backward_x),
    'conv3d_backward_k': (512, conv3d_backward_k),
    'conv3d_relu_forward': (512, conv3d_relu_forward),
    'conv3d_relu_backward_x': (512, conv3d_relu_backward_x),
    'conv3d_relu_backward_k': (512, conv3d_relu_backward_k),
}


def iter_conv(iterations, c_in=1, c_out=1, N=32, dilation=1, block_size=(4, 4, 4), fun="conv_backward_x"):
    # Determine convolution function to use
    _, f = functions[fun]

    # Create data        (constant time)
    x = torch.randn(1, c_in, N, N, N).cuda()
    grad_x = torch.randn(1, c_in, N, N, N).cuda()
    w = torch.randn(c_out, c_in, 3, 3, 3).cuda()
    grad_w = torch.randn(c_out, c_in, 3, 3, 3).cuda()
    bias = torch.randn(c_out).cuda()
    y = torch.randn(1, c_out, N, N, N).cuda()
    grad_y = torch.randn(1, c_out, N, N, N).cuda()

    # Execute function                   (variable time)
    for _ in range(iterations):
        f(x, w, bias, y, grad_y, grad_x, grad_w, dilation, block_size)
        torch.cuda.synchronize()


def main(args):
    if args.csv:
        print("function, N, time, time_ci_min, time_ci_max, block_size")
    else:
        print(cc)

    num_trials = args.num_trials
    for N in args.N:
        if "all" in args.functions:
            arg_functions = list(functions.keys())
        else:
            arg_functions = args.functions

        for fun in arg_functions:
            if not args.csv:
                print()
                print("------------------------------------------------------------")
                print(f"-- {fun}")
                print("------------------------------------------------------------")

            max_threads, _ = functions[fun]
            pows_of_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

            if args.block_sizes is None:
                block_sizes = []
                for block_z in [b for b in pows_of_2 if b <= max_threads]:
                    for block_y in [b for b in pows_of_2 if block_z * b <= max_threads]:
                        for block_x in [b for b in pows_of_2 if block_z * block_y * b <= max_threads]:
                            block_sizes.append((block_z, block_y, block_x))
            else:
                block_sizes = list(map(tuple, args.block_sizes))

            for block_size in block_sizes:
                f = partial(iter_conv, N=N, c_in=10, block_size=block_size, fun=fun)
                slope, (s, S) = measure_small_execution_time(f, num_trials=num_trials)
                if args.csv:
                    print(f"{fun}, {N:04d}, {slope:0.3e}, {s:0.3e}, {S:0.3e}, {repr(block_size).replace(', ', '_')}")
                else:
                    print(f"{N:04d}: {slope:0.3e} in [{s:0.3e} -- {S:0.3e}] ({block_size})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", default=30, type=int)
    parser.add_argument("--N", default=[256], nargs="+", type=int)
    parser.add_argument("--block_sizes", default=None, type=json.loads)
    parser.add_argument("--csv", default=False, action='store_const', const='True')
    parser.add_argument(
        "--functions",
        default=["conv3d_backward_x", "conv3d_relu_backward_x"],
        nargs="+",
        type=str,
    )
    args = parser.parse_args()
    main(args)
