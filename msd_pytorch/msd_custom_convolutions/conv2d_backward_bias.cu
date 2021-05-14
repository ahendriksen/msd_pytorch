#include <cuda.h>
#include <cuda_runtime.h>

#include "device_tensor.h"
#include "kernel_utils.cuh"


///////////////////////////////////////////////////////////////////////////////
//                         Convolution: Backward Bias                        //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_relu_backward_bias(dTensor4R output,
                        dTensor4R grad_output,
                        dTensor1R grad_bias)
{
    // This implementation of the backward pass wrt the convolution
    // bias uses warp reduction.

    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int H = grad_output.size(2);
    int W = grad_output.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.y * blockDim.x + threadIdx.x;

    // Used for warpReduceSum:
    bool leader = 0 == pId % 32;
    int mask = __activemask();

    if (W <= w || H <= h) {
        return;
    }
    for (int c_out=0; c_out < C_OUT; c_out++) {
        scalar_t g = 0;
        for (int b=0; b < B; b++) {
            if (0.0 < output[b][c_out][h][w]) {
                g += grad_output[b][c_out][h][w];
            }
        }
        g = warpReduceSum(mask, g);
        if (leader) {
            atomicAdd(&grad_bias[c_out], g);
        }
    }
}

// Specializations:
template
__global__ void
conv_relu_backward_bias<float>(dTensor4Rfloat output,
			       dTensor4Rfloat grad_output,
			       dTensor1Rfloat grad_bias);
template
__global__ void
conv_relu_backward_bias<double>(dTensor4Rdouble output,
			       dTensor4Rdouble grad_output,
			       dTensor1Rdouble grad_bias);