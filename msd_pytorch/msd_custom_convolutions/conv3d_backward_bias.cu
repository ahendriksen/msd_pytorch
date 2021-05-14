#include <cuda.h>
#include <cuda_runtime.h>

#include "device_tensor.h"
#include "kernel_utils.cuh"


///////////////////////////////////////////////////////////////////////////////
//                         Convolution: Backward Bias                        //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv3d_relu_backward_bias(dTensor5R output,
			  dTensor5R grad_output,
			  dTensor1R grad_bias)
{
    // This implementation of the backward pass wrt the convolution
    // bias uses warp reduction.

    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int D = grad_output.size(2);
    int H = grad_output.size(3);
    int W = grad_output.size(4);

    int d = threadIdx.z + blockDim.z * blockIdx.z;
    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    // Used for warpReduceSum:
    bool leader = 0 == pId % 32;
    int mask = __activemask();

    if (W <= w || H <= h || D <= d) {
        return;
    }
    for (int c_out=0; c_out < C_OUT; c_out++) {
        scalar_t g = 0;
        for (int b=0; b < B; b++) {
            if (0.0 < output[b][c_out][d][h][w]) {
                g += grad_output[b][c_out][d][h][w];
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
conv3d_relu_backward_bias<float>(dTensor5Rfloat output,
				 dTensor5Rfloat grad_output,
				 dTensor1Rfloat grad_bias);
template
__global__ void
conv3d_relu_backward_bias<double>(dTensor5Rdouble output,
				  dTensor5Rdouble grad_output,
				  dTensor1Rdouble grad_bias);