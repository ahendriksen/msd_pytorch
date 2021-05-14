#include <cuda.h>
#include <cuda_runtime.h>

#include "device_tensor.h"
#include "kernel_utils.cuh"

///////////////////////////////////////////////////////////////////////////////
//                        Convolution: Backward Kernel                       //
///////////////////////////////////////////////////////////////////////////////


template <typename scalar_t>
__global__ void
conv_backward_k(dTensor4R grad_output,
                dTensor4R input,
                dTensor4R grad_kernel,
                int dilation)
{
    // A less naive implementation of the backward pass wrt the
    // convolution kernel weights. Here, the gradient sums are reduced
    // accross the warp before being written to global memory.  This
    // implementation does not perform too badly.
    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int C_IN = input.size(1);
    int H = grad_output.size(2);
    int W = grad_output.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.y * blockDim.x + threadIdx.x;

    if (W <= w || H <= h) {
        return;
    }

    bool leader = 0 == pId % 32;
    int mask = __activemask();

    for (int b=0; b < B; b++) {
        for (int c_in=0; c_in < C_IN; c_in++) {
            for (int c_out=0; c_out < C_OUT; c_out++) {
                for (int p=-1; p <= 1; p++) {
                    for (int q=-1; q <= 1; q++) {
                        int h_ = reflect(h + p * dilation, (int) H);
                        int w_ = reflect(w + q * dilation, (int) W);
                        scalar_t g =
                            input[b][c_in][h_][w_] *
                            grad_output[b][c_out][h][w];
                        g = warpReduceSum(mask, g);
                        if (leader) {
                            atomicAdd(&grad_kernel[c_out][c_in][p+1][q+1], g);
                        }
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void
conv_relu_backward_k(dTensor4R output,
                     dTensor4R grad_output,
                     dTensor4R input,
                     dTensor4R grad_kernel,
                     int dilation)
{
    // In this approach, the gradient sums are reduced accross the
    // warp before being written to global memory.  This
    // implementation does not perform too badly.
    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int C_IN = input.size(1);
    int H = grad_output.size(2);
    int W = grad_output.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.y * blockDim.x + threadIdx.x;

    if (W <= w || H <= h) {
        return;
    }

    bool leader = 0 == pId % 32;
    int mask = __activemask();

    for (int b=0; b < B; b++) {
        for (int c_in=0; c_in < C_IN; c_in++) {
            for (int c_out=0; c_out < C_OUT; c_out++) {
                for (int p=-1; p <= 1; p++) {
                    for (int q=-1; q <= 1; q++) {
                        int h_ = reflect(h + p * dilation, (int) H);
                        int w_ = reflect(w + q * dilation, (int) W);
                        // Only propagate gradient if relu is positive.
                        scalar_t propagate = (0.0 < output[b][c_out][h][w]) ? 1.0 : 0.0;

                        scalar_t g =
                            input[b][c_in][h_][w_] *
                            grad_output[b][c_out][h][w] *
                            propagate;
                        g = warpReduceSum(mask, g);
                        if (leader) {
                            atomicAdd(&grad_kernel[c_out][c_in][p+1][q+1], g);
                        }
                    }
                }
            }
        }
    }
}


// Specialize:

template
__global__ void
conv_backward_k<float>(dTensor4Rfloat grad_output,
		       dTensor4Rfloat input,
		       dTensor4Rfloat grad_kernel,
		       int dilation);

template
__global__ void
conv_backward_k<double>(dTensor4Rdouble grad_output,
		       dTensor4Rdouble input,
		       dTensor4Rdouble grad_kernel,
		       int dilation);

template
__global__ void
conv_relu_backward_k<float>(dTensor4Rfloat output,
			    dTensor4Rfloat grad_output,
			    dTensor4Rfloat input,
			    dTensor4Rfloat grad_kernel,
			    int dilation);

template
__global__ void
conv_relu_backward_k<double>(dTensor4Rdouble output,
			     dTensor4Rdouble grad_output,
			     dTensor4Rdouble input,
			     dTensor4Rdouble grad_kernel,
			     int dilation);