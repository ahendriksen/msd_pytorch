#include <cuda.h>
#include <cuda_runtime.h>

#include "device_tensor.h"
#include "kernel_utils.cuh"

///////////////////////////////////////////////////////////////////////////////
//                        Convolution: Backward Kernel                       //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv3d_backward_k(dTensor5R grad_output,
                  dTensor5R input,
                  dTensor5R grad_kernel,
                  int dilation)
{
    // In this approach, the gradient sums are reduced accross the
    // warp before being written to global memory.  This
    // implementation does not perform too badly.
    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int C_IN = input.size(1);
    int D = grad_output.size(2);
    int H = grad_output.size(3);
    int W = grad_output.size(4);

    int d = threadIdx.z + blockDim.z * blockIdx.z;
    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    if (W <= w || H <= h || D <= d) {
        return;
    }

    bool leader = 0 == pId % 32;
    int mask = __activemask();

    for (int b=0; b < B; b++) {
        for (int c_in=0; c_in < C_IN; c_in++) {
            for (int c_out=0; c_out < C_OUT; c_out++) {
                for (int p=-1; p <= 1; p++) {
                    for (int q=-1; q <= 1; q++) {
                        for (int r=-1; r <= 1; r++) {
                            int d_ = reflect(d + p * dilation, (int) D);
                            int h_ = reflect(h + q * dilation, (int) H);
                            int w_ = reflect(w + r * dilation, (int) W);

                            scalar_t g =
                                input[b][c_in][d_][h_][w_] *
                                grad_output[b][c_out][d][h][w];

                            g = warpReduceSum(mask, g);
                            if (leader) {
                                atomicAdd(&grad_kernel[c_out][c_in][p+1][q+1][r+1], g);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void
conv3d_relu_backward_k(dTensor5R output,
                       dTensor5R grad_output,
                       dTensor5R input,
                       dTensor5R grad_kernel,
                       int dilation)
{
    // In this approach, the gradient sums are reduced accross the
    // warp before being written to global memory.  This
    // implementation does not perform too badly.
    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int C_IN = input.size(1);
    int D = grad_output.size(2);
    int H = grad_output.size(3);
    int W = grad_output.size(4);

    int d = threadIdx.z + blockDim.z * blockIdx.z;
    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    if (W <= w || H <= h || D <= d) {
        return;
    }

    bool leader = 0 == pId % 32;
    int mask = __activemask();

    for (int b=0; b < B; b++) {
        for (int c_in=0; c_in < C_IN; c_in++) {
            for (int c_out=0; c_out < C_OUT; c_out++) {
                for (int p=-1; p <= 1; p++) {
                    for (int q=-1; q <= 1; q++) {
                        for (int r=-1; r <= 1; r++) {
                            int d_ = reflect(d + p * dilation, (int) D);
                            int h_ = reflect(h + q * dilation, (int) H);
                            int w_ = reflect(w + r * dilation, (int) W);
                            // Only propagate gradient if relu is positive.
                            scalar_t propagate = (0.0 < output[b][c_out][d][h][w]) ? 1.0 : 0.0;

                            scalar_t g =
                                input[b][c_in][d_][h_][w_] *
                                grad_output[b][c_out][d][h][w] *
                                propagate;
                            g = warpReduceSum(mask, g);
                            if (leader) {
                                atomicAdd(&grad_kernel[c_out][c_in][p+1][q+1][r+1], g);
                            }
                        }
                    }
                }
            }
        }
    }
}

template
__global__ void
conv3d_backward_k<float>(dTensor5Rfloat grad_output,
                         dTensor5Rfloat input,
                         dTensor5Rfloat grad_kernel,
                         int dilation);

template
__global__ void
conv3d_backward_k<double>(dTensor5Rdouble grad_output,
                          dTensor5Rdouble input,
                          dTensor5Rdouble grad_kernel,
                          int dilation);

template
__global__ void
conv3d_relu_backward_k<float>(dTensor5Rfloat output,
                              dTensor5Rfloat grad_output,
                              dTensor5Rfloat input,
                              dTensor5Rfloat grad_kernel,
                              int dilation);

template
__global__ void
conv3d_relu_backward_k<double>(dTensor5Rdouble output,
                               dTensor5Rdouble grad_output,
                               dTensor5Rdouble input,
                               dTensor5Rdouble grad_kernel,
                               int dilation);