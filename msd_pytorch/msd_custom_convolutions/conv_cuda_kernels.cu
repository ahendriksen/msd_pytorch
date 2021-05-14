#include <cuda.h>
#include <cuda_runtime.h>

#include "device_tensor.h"
#include "kernel_utils.cuh"



///////////////////////////////////////////////////////////////////////////////
//                           Convolution: Backward Input                     //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_backward_x(dTensor4R grad_output,
                dTensor4R kernel,
                dTensor4R grad_input,
                int dilation)
{
    // Performance improvements:
    // 1) Store kernel in shared memory.

    // LIMITS:
    //    49152 bytes of shared memory per block
    //    12288 floats of shared memory per block
    // +-  1300 kernels can be stored in shared mem
    // So we must have:
    //     C_IN * C_OUT < 1300
    extern __shared__ int shared_memory[];

    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int C_IN = grad_input.size(1);
    int H = grad_output.size(2);
    int W = grad_output.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    int num_kernel_elems = kernel.size(0) * kernel.size(1) * kernel.size(2) * kernel.size(3);
    for (int i=pId; i < num_kernel_elems; i+=num_threads) {
        kernel_buf[i] = kernel.data()[i];
    }
    __syncthreads();

    // We can index kernel_buffer like a 4d tensor.
    mcc::TensorAccessor<PT4R32> kernel_buffer = kernel.unpack_from(kernel_buf);

    if (W <= w || H <= h) {
        return;
    }

    // Calculate right procession through filter if we are at the border.
    int prog_h[] = {0, 1, 2};
    if (h < dilation) {
        prog_h[2] = 0;
    }
    if (H <= h + dilation) {
        prog_h[0] = 2;
    }

    int prog_w[] = {0, 1, 2};
    if (w < dilation) {
        prog_w[2] = 0;
    }
    if (W <= w + dilation) {
        prog_w[0] = 2;
    }

    DT_INDEX data_offsets[9];
    DT_INDEX kernel_offsets[9];
    int i = 0;
    for (int p=-1; p <= 1; p++) {
        for (int q=-1; q <= 1; q++) {
            int hp = reflect(h - dilation * p, (int) H);
            int wq = reflect(w - dilation * q, (int) W);
            data_offsets[i] = ((&grad_output[0][0][hp][wq]) - (&grad_output[0][0][0][0]));
            int p_ = prog_h[p+1];
            int q_ = prog_w[q+1];
            kernel_offsets[i] =  ((&kernel_buffer[0][0][p_][q_]) - (&kernel_buffer[0][0][0][0]));
            i++;
        }
    }

    for (int b=0; b < B; b++) {
        for (int c_in=0; c_in < C_IN; c_in++) {
            scalar_t g = 0;
            for (int c_out=0; c_out < C_OUT; c_out++) {
                int i = 0;
                scalar_t *data = &grad_output[b][c_out][0][0];
                scalar_t *kdata = &kernel_buffer[c_out][c_in][0][0];
                for (int p=-1; p <= 1; p++) {
                    for (int q=-1; q <= 1; q++) {
                        g += *(data + data_offsets[i]) *
                            *(kdata + kernel_offsets[i]);
                        i++;
                    }
                }
            }
            grad_input[b][c_in][h][w] += g;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
//                            Convolution:Forward                            //
///////////////////////////////////////////////////////////////////////////////


template <typename scalar_t>
__global__ void
conv_forward(dTensor4R input,
             dTensor4R kernel,
             dTensor1R bias,
             dTensor4R output,
             int dilation)
{
    // The following has been done to improve performance:
    // 1) This implementation caches the kernel weights.
    // 2) This implementation precomputes data offsets in x and y
    //    direction instead of pointers.

    // LIMITS:
    //    49152 bytes of shared memory per block
    //    12288 floats of shared memory per block
    // +-  1300 kernels can be stored in shared mem
    // So we must have:
    //     C_IN * C_OUT < 1300
    extern __shared__ int shared_memory[];

    int B = output.size(0);
    int C_OUT = output.size(1);
    int C_IN = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    int num_kernel_elems = kernel.size(0) * kernel.size(1) * kernel.size(2) * kernel.size(3);

    for (int i=pId; i < num_kernel_elems; i+=num_threads) {
        kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 4d tensor.
    mcc::TensorAccessor<PT4R32> kernel_buffer = kernel.unpack_from(kernel_buf);

    __syncthreads();

    if (W <= w || H <= h) {
        return;
    }

    // Precompute data offsets:
    int hs[3] = {0};
    int ws[3] = {0};

    for (int i=-1; i <= 1; i++) {
        hs[i + 1] = reflect(h + dilation * i, (int) H);
        ws[i + 1] = reflect(w + dilation * i, (int) W);
    }

    // Actually compute the convolution
    for (int b=0; b < B; b++) {
        for (int c_out=0; c_out < C_OUT; c_out++) {
            scalar_t o = bias[c_out];
            for (int c_in=0; c_in < C_IN; c_in++) {
                scalar_t *kernel0 = &kernel_buffer[c_out][c_in][0][0];
                #pragma unroll
                for (int p=-1; p <= 1; p++) {
                    #pragma unroll
                    for (int q=-1; q <= 1; q++) {
                        o += input[b][c_in][hs[p + 1]][ws[q + 1]] * (*kernel0);
                        // Incrementing the kernel pointer works because
                        // the kernel weights are contiguous and the
                        // data_offsets are prepared to be in the same
                        // order as the kernel weights.
                        kernel0++;
                    }
                }
            }
            output[b][c_out][h][w] = o;
        }
    }
}