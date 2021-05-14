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


template <typename scalar_t>
__global__ void
conv_relu_backward_x(dTensor4R output,
                      dTensor4R grad_output,
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

    DT_INDEX output_offsets[9];
    DT_INDEX grad_output_offsets[9];
    DT_INDEX kernel_offsets[9];
    int i = 0;
    for (int p=-1; p <= 1; p++) {
        for (int q=-1; q <= 1; q++) {
            int hp = reflect(h - dilation * p, (int) H);
            int wq = reflect(w - dilation * q, (int) W);
            output_offsets[i] = ((&output[0][0][hp][wq]) - (&output[0][0][0][0]));
            grad_output_offsets[i] = ((&grad_output[0][0][hp][wq]) - (&grad_output[0][0][0][0]));
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
                scalar_t *grad_o_p = &grad_output[b][c_out][0][0];
                scalar_t *o_p = &output[b][c_out][0][0];
                scalar_t *kdata = &kernel_buffer[c_out][c_in][0][0];
                for (int p=-1; p <= 1; p++) {
                    for (int q=-1; q <= 1; q++) {
                        if (0.0 < *(o_p + output_offsets[i])) {
                            g += *(grad_o_p + grad_output_offsets[i]) *
                                *(kdata + kernel_offsets[i]);
                        }
                        i++;
                    }
                }
            }
            grad_input[b][c_in][h][w] += g;
        }
    }
}

template
__global__ void
conv_backward_x<float>(dTensor4Rfloat grad_output,
		      dTensor4Rfloat kernel,
		      dTensor4Rfloat grad_input,
		      int dilation);

template
__global__ void
conv_backward_x<double>(dTensor4Rdouble grad_output,
			dTensor4Rdouble kernel,
			dTensor4Rdouble grad_input,
			int dilation);

template
__global__ void
conv_relu_backward_x<float>(dTensor4Rfloat output,
			    dTensor4Rfloat grad_output,
			    dTensor4Rfloat kernel,
			    dTensor4Rfloat grad_input,
			    int dilation);

template
__global__ void
conv_relu_backward_x<double>(dTensor4Rdouble output,
			    dTensor4Rdouble grad_output,
			    dTensor4Rdouble kernel,
			    dTensor4Rdouble grad_input,
			    int dilation);