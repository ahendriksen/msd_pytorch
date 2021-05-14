#include <cuda.h>
#include <cuda_runtime.h>

#include "device_tensor.h"
#include "kernel_utils.cuh"


template <typename scalar_t>
static __inline__ __device__ void pack_shared_mem(scalar_t* dest, scalar_t* src, int n){
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    for (int i=pId; i < n; i+=num_threads) {
        dest[i] = src[i];
    }
    __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
//                            Convolution:Forward                            //
///////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void
conv3d_forward(dTensor5R input,
	       dTensor5R kernel,
	       dTensor1R bias,
	       dTensor5R output,
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
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int d = threadIdx.z + blockDim.z * blockIdx.z;
    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int num_threads = blockDim.x * blockDim.y * blockDim.z;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    int num_kernel_elems = kernel.size(0) * kernel.size(1) * kernel.size(2) * kernel.size(3) * kernel.size(4);

    for (int i=pId; i < num_kernel_elems; i+=num_threads) {
        kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 5d tensor.
    mcc::TensorAccessor<PT5R32> kernel_buffer = kernel.unpack_from(kernel_buf);

    __syncthreads();

    if (W <= w || H <= h || D <= d) {
        return;
    }

    // Precompute data offsets:
    int ds[3] = {0};
    int hs[3] = {0};
    int ws[3] = {0};

    for (int i=-1; i <= 1; i++) {
        ds[i + 1] = reflect(d + dilation * i, (int) D);
        hs[i + 1] = reflect(h + dilation * i, (int) H);
        ws[i + 1] = reflect(w + dilation * i, (int) W);
    }

    // Actually compute the convolution
    for (int b=0; b < B; b++) {
        for (int c_out=0; c_out < C_OUT; c_out++) {
            scalar_t o = bias[c_out];
            for (int c_in=0; c_in < C_IN; c_in++) {
                scalar_t *kernel0 = &kernel_buffer[c_out][c_in][0][0][0];
                #pragma unroll
                for (int p=-1; p <= 1; p++) {
                    #pragma unroll
                    for (int q=-1; q <= 1; q++) {
                        #pragma unroll
                        for (int r=-1; r <= 1; r++) {
                            o += input[b][c_in][ds[p + 1]][hs[q + 1]][ws[r + 1]] * (*kernel0);
                            // Incrementing the kernel pointer works because
                            // the kernel weights are contiguous and the
                            // data_offsets are prepared to be in the same
                            // order as the kernel weights.
                            kernel0++;
                        }
                    }
                }
            }
            output[b][c_out][d][h][w] = o;
        }
    }
}

template <typename scalar_t>
__global__ void
conv3d_relu_forward(dTensor5R input,
		    dTensor5R kernel,
		    dTensor1R bias,
		    dTensor5R output,
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
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int d = threadIdx.z + blockDim.z * blockIdx.z;
    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int num_threads = blockDim.x * blockDim.y * blockDim.z;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    int num_kernel_elems = kernel.size(0) * kernel.size(1) * kernel.size(2) * kernel.size(3) * kernel.size(4);

    for (int i=pId; i < num_kernel_elems; i+=num_threads) {
        kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 5d tensor.
    mcc::TensorAccessor<PT5R32> kernel_buffer = kernel.unpack_from(kernel_buf);

    __syncthreads();

    if (W <= w || H <= h || D <= d) {
        return;
    }

    // Precompute data offsets:
    int ds[3] = {0};
    int hs[3] = {0};
    int ws[3] = {0};

    for (int i=-1; i <= 1; i++) {
        ds[i + 1] = reflect(d + dilation * i, (int) D);
        hs[i + 1] = reflect(h + dilation * i, (int) H);
        ws[i + 1] = reflect(w + dilation * i, (int) W);
    }

    // Actually compute the convolution
    for (int b=0; b < B; b++) {
        for (int c_out=0; c_out < C_OUT; c_out++) {
            scalar_t o = bias[c_out];
            for (int c_in=0; c_in < C_IN; c_in++) {
                scalar_t *kernel0 = &kernel_buffer[c_out][c_in][0][0][0];
                #pragma unroll
                for (int p=-1; p <= 1; p++) {
                    #pragma unroll
                    for (int q=-1; q <= 1; q++) {
                        #pragma unroll
                        for (int r=-1; r <= 1; r++) {
                            o += input[b][c_in][ds[p + 1]][hs[q + 1]][ws[r + 1]] * (*kernel0);
                            // Incrementing the kernel pointer works because
                            // the kernel weights are contiguous and the
                            // data_offsets are prepared to be in the same
                            // order as the kernel weights.
                            kernel0++;
                        }
                    }
                }
            }
            output[b][c_out][d][h][w] = max(0.0, o);
        }
    }
}


template
__global__ void
conv3d_forward<float>(dTensor5Rfloat input,
		      dTensor5Rfloat kernel,
		      dTensor1Rfloat bias,
		      dTensor5Rfloat output,
		      int dilation);

template
__global__ void
conv3d_forward<double>(dTensor5Rdouble input,
		       dTensor5Rdouble kernel,
		       dTensor1Rdouble bias,
		       dTensor5Rdouble output,
		       int dilation);

template
__global__ void
conv3d_relu_forward<float>(dTensor5Rfloat input,
			   dTensor5Rfloat kernel,
			   dTensor1Rfloat bias,
			   dTensor5Rfloat output,
			   int dilation);

template
__global__ void
conv3d_relu_forward<double>(dTensor5Rdouble input,
			    dTensor5Rdouble kernel,
			    dTensor1Rdouble bias,
			    dTensor5Rdouble output,
			    int dilation);