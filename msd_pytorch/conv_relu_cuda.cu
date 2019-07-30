// -*- eval:(c++-mode); c-file-style: "bsd"; -*-
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "THC/THC.h"
#include "THC/THCDeviceTensor.cuh"
#include "THC/THCAtomics.cuh"
#include "THC/THCDeviceUtils.cuh"
#include "device_tensor.h"

using at::OptionalDeviceGuard;

__device__ __forceinline__ int
reflect(int i, int dimi) {
    if (i < 0) {
	i = -i - 1;
    } else if (i >= dimi) {
	i = 2 * dimi - i - 1;
    }
    return i;
}

template <typename scalar_t>
__inline__ __device__
scalar_t warpReduceSum(int mask, scalar_t val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
      val += __shfl_down_sync(mask, val, offset);
  return val;
}

///////////////////////////////////////////////////////////////////////////////
//                         Convolution: Backward Bias                        //
///////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void
conv_relu_backward_bias0(dTensor4R output,
			 dTensor4R grad_output,
			 dTensor1R grad_bias)
{
    // A very naive implementation of the backward pass wrt the
    // convolution bias.
    int B = grad_output.getSize(0);
    int C_OUT = grad_output.getSize(1);
    int H = grad_output.getSize(2);
    int W = grad_output.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;

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
	atomicAdd(&grad_bias[c_out], g);
    }
}

template <typename scalar_t>
__global__ void
conv_relu_backward_bias1(dTensor4R output,
			 dTensor4R grad_output,
			 dTensor1R grad_bias)
{
    // An implementation of the backward pass wrt the
    // convolution bias that uses warp reduction.

    int B = grad_output.getSize(0);
    int C_OUT = grad_output.getSize(1);
    int H = grad_output.getSize(2);
    int W = grad_output.getSize(3);

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
///////////////////////////////////////////////////////////////////////////////
//                        Convolution: Backward Kernel                       //
///////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void
conv_relu_backward_k0(dTensor4R output,
		      dTensor4R grad_output,
		      dTensor4R input,
		      dTensor4R grad_kernel,
		      int dilation)
{
    // A very naive implementation of the backward pass wrt the
    // convolution kernel weights. This is very very slow.
    int B = grad_output.getSize(0);
    int C_OUT = grad_output.getSize(1);
    int C_IN = input.getSize(1);
    int H = grad_output.getSize(2);
    int W = grad_output.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;

    if (W <= w || H <= h) {
	return;
    }

    for (int b=0; b < B; b++) {
	for (int c_in=0; c_in < C_IN; c_in++) {
	    for (int c_out=0; c_out < C_OUT; c_out++) {
		for (int p=-1; p <= 1; p++) {
		    for (int q=-1; q <= 1; q++) {
			int h_ = reflect(h + p * dilation, (int) H);
			int w_ = reflect(w + q * dilation, (int) W);
			if (0.0 < output[b][c_out][h][w]) {
			    atomicAdd(&grad_kernel[c_out][c_in][p+1][q+1],
				      input[b][c_in][h_][w_] *
				      grad_output[b][c_out][h][w]);
			}
		    }
		}
	    }
	}
    }
}

template <typename scalar_t>
__global__ void
conv_relu_backward_k1(dTensor4R output,
		      dTensor4R grad_output,
		      dTensor4R input,
		      dTensor4R grad_kernel,
		      int dilation)
{
    // A less naive approach where the gradient sums are reduced
    // accross the warp before being written to global memory.
    // This implementation does not perform too badly.
    int B = grad_output.getSize(0);
    int C_OUT = grad_output.getSize(1);
    int C_IN = input.getSize(1);
    int H = grad_output.getSize(2);
    int W = grad_output.getSize(3);

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


///////////////////////////////////////////////////////////////////////////////
//                           Convolution: Backward Input                     //
///////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void
conv_relu_backward_x0(dTensor4R output,
		      dTensor4R grad_output,
		      dTensor4R kernel,
		      dTensor4R grad_input,
		      int dilation)
{
    int B = grad_output.getSize(0);
    int C_OUT = grad_output.getSize(1);
    int C_IN = grad_input.getSize(1);
    int H = grad_output.getSize(2);
    int W = grad_output.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;

    if (W <= w || H <= h) {
	return;
    }

    int pi[] = {0, 1, 2};
    if (h < dilation) {
	pi[2] = 0;
    }
    if (H <= h + dilation) {
	pi[0] = 2;
    }

    int qi[] = {0, 1, 2};
    if (w < dilation) {
	qi[2] = 0;
    }
    if (W <= w + dilation) {
	qi[0] = 2;
    }

    for (int b=0; b < B; b++) {
	for (int c_in=0; c_in < C_IN; c_in++) {
	    scalar_t g = 0;
	    for (int c_out=0; c_out < C_OUT; c_out++) {
		for (int p=-1; p <= 1; p++) {
		    for (int q=-1; q <= 1; q++) {
			int hp = reflect(h - dilation * p, (int) H);
			int wq = reflect(w - dilation * q, (int) W);
			if (0.0 < output[b][c_out][hp][wq]) {
			    g += kernel[c_out][c_in][pi[p + 1]][qi[q + 1]] // p and q can be negative
				* grad_output[b][c_out][hp][wq];
			}
		    }
		}
	    }
	    grad_input[b][c_in][h][w] += g;
	}
    }
}

template <typename scalar_t>
__global__ void
conv_relu_backward_x1(dTensor4R output,
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

    int B = grad_output.getSize(0);
    int C_OUT = grad_output.getSize(1);
    int C_IN = grad_input.getSize(1);
    int H = grad_output.getSize(2);
    int W = grad_output.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    for (int i=pId; i < kernel.numElements(); i+=num_threads) {
	kernel_buf[i] = kernel.data()[i];
    }
    __syncthreads();

    // We can index kernel_buffer like a 4d tensor.
    dTensor4R kernel_buffer = THCDeviceTensor<scalar_t, 4, THC_INDEX, RestrictPtrTraits>
	(kernel_buf, kernel.sizes(), kernel.strides());

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

    THC_INDEX output_offsets[9];
    THC_INDEX grad_output_offsets[9];
    THC_INDEX kernel_offsets[9];
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


///////////////////////////////////////////////////////////////////////////////
//                            Convolution:Forward                            //
///////////////////////////////////////////////////////////////////////////////


template <typename scalar_t>
__global__ void
conv_relu2(dTensor4R input,
	  dTensor4R kernel,
	  dTensor1R bias,
	  dTensor4R output,
	  int dilation)
{
    // This is an unoptimized reference implementation. It could serve
    // as a starting point for further optimization.

    int B = output.getSize(0);
    int C_OUT = output.getSize(1);
    int C_IN = input.getSize(1);
    int H = input.getSize(2);
    int W = input.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;

    if (W <= w || H <= h) {
	return;
    }

    for (int b=0; b < B; b++) {
	for (int c_out=0; c_out < C_OUT; c_out++) {
	    scalar_t o = bias[c_out];
	    for (int c_in=0; c_in < C_IN; c_in++) {
		for (int p=-1; p <= 1; p++) {
		    for (int q=-1; q <= 1; q++) {
			int hp = reflect(h + dilation * p, (int) H);
			int wq = reflect(w + dilation * q, (int) W);
			o += kernel[c_out][c_in][p + 1][q + 1] // p and q can be negative
			    * input[b][c_in][hp][wq];
		    }
		}
	    }
	    output[b][c_out][h][w] = max(0.0, o);
	}
    }
}

template <typename scalar_t>
__global__ void
conv_relu3(dTensor4R input,
	   dTensor4R kernel,
	   dTensor1R bias,
	   dTensor4R output,
	   int dilation)
{
    // This implementation caches the kernel weights.

    // LIMITS:
    //    49152 bytes of shared memory per block
    //    12288 floats of shared memory per block
    // +-  1300 kernels can be stored in shared mem
    // So we must have:
    //     C_IN * C_OUT < 1300
    extern __shared__ int shared_memory[];

    int B = output.getSize(0);
    int C_OUT = output.getSize(1);
    int C_IN = input.getSize(1);
    int H = input.getSize(2);
    int W = input.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    for (int i=pId; i < kernel.numElements(); i+=num_threads) {
	kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 4d tensor.
    dTensor4R kernel_buffer = THCDeviceTensor<scalar_t, 4, THC_INDEX, RestrictPtrTraits>
	(kernel_buf, kernel.sizes(), kernel.strides());

    __syncthreads();

    if (W <= w || H <= h) {
	return;
    }

    for (int b=0; b < B; b++) {
	for (int c_out=0; c_out < C_OUT; c_out++) {
	    scalar_t o = bias[c_out];
	    for (int c_in=0; c_in < C_IN; c_in++) {
		for (int p=-1; p <= 1; p++) {
		    for (int q=-1; q <= 1; q++) {
			int hp = reflect(h + dilation * p, (int) H);
			int wq = reflect(w + dilation * q, (int) W);
			o += kernel_buffer[c_out][c_in][p + 1][q + 1]
			    * input[b][c_in][hp][wq];
		    }
		}
	    }
	    output[b][c_out][h][w] = max(0.0, o);
	}
    }
}

template <typename scalar_t>
__global__ void
conv_relu4(dTensor4R input,
	    dTensor4R kernel,
	    dTensor1R bias,
	    dTensor4R output,
	    int dilation)
{
    // Performance improvements:
    // 1) This implementation caches the kernel weights.
    // 2) This implementation precomputes data and kernel offsets.

    // LIMITS:
    //    49152 bytes of shared memory per block
    //    12288 floats of shared memory per block
    // +-  1300 kernels can be stored in shared mem
    // So we must have:
    //     C_IN * C_OUT < 1300
    extern __shared__ int shared_memory[];

    int B = output.getSize(0);
    int C_OUT = output.getSize(1);
    int C_IN = input.getSize(1);
    int H = input.getSize(2);
    int W = input.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    for (int i=pId; i < kernel.numElements(); i+=num_threads) {
	kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 4d tensor.
    dTensor4R kernel_buffer = THCDeviceTensor<scalar_t, 4, THC_INDEX, RestrictPtrTraits>
	(kernel_buf, kernel.sizes(), kernel.strides());

    __syncthreads();

    if (W <= w || H <= h) {
	return;
    }

    // Precompute data offsets:
    THC_INDEX data_offsets[9];
    scalar_t *data0 = &input[0][0][0][0];
    int i = 0;
    for (int p=-1; p <= 1; p++) {
	for (int q=-1; q <= 1; q++) {
	    int hp = reflect(h + dilation * p, (int) H);
	    int wq = reflect(w + dilation * q, (int) W);
	    data_offsets[i] = &input[0][0][hp][wq] - data0;
	    i++;
	}
    }
    // Actually compute the convolution
    for (int b=0; b < B; b++) {
	for (int c_out=0; c_out < C_OUT; c_out++) {
	    scalar_t o = bias[c_out];
	    for (int c_in=0; c_in < C_IN; c_in++) {
		data0 = &input[b][c_in][0][0];
		scalar_t *kernel0 = &kernel_buffer[c_out][c_in][0][0];
		for (int i= 0; i < 9; i++) {
		    o += *(data0 + data_offsets[i]) * (*kernel0);
		    // Incrementing the kernel pointer works because
		    // the kernel weights are contiguous and the
		    // data_offsets are prepared to be in the same
		    // order as the kernel weights.
		    kernel0++;
		}
	    }
	    output[b][c_out][h][w] = max(0.0, o);
	}
    }
}

template <typename scalar_t>
__global__ void
conv_relu5(dTensor4R input,
	   dTensor4R kernel,
	   dTensor1R bias,
	   dTensor4R output,
	   int dilation)
{
    // Performance improvements:
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

    int B = output.getSize(0);
    int C_OUT = output.getSize(1);
    int C_IN = input.getSize(1);
    int H = input.getSize(2);
    int W = input.getSize(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    for (int i=pId; i < kernel.numElements(); i+=num_threads) {
	kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 4d tensor.
    dTensor4R kernel_buffer = THCDeviceTensor<scalar_t, 4, THC_INDEX, RestrictPtrTraits>
	(kernel_buf, kernel.sizes(), kernel.strides());

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
	    output[b][c_out][h][w] = max(0.0, o);
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
//                        Kernel preparation functions                       //
///////////////////////////////////////////////////////////////////////////////

at::Tensor conv_relu_cuda_forward(at::Tensor input_t,
			     at::Tensor kernel_t,
			     at::Tensor bias_t,
			     at::Tensor out_t,
			     int dilation,
			     int implementation,
			     int block_size) {
    OptionalDeviceGuard device_guard(device_of(input_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input_t.scalar_type(), "conv_relu_cuda_forward", ([&] {
        // Create device tensors:
        dTensor4R input_d = toDeviceTensorR<scalar_t,4>(input_t);
	dTensor4R kernel_d = toDeviceTensorR<scalar_t,4>(kernel_t);
        dTensor4R out_d = toDeviceTensorR<scalar_t,4>(out_t);
	dTensor1R bias_d = toDeviceTensorR<scalar_t, 1>(bias_t);

        dim3 gridSize(THCCeilDiv(input_d.getSize(3), block_size),
    		      THCCeilDiv(input_d.getSize(2), block_size));
        dim3 blockSize(block_size, block_size);
    	auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
	if (implementation == 2) {
	    conv_relu2<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
		(input_d, kernel_d, bias_d, out_d, dilation);
	} else if (implementation == 3) {
	    conv_relu3<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
		(input_d, kernel_d, bias_d, out_d, dilation);
	} else if (implementation == 4) {
	    conv_relu4<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
		(input_d, kernel_d, bias_d, out_d, dilation);
	} else if (implementation == 5) {
	    conv_relu5<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
		(input_d, kernel_d, bias_d, out_d, dilation);
	}

    	THCudaCheck(cudaGetLastError());
    }));
    return out_t;
}

void conv_relu_cuda_backward_x(at::Tensor output_t,
			       at::Tensor grad_output_t,
			       at::Tensor kernel_t,
			       at::Tensor grad_input_t,
			       int dilation,
			       int implementation,
			       int block_size) {
    OptionalDeviceGuard device_guard(device_of(grad_output_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(output_t.scalar_type(), "conv_relu_cuda_backward_x", ([&] {
        // Create device tensors:
        dTensor4R output_d = toDeviceTensorR<scalar_t,4>(output_t);
        dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output_t);
        dTensor4R grad_input_d = toDeviceTensorR<scalar_t,4>(grad_input_t);
	dTensor4R kernel_d = toDeviceTensorR<scalar_t,4>(kernel_t);
        dim3 gridSize(THCCeilDiv((int) grad_output_d.getSize(3), block_size),
    		      THCCeilDiv((int) grad_output_d.getSize(2), block_size));
        dim3 blockSize(block_size, block_size);
    	auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
	if (implementation == 0) {
	    conv_relu_backward_x0<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
		(output_d, grad_output_d, kernel_d, grad_input_d, dilation);
	} else if (implementation == 1) {
	    conv_relu_backward_x1<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
		(output_d, grad_output_d, kernel_d, grad_input_d, dilation);
	}
    	THCudaCheck(cudaGetLastError());
    }));
}

void conv_relu_cuda_backward_k(at::Tensor output, at::Tensor grad_output, at::Tensor input,
			       at::Tensor grad_kernel,
			       int dilation, int implementation, int block_size)
{
    OptionalDeviceGuard device_guard(device_of(grad_output));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv_relu_cuda_backward_k", ([&] {
        // Create device tensors:
        dTensor4R output_d = toDeviceTensorR<scalar_t,4>(output);
        dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output);
        dTensor4R input_d = toDeviceTensorR<scalar_t,4>(input);
	dTensor4R grad_kernel_d = toDeviceTensorR<scalar_t,4>(grad_kernel);
        dim3 gridSize(THCCeilDiv((int) grad_output_d.getSize(3), block_size),
    		      THCCeilDiv((int) grad_output_d.getSize(2), block_size));
        dim3 blockSize(block_size, block_size);
	if (implementation == 0) {
	    conv_relu_backward_k0<scalar_t><<<gridSize, blockSize, 0, stream>>>
		(output_d, grad_output_d, input_d, grad_kernel_d, dilation);
	} else if (implementation == 1) {
	    conv_relu_backward_k1<scalar_t><<<gridSize, blockSize, 0, stream>>>
		(output_d, grad_output_d, input_d, grad_kernel_d, dilation);
	}


    	THCudaCheck(cudaGetLastError());
    }));
}

void conv_relu_cuda_backward_bias(at::Tensor output,
				  at::Tensor grad_output,
				  at::Tensor grad_bias,
				  int implementation, int block_size)
{
    OptionalDeviceGuard device_guard(device_of(grad_output));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv_relu_cuda_backward_bias", ([&] {
        // Create device tensors:
        dTensor4R output_d = toDeviceTensorR<scalar_t,4>(output);
        dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output);
	dTensor1R grad_bias_d = toDeviceTensorR<scalar_t,1>(grad_bias);
        dim3 gridSize(THCCeilDiv((int) grad_output_d.getSize(3), block_size),
    		      THCCeilDiv((int) grad_output_d.getSize(2), block_size));
        dim3 blockSize(block_size, block_size);
	if (implementation == 0) {
	    conv_relu_backward_bias0<scalar_t><<<gridSize, blockSize, 0, stream>>>
		(output_d, grad_output_d, grad_bias_d);
	} else if (implementation == 1) {
	    conv_relu_backward_bias1<scalar_t><<<gridSize, blockSize, 0, stream>>>
		(output_d, grad_output_d, grad_bias_d);
	}
    	THCudaCheck(cudaGetLastError());
    }));
}
