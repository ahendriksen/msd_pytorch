// -*- eval:(c++-mode); c-file-style: "bsd"; -*-
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void relu_inplace_cuda_forward_kernel(scalar_t* __restrict__ input, size_t size) {
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = start; i < size; i += blockDim.x * gridDim.x) {
	input[i] = max(0.0, input[i]);
    }
}

template <typename scalar_t>
__global__ void relu_inplace_cuda_backward_kernel(const scalar_t* __restrict__ input,
						  const scalar_t* __restrict__ grad_output,
						  scalar_t* __restrict__ grad_input,
						  size_t size) {
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = start; i < size; i += blockDim.x * gridDim.x) {
	grad_input[i] = input[i] > 0 ? grad_output[i] : 0;
    }
}

at::Tensor relu_inplace_cuda_forward(at::Tensor input){

    const int threads = 1024;
    auto size = input.numel();
    const dim3 blocks((size + threads - 1) / threads, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_inplace_cuda_forward", ([&] {
		relu_inplace_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>(), size);
	    }));
    return input;
}

at::Tensor relu_inplace_cuda_backward(at::Tensor input, at::Tensor grad_output){

    auto grad_input = at::zeros_like(grad_output);

    const int threads = 1024;
    auto size = input.numel();
    const dim3 blocks((size + threads - 1) / threads, 1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_inplace_cuda_forward", ([&] {
		relu_inplace_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>(),
										 grad_output.data<scalar_t>(),
										 grad_input.data<scalar_t>(),
										 size);
	    }));
    return grad_input;
}
