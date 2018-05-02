// relu_inplace.cpp
#include <torch/torch.h>
#include <vector>

#include "ATen/TensorUtils.h"

// CUDA forward declarations
at::Tensor relu_inplace_cuda_forward(at::Tensor input);
at::Tensor relu_inplace_cuda_backward(at::Tensor input, at::Tensor grad_output);

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor relu_inplace_forward(at::Tensor input){
    CHECK_INPUT(input);
    return relu_inplace_cuda_forward(input);
}

at::Tensor relu_inplace_backward(at::Tensor input, at::Tensor grad_output){
    CHECK_INPUT(input);
    CHECK_INPUT(grad_output);

    if (! input.sizes().equals(grad_output.sizes())){
	at::CheckedFrom c = "relu_inplace_backward";
	std::ostringstream oss;
    	oss << "Expected grad_output to have size " << input.sizes()
	    << ". Got " << grad_output.sizes() << " instead."
    	    << " (while checking arguments for " << c << ")";
    	throw std::runtime_error(oss.str());
    }

    return relu_inplace_cuda_backward(input, grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &relu_inplace_forward, "ReLU inplace forward (CUDA)");
    m.def("backward", &relu_inplace_backward, "ReLU inplace backward (CUDA)");
}
