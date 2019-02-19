#include <torch/extension.h>
#include <vector>
#include "ATen/TensorUtils.h"
#include "ATen/ScalarType.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

///////////////////////////////////////////////////////////////////////////////
//                   Forward declaration of CUDA functions                   //
///////////////////////////////////////////////////////////////////////////////
at::Tensor conv_relu_cuda_forward(at::Tensor input_t,
				  at::Tensor kernel_t,
				  at::Tensor bias_t,
				  at::Tensor out_t,
				  int dilation,
				  int implementation,
				  int block_size);

void conv_relu_cuda_backward_x(at::Tensor output_t,
			       at::Tensor grad_output_t,
			       at::Tensor kernel_t,
			       at::Tensor grad_input_t,
			       int dilation,
			       int implementation,
			       int block_size);

void conv_relu_cuda_backward_k(at::Tensor output,
			       at::Tensor grad_output,
			       at::Tensor input,
			       at::Tensor grad_kernel,
			       int dilation, int implementation, int block_size);

void conv_relu_cuda_backward_bias(at::Tensor output,
				  at::Tensor grad_output,
				  at::Tensor grad_bias,
				  int implementation, int block_size);

///////////////////////////////////////////////////////////////////////////////
//                                  Macros                                   //
///////////////////////////////////////////////////////////////////////////////
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)


///////////////////////////////////////////////////////////////////////////////
//                                 Functions                                 //
///////////////////////////////////////////////////////////////////////////////
at::Tensor conv_relu_forward(at::Tensor input,
			at::Tensor kernel,
			at::Tensor bias,
			at::Tensor output,
			int dilation,
			int implementation,
			int block_size) {
    CHECK_CUDA(input);
    CHECK_CUDA(output);
    CHECK_CUDA(bias);
    CHECK_INPUT(kernel); 	// kernel must be contiguous.

    // Check data type
    AT_ASSERTM(input.type() == kernel.type(), "input and kernel must have same type");
    AT_ASSERTM(input.type() == bias.type(), "input and bias must have same type");
    AT_ASSERTM(input.type() == output.type(), "input and output must have same type");

    // Check dimensions
    AT_ASSERTM(input.dim() == 4, "input must be four-dimensional");
    AT_ASSERTM(kernel.dim() == 4, "kernel must be four-dimensional");
    AT_ASSERTM(output.dim() == 4, "output must be four-dimensional");
    AT_ASSERTM(bias.dim() == 1, "bias must be one-dimensional");

    // Check that kernel is 3x3:
    AT_ASSERTM(kernel.size(2) == 3, "Kernel shape must be 3x3");
    AT_ASSERTM(kernel.size(3) == 3, "Kernel shape must be 3x3");

    // Check kernel batch == output channel
    AT_ASSERTM(kernel.size(0) == output.size(1), "Kernel shape does not match output channels");
    // Check bias length  == output channel
    AT_ASSERTM(bias.size(0) == output.size(1), "Bias shape does not match output channels");
    // Check kernel channel == input channel
    AT_ASSERTM(kernel.size(1) == input.size(1), "Kernel shape does not match input channels");
    // Check input batch == output batch
    AT_ASSERTM(input.size(0) == output.size(0), "Input and output batch size do not match");
    // Check input img size == output img size
    AT_ASSERTM(input.size(2) == output.size(2), "Input and output shape do not match");
    AT_ASSERTM(input.size(3) == output.size(3), "Input and output shape do not match");
    return conv_relu_cuda_forward(input, kernel, bias, output, dilation, implementation, block_size);
}

void conv_relu_backward_x(at::Tensor output,
			  at::Tensor grad_output,
			  at::Tensor kernel,
			  at::Tensor grad_input,
			  int dilation,
			  int implementation,
			  int block_size) {
    CHECK_INPUT(kernel); 	// kernel must be contiguous
    CHECK_CUDA(output);
    CHECK_CUDA(grad_output);
    CHECK_CUDA(grad_input);

    // Check data type
    AT_ASSERTM(grad_output.type() == output.type(), "grad_output and output must have same type");
    AT_ASSERTM(grad_output.type() == kernel.type(), "grad_output and kernel must have same type");
    AT_ASSERTM(grad_output.type() == grad_input.type(), "grad_output and grad_input must have same type");

    // check dims
    AT_ASSERTM(output.dim() == 4, "output must be four-dimensional");
    AT_ASSERTM(grad_output.dim() == 4, "grad_output must be four-dimensional");
    AT_ASSERTM(kernel.dim() == 4, "kernel must be four-dimensional");
    AT_ASSERTM(grad_input.dim() == 4, "grad_input must be four-dimensional");

    // Check shape of output and grad_output
    AT_ASSERTM(output.is_same_size(grad_output), "Shape of output and grad_output does not match");
    // Check kernel batch == output channel
    AT_ASSERTM(kernel.size(0) == grad_output.size(1), "Kernel shape does not match grad_output channels");
    // Check kernel channel == input channel
    AT_ASSERTM(kernel.size(1) == grad_input.size(1), "Kernel shape does not match grad_input channels");
    // Check input batch == output batch
    AT_ASSERTM(grad_input.size(0) == grad_output.size(0), "Grad_input and grad_output batch size do not match");
    // Check input img size == output img size
    AT_ASSERTM(grad_input.size(2) == grad_output.size(2), "Grad_input and grad_output shape do not match");
    AT_ASSERTM(grad_input.size(3) == grad_output.size(3), "Grad_input and grad_output shape do not match");

    conv_relu_cuda_backward_x(output, grad_output, kernel, grad_input, dilation, implementation, block_size);
}

void conv_relu_backward_k(at::Tensor output,
			  at::Tensor grad_output,
			  at::Tensor input,
			  at::Tensor grad_kernel,
			  int dilation,
			  int implementation,
			  int block_size) {

    CHECK_CUDA(output);
    CHECK_CUDA(grad_output);
    CHECK_CUDA(input);
    CHECK_CUDA(grad_kernel);

    // Check data type
    AT_ASSERTM(grad_output.type() == output.type(), "grad_output and output must have same type");
    AT_ASSERTM(grad_output.type() == input.type(), "grad_output and input must have same type");
    AT_ASSERTM(grad_output.type() == grad_kernel.type(), "grad_output and grad_kernel must have same type");

    // check dims
    AT_ASSERTM(output.dim() == 4, "output must be four-dimensional");
    AT_ASSERTM(grad_output.dim() == 4, "grad_output must be four-dimensional");
    AT_ASSERTM(input.dim() == 4, "grad_kernel must be four-dimensional");
    AT_ASSERTM(grad_kernel.dim() == 4, "grad_kernel must be four-dimensional");

    // Check shape of output and grad_output
    AT_ASSERTM(output.is_same_size(grad_output), "Shape of output and grad_output does not match");
    // Check kernel batch == output channel
    AT_ASSERTM(grad_kernel.size(0) == grad_output.size(1), "Grad_kernel shape does not match grad_output channels");
    // Check kernel channel == input channel
    AT_ASSERTM(grad_kernel.size(1) == input.size(1), "Grad_kernel shape does not match grad_input channels");
    // Check input batch == output batch
    AT_ASSERTM(input.size(0) == grad_output.size(0), "Grad_input and grad_output batch size do not match");
    // Check input img size == output img size
    AT_ASSERTM(input.size(2) == grad_output.size(2), "Grad_input and grad_output shape do not match");
    AT_ASSERTM(input.size(3) == grad_output.size(3), "Grad_input and grad_output shape do not match");

    conv_relu_cuda_backward_k(output, grad_output, input, grad_kernel, dilation, implementation, block_size);
}

void conv_relu_backward_bias(at::Tensor output,
			     at::Tensor grad_output,
			     at::Tensor grad_bias,
			     int implementation,
			     int block_size) {

    CHECK_CUDA(output);
    CHECK_CUDA(grad_output);

    // Check data type
    AT_ASSERTM(grad_output.type() == output.type(), "grad_output and output must have same type");
    AT_ASSERTM(grad_output.type() == grad_bias.type(), "grad_output and grad_bias must have same type");

    // check dims
    AT_ASSERTM(output.dim() == 4, "output must be four-dimensional");
    AT_ASSERTM(grad_output.dim() == 4, "grad_output must be four-dimensional");
    AT_ASSERTM(grad_bias.dim() == 1, "grad_bias must be one-dimensional");

    // Check shape of output and grad_output
    AT_ASSERTM(output.is_same_size(grad_output), "Shape of output and grad_output does not match");
    // Check bias length == output channel
    AT_ASSERTM(grad_bias.size(0) == grad_output.size(1), "Grad_bias shape does not match grad_output channels");

    conv_relu_cuda_backward_bias(output, grad_output, grad_bias, implementation, block_size);
}

///////////////////////////////////////////////////////////////////////////////
//                             Module declaration                            //
///////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_relu_forward", &conv_relu_forward, "Forward convolution"
	  "input"_a, "kernel"_a, "bias"_a, "output"_a, "dilation"_a, "implementation"_a=5,
	  "block_size"_a=16);
    m.def("conv_relu_backward_x", &conv_relu_backward_x, "Transpose of the forward convolution",
	  "output"_a, "grad_output"_a, "kernel"_a, "grad_input"_a, "dilation"_a, "implementation"_a=1,
	  "block_size"_a=16);
    m.def("conv_relu_backward_k", &conv_relu_backward_k, "Transpose of the forward convolution",
	  "output"_a, "grad_output"_a, "input"_a, "grad_kernel"_a, "dilation"_a, "implementation"_a=1,
	  "block_size"_a=16);
    m.def("conv_relu_backward_bias", &conv_relu_backward_bias, "Backward bias",
	  "output"_a, "grad_output"_a, "grad_bias"_a, "implementation"_a=1,
	  "block_size"_a=16);

}
