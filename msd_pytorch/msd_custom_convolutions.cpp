#include <tuple>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "msd_custom_convolutions/torch_cuda_dispatch.h"

namespace py = pybind11;
using namespace pybind11::literals;

using at::CheckedFrom;
///////////////////////////////////////////////////////////////////////////////
//                                  Macros                                   //
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//                                 Functions                                 //
///////////////////////////////////////////////////////////////////////////////

bool is_kernel_2d(torch::Tensor kernel){
    return kernel.dim() == 4;
}
bool is_kernel_3d(torch::Tensor kernel){
    return kernel.dim() == 5;
}

void checkKernel3x3(CheckedFrom c, torch::Tensor kernel) {
    if (is_kernel_2d(kernel)) {
        // Check that kernel is 3x3:
        TORCH_CHECK(kernel.size(2) == 3, "Kernel shape must be 3x3");
        TORCH_CHECK(kernel.size(3) == 3, "Kernel shape must be 3x3");
    } else if (is_kernel_3d(kernel)){
        TORCH_CHECK(kernel.size(2) == 3, "Kernel shape must be 3x3x3");
        TORCH_CHECK(kernel.size(3) == 3, "Kernel shape must be 3x3x3");
        TORCH_CHECK(kernel.size(4) == 3, "Kernel shape must be 3x3x3");
    } else {
        TORCH_CHECK(false, "Expected 4 or 5 dimensional kernel.")
    }
}

void checkDilation(CheckedFrom c, const torch::TensorGeometryArg& tensor, int dilation) {
    TORCH_CHECK(dilation > 0,
	     c, ": Dilation must be 1 or greater");
    if (tensor->dim() == 4) {
	TORCH_CHECK(dilation <= tensor->size(2) && dilation <= tensor->size(3),
		 c, ": Dilation cannot be greater than size of image. ");
    } else if (tensor->dim() == 5) {
	TORCH_CHECK(dilation <= tensor->size(2) && dilation <= tensor->size(3) && dilation <= tensor->size(4),
		 c, ": Dilation cannot be greater than size of image. ");
    } else {
	// unreachable arm.
	TORCH_CHECK(false, c, ": checkDilation got unsupported tensor dimension. Expected 4 or 5. Got: ", tensor->dim());
    }
}

void checkOutputChannels(CheckedFrom c, const torch::TensorGeometryArg& k, const torch::TensorGeometryArg& output) {
    TORCH_CHECK(k->size(0) == output->size(1),
	     c, ": Kernel shape does not match output channels. ");
}

void checkBiasShape(CheckedFrom c, const torch::TensorGeometryArg& bias, const torch::TensorGeometryArg& output) {
    TORCH_CHECK(bias->size(0) == output->size(1),
	     c, ": Bias shape does not match output channels. ");
}

void checkInputChannels(CheckedFrom c, const torch::TensorGeometryArg& input, const torch::TensorGeometryArg& k) {
    TORCH_CHECK(k->size(1) == input->size(1),
	     c, ": Kernel shape does not match input channels. ");
}

void checkInOutShape(CheckedFrom c, const torch::TensorGeometryArg& input, const torch::TensorGeometryArg& output) {
    torch::checkSameDim(c, input, output);
    torch::checkDimRange(c, input, 4, 6);
    torch::checkDimRange(c, output, 4, 6);
    TORCH_CHECK(input->size(0) == output->size(0),
	     c, ": Input batch dimension does not match output batch dimension. ");
    TORCH_CHECK(input->size(2) == output->size(2),
	     c, ": Input shape does not match output shape. ");
    TORCH_CHECK(input->size(3) == output->size(3),
	     c, ": Input shape does not match output shape. ");
    if (input->dim() == 5) {
	TORCH_CHECK(input->size(4) == output->size(4),
		 c, ": Input shape does not match output shape. ");
    }
}

void conv_forward(torch::Tensor input,
		  torch::Tensor kernel,
		  torch::Tensor bias,
		  torch::Tensor output,
		  int dilation,
		  std::tuple<int, int, int> block_size) {
    auto c = "conv_forward";

    torch::TensorArg arg_input(input, "input", 0);
    torch::TensorArg arg_kernel(kernel, "kernel", 1);
    torch::TensorArg arg_bias(bias, "bias", 2);
    torch::TensorArg arg_output(output, "output", 3);

    torch::checkAllSameGPU(c, {arg_input, arg_kernel, arg_bias, arg_output});
    torch::checkAllSameType(c, {arg_input, arg_kernel, arg_bias, arg_output});
    torch::checkContiguous(c, arg_kernel);
    checkInOutShape(c, arg_input, arg_output);

    checkKernel3x3(c, kernel);
    int data_dim = (is_kernel_2d(kernel)) ? 4: 5;

    torch::checkDim(c, arg_input, data_dim);
    torch::checkDim(c, arg_kernel, data_dim);
    torch::checkDim(c, arg_output, data_dim);
    torch::checkDim(c, arg_bias, 1);

    checkInputChannels(c, arg_input, arg_kernel);
    checkOutputChannels(c, arg_kernel, arg_output);
    checkBiasShape(c, arg_bias, arg_output);
    checkDilation(c, arg_input, dilation);

    conv_cuda_forward(input, kernel, bias, output, dilation, block_size);
}

void conv_backward_x(torch::Tensor grad_output,
                     torch::Tensor kernel,
                     torch::Tensor grad_input,
                     int dilation,
		     std::tuple<int, int, int> block_size) {
    auto c = "conv_backward_x";

    torch::TensorArg arg_output(grad_output, "grad_output", 0);
    torch::TensorArg arg_kernel(kernel, "kernel", 1);
    torch::TensorArg arg_input(grad_input, "grad_input", 2);

    // Check same device (only for forward pass)
    torch::checkAllSameGPU(c, {arg_input, arg_kernel, arg_output});
    torch::checkAllSameType(c, {arg_input, arg_kernel, arg_output});
    torch::checkContiguous(c, arg_kernel);
    checkInOutShape(c, arg_input, arg_output);

    checkKernel3x3(c, kernel);

    int data_dim = (is_kernel_2d(kernel)) ? 4: 5;
    torch::checkDim(c, arg_input, data_dim);
    torch::checkDim(c, arg_kernel, data_dim);
    torch::checkDim(c, arg_output, data_dim);

    checkInputChannels(c, arg_input, arg_kernel);
    checkOutputChannels(c, arg_kernel, arg_output);
    checkDilation(c, arg_input, dilation);

    conv_cuda_backward_x(grad_output, kernel, grad_input, dilation, block_size);
}

void conv_backward_k(torch::Tensor grad_output,
                     torch::Tensor input,
                     torch::Tensor grad_kernel,
                     int dilation,
                     std::tuple<int, int, int> block_size) {
    auto c = "conv_backward_k";

    torch::TensorArg arg_output(grad_output, "grad_output", 0);
    torch::TensorArg arg_input(input, "input", 1);
    torch::TensorArg arg_kernel(grad_kernel, "grad_kernel", 2);

    torch::checkAllSameGPU(c, {arg_input, arg_kernel, arg_output});
    torch::checkAllSameType(c, {arg_input, arg_kernel, arg_output});
    torch::checkContiguous(c, arg_kernel);
    checkInOutShape(c, arg_input, arg_output);

    checkKernel3x3(c, grad_kernel);

    int data_dim = (is_kernel_2d(grad_kernel)) ? 4: 5;
    torch::checkDim(c, arg_input, data_dim);
    torch::checkDim(c, arg_kernel, data_dim);
    torch::checkDim(c, arg_output, data_dim);

    checkInputChannels(c, arg_input, arg_kernel);
    checkOutputChannels(c, arg_kernel, arg_output);
    checkDilation(c, arg_input, dilation);

    conv_cuda_backward_k(grad_output, input, grad_kernel, dilation, block_size);
}

void conv_backward_bias(torch::Tensor grad_output,
                        torch::Tensor grad_bias,
                        std::tuple<int, int, int> block_size) {

    auto c = "conv_backward_bias";

    torch::TensorArg arg_bias(grad_bias, "grad_bias", 0);
    torch::TensorArg arg_output(grad_output, "grad_output", 1);

    torch::checkAllSameGPU(c, {arg_bias, arg_output});
    torch::checkAllSameType(c, {arg_bias, arg_output});

    checkBiasShape(c, arg_bias, arg_output);
    torch::checkDim(c, arg_bias, 1);
    torch::checkDimRange(c, arg_output, 4, 6);

    if (grad_output.dim() == 4) {
	// 2D
	const int64_t dims_[3]  = {0, 2, 3};
	auto dims = torch::IntList(dims_, 3);
	auto g = grad_output.sum(dims);
	grad_bias += g;
    } else if (grad_output.dim() == 5) {
	// 3D
	const int64_t dims_[4]  = {0, 2, 3, 4};
	auto dims = torch::IntList(dims_, 4);
	auto g = grad_output.sum(dims);
	grad_bias += g;
    } else {
	TORCH_CHECK(false, "Unreachable code: grad_output has wrong dimension");
    }
}

void conv_relu_forward(torch::Tensor input,
		       torch::Tensor kernel,
		       torch::Tensor bias,
		       torch::Tensor output,
		       int dilation,
		       std::tuple<int, int, int> block_size) {
    auto c = "conv_relu_forward";

    torch::TensorArg arg_input(input, "input", 0);
    torch::TensorArg arg_kernel(kernel, "kernel", 1);
    torch::TensorArg arg_bias(bias, "bias", 2);
    torch::TensorArg arg_output(output, "output", 3);

    torch::checkAllSameGPU(c, {arg_input, arg_kernel, arg_bias, arg_output});
    torch::checkAllSameType(c, {arg_input, arg_kernel, arg_bias, arg_output});
    torch::checkContiguous(c, arg_kernel);
    checkInOutShape(c, arg_input, arg_output);

    checkKernel3x3(c, kernel);
    int data_dim = (is_kernel_2d(kernel)) ? 4: 5;

    torch::checkDim(c, arg_input, data_dim);
    torch::checkDim(c, arg_kernel, data_dim);
    torch::checkDim(c, arg_output, data_dim);
    torch::checkDim(c, arg_bias, 1);

    checkInputChannels(c, arg_input, arg_kernel);
    checkOutputChannels(c, arg_kernel, arg_output);
    checkBiasShape(c, arg_bias, arg_output);
    checkDilation(c, arg_input, dilation);

    conv_relu_cuda_forward(input, kernel, bias, output, dilation, block_size);
}

void conv_relu_backward_x(torch::Tensor output,
                          torch::Tensor grad_output,
                          torch::Tensor kernel,
                          torch::Tensor grad_input,
                          int dilation,
                          std::tuple<int, int, int> block_size) {
    auto c = "conv_relu_backward_x";

    torch::TensorArg arg_output(output, "output", 0);
    torch::TensorArg arg_grad_output(grad_output, "grad_output", 1);
    torch::TensorArg arg_kernel(kernel, "kernel", 2);
    torch::TensorArg arg_input(grad_input, "grad_input", 3);

    // Check same device (only for forward pass)
    torch::checkAllSameGPU(c, {arg_input, arg_kernel, arg_output, arg_grad_output});
    torch::checkAllSameType(c, {arg_input, arg_kernel, arg_output, arg_grad_output});
    torch::checkContiguous(c, arg_kernel);
    checkInOutShape(c, arg_input, arg_output);
    checkInOutShape(c, arg_input, arg_grad_output);

    checkKernel3x3(c, kernel);

    int data_dim = (is_kernel_2d(kernel)) ? 4: 5;
    torch::checkDim(c, arg_input, data_dim);
    torch::checkDim(c, arg_kernel, data_dim);
    torch::checkDim(c, arg_output, data_dim);
    torch::checkDim(c, arg_grad_output, data_dim);
    checkDilation(c, arg_input, dilation);

    checkInputChannels(c, arg_input, arg_kernel);
    checkOutputChannels(c, arg_kernel, arg_output);
    checkOutputChannels(c, arg_kernel, arg_grad_output);

    conv_relu_cuda_backward_x(output, grad_output, kernel, grad_input, dilation, block_size);
}

void conv_relu_backward_k(torch::Tensor output,
                          torch::Tensor grad_output,
                          torch::Tensor input,
                          torch::Tensor grad_kernel,
                          int dilation,
                          std::tuple<int, int, int> block_size) {
    auto c = "conv_relu_backward_k";

    torch::TensorArg arg_output(output, "output", 0);
    torch::TensorArg arg_grad_output(grad_output, "grad_output", 1);
    torch::TensorArg arg_input(input, "input", 2);
    torch::TensorArg arg_kernel(grad_kernel, "grad_kernel", 3);

    torch::checkAllSameGPU(c, {arg_input, arg_kernel, arg_output, arg_grad_output});
    torch::checkAllSameType(c, {arg_input, arg_kernel, arg_output, arg_grad_output});
    torch::checkContiguous(c, arg_kernel);
    checkInOutShape(c, arg_input, arg_output);
    checkInOutShape(c, arg_input, arg_grad_output);

    checkKernel3x3(c, grad_kernel);

    int data_dim = (is_kernel_2d(grad_kernel)) ? 4: 5;
    torch::checkDim(c, arg_input, data_dim);
    torch::checkDim(c, arg_kernel, data_dim);
    torch::checkDim(c, arg_output, data_dim);
    torch::checkDim(c, arg_grad_output, data_dim);

    checkInputChannels(c, arg_input, arg_kernel);
    checkOutputChannels(c, arg_kernel, arg_output);
    checkOutputChannels(c, arg_kernel, arg_grad_output);
    checkDilation(c, arg_input, dilation);

    conv_relu_cuda_backward_k(output, grad_output, input, grad_kernel, dilation, block_size);
}

void conv_relu_backward_bias(torch::Tensor output,
                             torch::Tensor grad_output,
                             torch::Tensor grad_bias,
                             std::tuple<int, int, int> block_size) {
    auto c = "conv_relu_backward_bias";

    torch::TensorArg arg_output(output, "output", 0);
    torch::TensorArg arg_grad_output(grad_output, "grad_output", 1);
    torch::TensorArg arg_bias(grad_bias, "grad_bias", 2);

    torch::checkAllSameGPU(c, {arg_bias, arg_output, arg_grad_output});
    torch::checkAllSameType(c, {arg_bias, arg_output, arg_grad_output});
    checkInOutShape(c, arg_output, arg_grad_output);

    checkBiasShape(c, arg_bias, arg_output);
    checkBiasShape(c, arg_bias, arg_grad_output);

    torch::checkDim(c, arg_bias, 1);

    conv_relu_cuda_backward_bias(output, grad_output, grad_bias, block_size);
}

///////////////////////////////////////////////////////////////////////////////
//                             Module declaration                            //
///////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward, "Forward convolution"
          "input"_a, "kernel"_a, "bias"_a, "output"_a, "dilation"_a,
          "block_size"_a);
    m.def("conv_backward_x", &conv_backward_x, "Transpose of the forward convolution",
          "grad_output"_a, "kernel"_a, "grad_input"_a, "dilation"_a,
          "block_size"_a);
    m.def("conv_backward_k", &conv_backward_k, "Transpose of the forward convolution",
          "grad_output"_a, "input"_a, "grad_kernel"_a, "dilation"_a,
          "block_size"_a);
    m.def("conv_backward_bias", &conv_backward_bias, "Backward bias",
          "grad_output"_a, "grad_bias"_a,
          "block_size"_a);

    m.def("conv_relu_forward", &conv_relu_forward, "Forward convolution"
          "input"_a, "kernel"_a, "bias"_a, "output"_a, "dilation"_a,
          "block_size"_a);
    m.def("conv_relu_backward_x", &conv_relu_backward_x, "Transpose of the forward convolution",
          "output"_a, "grad_output"_a, "kernel"_a, "grad_input"_a, "dilation"_a,
          "block_size"_a);
    m.def("conv_relu_backward_k", &conv_relu_backward_k, "Transpose of the forward convolution",
          "output"_a, "grad_output"_a, "input"_a, "grad_kernel"_a, "dilation"_a,
          "block_size"_a);
    m.def("conv_relu_backward_bias", &conv_relu_backward_bias, "Backward bias",
          "output"_a, "grad_output"_a, "grad_bias"_a,
          "block_size"_a);
}
