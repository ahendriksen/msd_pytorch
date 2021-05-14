#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "msd_custom_convolutions/torch_cuda_dispatch.h"

namespace py = pybind11;
using namespace pybind11::literals;

///////////////////////////////////////////////////////////////////////////////
//                                  Macros                                   //
///////////////////////////////////////////////////////////////////////////////
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

///////////////////////////////////////////////////////////////////////////////
//                                 Functions                                 //
///////////////////////////////////////////////////////////////////////////////
torch::Tensor conv_forward(torch::Tensor input,
                        torch::Tensor kernel,
                        torch::Tensor bias,
                        torch::Tensor output,
                        int dilation,
                        int block_size) {
    CHECK_CUDA(input);
    CHECK_CUDA(output);
    CHECK_CUDA(bias);
    CHECK_INPUT(kernel);        // kernel must be contiguous.

    torch::TensorArg arg_input(input, "input", 0);
    torch::TensorArg arg_kernel(kernel, "kernel", 1);
    torch::TensorArg arg_bias(bias, "bias", 2);
    torch::TensorArg arg_output(output, "output", 3);

    // Check same device (only for forward pass)
    torch::checkAllSameGPU("conv_forward", {arg_input, arg_kernel, arg_bias, arg_output});

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

    return conv_cuda_forward(input, kernel, bias, output, dilation, block_size);
}

void conv_backward_x(torch::Tensor grad_output,
                     torch::Tensor kernel,
                     torch::Tensor grad_input,
                     int dilation,
                     int block_size) {
    CHECK_INPUT(kernel);        // kernel must be contiguous
    CHECK_CUDA(grad_output);
    CHECK_CUDA(grad_input);

    // Check data type
    AT_ASSERTM(grad_output.type() == kernel.type(), "grad_output and kernel must have same type");
    AT_ASSERTM(grad_output.type() == grad_input.type(), "grad_output and grad_input must have same type");

    // check dims
    AT_ASSERTM(grad_output.dim() == 4, "grad_output must be four-dimensional");
    AT_ASSERTM(kernel.dim() == 4, "kernel must be four-dimensional");
    AT_ASSERTM(grad_input.dim() == 4, "grad_input must be four-dimensional");

    // Check kernel batch == output channel
    AT_ASSERTM(kernel.size(0) == grad_output.size(1), "Kernel shape does not match grad_output channels");
    // Check kernel channel == input channel
    AT_ASSERTM(kernel.size(1) == grad_input.size(1), "Kernel shape does not match grad_input channels");
    // Check input batch == output batch
    AT_ASSERTM(grad_input.size(0) == grad_output.size(0), "Grad_input and grad_output batch size do not match");
    // Check input img size == output img size
    AT_ASSERTM(grad_input.size(2) == grad_output.size(2), "Grad_input and grad_output shape do not match");
    AT_ASSERTM(grad_input.size(3) == grad_output.size(3), "Grad_input and grad_output shape do not match");

    conv_cuda_backward_x(grad_output, kernel, grad_input, dilation, block_size);
}

void conv_backward_k(torch::Tensor grad_output,
                     torch::Tensor input,
                     torch::Tensor grad_kernel,
                     int dilation,
                     int block_size) {

    CHECK_CUDA(grad_output);
    CHECK_CUDA(input);
    CHECK_CUDA(grad_kernel);

    // Check data type
    AT_ASSERTM(grad_output.type() == input.type(), "grad_output and input must have same type");
    AT_ASSERTM(grad_output.type() == grad_kernel.type(), "grad_output and grad_kernel must have same type");

    // check dims
    AT_ASSERTM(grad_output.dim() == 4, "grad_output must be four-dimensional");
    AT_ASSERTM(input.dim() == 4, "grad_kernel must be four-dimensional");
    AT_ASSERTM(grad_kernel.dim() == 4, "grad_kernel must be four-dimensional");

    // Check kernel batch == output channel
    AT_ASSERTM(grad_kernel.size(0) == grad_output.size(1), "Grad_kernel shape does not match grad_output channels");
    // Check kernel channel == input channel
    AT_ASSERTM(grad_kernel.size(1) == input.size(1), "Grad_kernel shape does not match grad_input channels");
    // Check input batch == output batch
    AT_ASSERTM(input.size(0) == grad_output.size(0), "Grad_input and grad_output batch size do not match");
    // Check input img size == output img size
    AT_ASSERTM(input.size(2) == grad_output.size(2), "Grad_input and grad_output shape do not match");
    AT_ASSERTM(input.size(3) == grad_output.size(3), "Grad_input and grad_output shape do not match");

    conv_cuda_backward_k(grad_output, input, grad_kernel, dilation, block_size);
}

void conv_backward_bias(torch::Tensor grad_output,
                        torch::Tensor grad_bias,
                        int block_size) {

    CHECK_CUDA(grad_output);

    // Check data type
    AT_ASSERTM(grad_output.type() == grad_bias.type(), "grad_output and grad_bias must have same type");

    // check dims
    AT_ASSERTM(grad_output.dim() == 4, "grad_output must be four-dimensional");
    AT_ASSERTM(grad_bias.dim() == 1, "grad_bias must be one-dimensional");

    // Check bias length == output channel
    AT_ASSERTM(grad_bias.size(0) == grad_output.size(1), "Grad_bias shape does not match grad_output channels");

    const int64_t dims_[3]  = {0, 2, 3};
    auto dims = torch::IntList(dims_, 3);
    auto g = grad_output.sum(dims);
    grad_bias += g;
}

torch::Tensor conv_relu_forward(torch::Tensor input,
                        torch::Tensor kernel,
                        torch::Tensor bias,
                        torch::Tensor output,
                        int dilation,
                        int block_size) {
    CHECK_CUDA(input);
    CHECK_CUDA(output);
    CHECK_CUDA(bias);
    CHECK_INPUT(kernel);        // kernel must be contiguous.

    torch::TensorArg arg_input(input, "input", 0);
    torch::TensorArg arg_kernel(kernel, "kernel", 1);
    torch::TensorArg arg_bias(bias, "bias", 2);
    torch::TensorArg arg_output(output, "output", 3);

    // Check same device (only for forward pass)
    torch::checkAllSameGPU("conv_relu_forward", {arg_input, arg_kernel, arg_bias, arg_output});

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

    return conv_relu_cuda_forward(input, kernel, bias, output, dilation, block_size);
}

void conv_relu_backward_x(torch::Tensor output,
                          torch::Tensor grad_output,
                          torch::Tensor kernel,
                          torch::Tensor grad_input,
                          int dilation,
                          int block_size) {
    CHECK_INPUT(kernel);        // kernel must be contiguous
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

    conv_relu_cuda_backward_x(output, grad_output, kernel, grad_input, dilation, block_size);
}

void conv_relu_backward_k(torch::Tensor output,
                          torch::Tensor grad_output,
                          torch::Tensor input,
                          torch::Tensor grad_kernel,
                          int dilation,
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

    conv_relu_cuda_backward_k(output, grad_output, input, grad_kernel, dilation, block_size);
}

void conv_relu_backward_bias(torch::Tensor output,
                             torch::Tensor grad_output,
                             torch::Tensor grad_bias,
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

    conv_relu_cuda_backward_bias(output, grad_output, grad_bias, block_size);
}

///////////////////////////////////////////////////////////////////////////////
//                             Module declaration                            //
///////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward, "Forward convolution"
          "input"_a, "kernel"_a, "bias"_a, "output"_a, "dilation"_a,
          "block_size"_a=16);
    m.def("conv_backward_x", &conv_backward_x, "Transpose of the forward convolution",
          "grad_output"_a, "kernel"_a, "grad_input"_a, "dilation"_a,
          "block_size"_a=16);
    m.def("conv_backward_k", &conv_backward_k, "Transpose of the forward convolution",
          "grad_output"_a, "input"_a, "grad_kernel"_a, "dilation"_a,
          "block_size"_a=16);
    m.def("conv_backward_bias", &conv_backward_bias, "Backward bias",
          "grad_output"_a, "grad_bias"_a,
          "block_size"_a=16);

    m.def("conv_relu_forward", &conv_relu_forward, "Forward convolution"
          "input"_a, "kernel"_a, "bias"_a, "output"_a, "dilation"_a,
          "block_size"_a=16);
    m.def("conv_relu_backward_x", &conv_relu_backward_x, "Transpose of the forward convolution",
          "output"_a, "grad_output"_a, "kernel"_a, "grad_input"_a, "dilation"_a,
          "block_size"_a=16);
    m.def("conv_relu_backward_k", &conv_relu_backward_k, "Transpose of the forward convolution",
          "output"_a, "grad_output"_a, "input"_a, "grad_kernel"_a, "dilation"_a,
          "block_size"_a=16);
    m.def("conv_relu_backward_bias", &conv_relu_backward_bias, "Backward bias",
          "output"_a, "grad_output"_a, "grad_bias"_a,
          "block_size"_a=16);

}
