// reflectionpad_inplace.cpp
#include <torch/torch.h>
#include <vector>

#include "ATen/TensorUtils.h"

// CUDA forward declarations
at::Tensor reflectionpad_inplace_cuda_forward(at::Tensor input, int padL,
                                              int padR, int padT, int padB);
at::Tensor reflectionpad_inplace_cuda_backward(at::Tensor grad_output, int padL,
                                               int padR, int padT, int padB);

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

at::Tensor reflectionpad_inplace_forward(at::Tensor input, int padding) {
  CHECK_INPUT(input);
  return reflectionpad_inplace_cuda_forward(input, padding, padding, padding,
                                            padding);
}

at::Tensor reflectionpad_inplace_backward(at::Tensor grad_output, int padding) {
  CHECK_INPUT(grad_output);
  return reflectionpad_inplace_cuda_backward(grad_output, padding, padding,
                                             padding, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &reflectionpad_inplace_forward,
        "ReflectionPad inplace forward (CUDA)");
  m.def("backward", &reflectionpad_inplace_backward,
        "ReflectionPad inplace backward (CUDA)");
}
