import torch.nn as nn
import torch.utils.cpp_extension as cppe
import torch as t
from torch.autograd import (Variable, Function)
from torch.nn import Parameter
from torch.backends import cudnn


_C = cppe.load('conv_inplace',
               sources=['msd_pytorch/conv_inplace.cpp'],
               extra_cflags=['-Wall', '-Werror', '-Wfatal-errors', '-Wextra'],
               extra_include_paths=cppe.include_paths(cuda=True),
               verbose = True)

cudnn_forward = _C.cudnn_convolution_full_forward
cudnn_backward_data_ = _C.cudnn_convolution_backward_data_
cudnn_backward_weight_ = _C.cudnn_convolution_backward_weight_
cudnn_backward_bias = _C.cudnn_convolution_backward_bias

class ConvNdInPlaceFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, output, padding, stride, dilation):
        # save_for_backward can only save input or output
        # tensors. Since we require the output to be contiguous, we
        # save the contiguous version of output. We cannot save the
        # contiguous version of input, so we save the (possibly)
        # non-contiguous version.

        assert (output.is_contiguous() and input.is_contiguous()), \
            "non-contiguous input or output not supported (ConvNdInPlaceFunction)"

        ctx.save_for_backward(input, weight, bias)
        ctx.padding, ctx.stride, ctx.dilation = padding, stride, dilation

        cudnn_forward(input, weight, bias, output.data, padding, stride, dilation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # restore variables
        input, weight, bias = ctx.saved_tensors
        padding, stride, dilation = ctx.padding, ctx.stride, ctx.dilation

        # ensure that grad_output is contiguous
        grad_output = grad_output.contiguous()

        # Input
        grad_input = input.clone()
        cudnn_backward_data_(grad_output, grad_input, weight, padding, stride, dilation)
        # Weight
        grad_weight = weight.clone()
        cudnn_backward_weight_(grad_output, input, grad_weight, padding, stride, dilation)
        # Bias
        grad_bias = cudnn_backward_bias(grad_output)

        return grad_input, grad_weight, grad_bias, None, None, None, None


convNdInPlace = ConvNdInPlaceFunction.apply


class Conv2dInPlaceModule(nn.Module):

    def __init__(self, output, in_channels, out_channels,
                 kernel_size=3, dilation=1, padding=1):
        super(Conv2dInPlaceModule, self).__init__()

        w = t.Tensor(out_channels, in_channels,
                     kernel_size, kernel_size)
        self.weight = Parameter(w.cuda())
        self.bias = Parameter(t.Tensor(out_channels).cuda())
        self.output = output

        self.stride = (1,) * 2
        self.dilation = (dilation, ) * 2
        self.padding = (padding, ) * 2

    def forward(self, input):
        assert(self.output.is_cuda)
        return convNdInPlace(input, self.weight, self.bias,
                             self.output, self.padding, self.stride,
                             self.dilation)


class Conv3dInPlaceModule(nn.Module):

    def __init__(self, output, in_channels, out_channels,
                 kernel_size=3, dilation=1, padding=1):
        super(Conv3dInPlaceModule, self).__init__()

        w = t.cuda.FloatTensor(out_channels, in_channels, *([kernel_size] * 3))
        self.weight = Parameter(w.cuda())
        self.bias = Parameter(t.cuda.FloatTensor(out_channels).cuda())
        self.output = output
        self.stride = (1, ) * 3
        self.dilation = (dilation, ) * 3
        self.padding = (padding, ) * 3

    def forward(self, input):
        assert(self.output.is_cuda)
        return convNdInPlace(input, self.weight, self.bias,
                             self.output, self.padding, self.stride,
                             self.dilation)
