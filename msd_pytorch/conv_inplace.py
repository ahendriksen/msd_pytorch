import torch.nn as nn
import torch as t
from torch.autograd import (Variable, Function)
from torch.nn import Parameter
from torch.backends import cudnn

benchmark = False
# If deterministic is set to True, then torch will always use the
# CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
# algorithm. Unfortunately, this algorithm does not support
# dilation. If you set the dilation to anything other than 1, the
# computation will fail. Thus we set deterministic to False.
deterministic = False


def cudnn_convolution_full_forward(input, weight, bias, output_contig,
                                   padding, stride, dilation):
    groups = 1
    return t._C._cudnn_convolution_full_forward(
        input, weight, bias, output_contig,
        padding, stride, dilation,
        groups, benchmark, deterministic)


def cudnn_convolution_backward_data(grad_output, grad_input, weight, info):
    t._C._cudnn_convolution_backward_data(
        grad_output, grad_input, weight,
        info, benchmark, deterministic)


def cudnn_convolution_backward_filter(grad_output, input, grad_weight, info):
    t._C._cudnn_convolution_backward_filter(
        grad_output, input, grad_weight,
        info, benchmark, deterministic)


def cudnn_convolution_backward_bias(grad_output, grad_bias, info):
    t._C._cudnn_convolution_backward_bias(grad_output, grad_bias, info)


class ConvNdInPlaceFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, output, padding, stride, dilation):
        # save_for_backward can only save input or output
        # tensors. Since we require the output to be contiguous, we
        # save the contiguous version of output. We cannot save the
        # contiguous version of input, so we save the (possibly)
        # non-contiguous version.

        # if not (output.is_contiguous() and input.is_contiguous()):
        #     print("Warning input or output non-contiguous")

        copy_output = not output.is_contiguous()
        output_contig = output.contiguous()
        ctx.save_for_backward(input, weight, bias)
        input = input.contiguous()

        ctx.info = cudnn_convolution_full_forward(
            input, weight, bias, output_contig, padding, stride, dilation)

        if copy_output:
            output.copy_(output_contig)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # restore variables
        input, weight, bias = ctx.saved_tensors
        info = ctx.info
        grad_output = grad_output.data

        # make input contiguous (again)
        input = input.contiguous()
        # ensure that grad_output is contiguous
        grad_output = grad_output.contiguous()

        grad_input = input.clone()
        cudnn_convolution_backward_data(
            grad_output, grad_input, weight, info)

        grad_input = Variable(grad_input)

        grad_weight = weight.clone()
        cudnn_convolution_backward_filter(grad_output, input, grad_weight,
                                          info)

        grad_weight = Variable(grad_weight)

        grad_bias = bias.clone()
        cudnn_convolution_backward_bias(grad_output, grad_bias, info)
        grad_bias = Variable(grad_bias)

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
