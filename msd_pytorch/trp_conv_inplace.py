import torch.nn as nn
import torch as t
from torch.autograd import Function
from torch.nn import Parameter
import conv_cuda


class ConvNdInPlaceFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, output, stride, dilation):
        # save_for_backward can only save input or output
        # tensors. Since we require the output to be contiguous, we
        # save the contiguous version of output. We cannot save the
        # contiguous version of input, so we save the (possibly)
        # non-contiguous version.

        ctx.save_for_backward(input, weight, bias)
        ctx.dilation = dilation

        conv_cuda.conv_forward(input, weight, bias, output.data, dilation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # restore variables
        input, weight, bias = ctx.saved_tensors
        dilation = ctx.dilation

        # ensure that grad_output is contiguous
        grad_output = grad_output.contiguous()

        # Input
        grad_input = t.zeros_like(input)
        conv_cuda.conv_backward_x(grad_output, weight, grad_input, dilation)
        # Weight
        grad_weight = t.zeros_like(weight)
        conv_cuda.conv_backward_k(grad_output, input, grad_weight, dilation)
        # Bias
        grad_bias = t.zeros_like(bias)
        conv_cuda.conv_backward_bias(grad_output, grad_bias)

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

        self.dilation = dilation
        self.stride = 1

    def forward(self, input):
        assert(self.output.is_cuda)
        return convNdInPlace(input, self.weight, self.bias,
                             self.output, self.stride, self.dilation)


class Conv3dInPlaceModule(nn.Module):

    def __init__(self, output, in_channels, out_channels,
                 kernel_size=3, dilation=1, padding=1):
        super(Conv3dInPlaceModule, self).__init__()

        w = t.cuda.FloatTensor(out_channels, in_channels, *([kernel_size] * 3))
        self.weight = Parameter(w.cuda())
        self.bias = Parameter(t.cuda.FloatTensor(out_channels).cuda())
        self.output = output
        self.stride = 1
        self.dilation = dilation

    def forward(self, input):
        assert(self.output.is_cuda)
        return convNdInPlace(input, self.weight, self.bias,
                             self.output, self.stride, self.dilation)
