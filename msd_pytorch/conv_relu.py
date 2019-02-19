import torch.nn as nn
import torch as t
from torch.autograd import Function
from torch.nn import Parameter
import conv_relu_cuda


class ConvRelu2dInPlaceFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, output, stride, dilation):
        ctx.dilation = dilation
        conv_relu_cuda.conv_relu_forward(input, weight, bias, output.data, dilation)
        ctx.save_for_backward(input, output, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # restore variables
        input, output, weight, bias = ctx.saved_tensors
        dilation = ctx.dilation

        # ensure that grad_output is contiguous
        grad_output = grad_output.contiguous()

        # Input
        grad_input = t.zeros_like(input)
        conv_relu_cuda.conv_relu_backward_x(
            output, grad_output, weight, grad_input, dilation
        )
        # Weight
        grad_weight = t.zeros_like(weight)
        conv_relu_cuda.conv_relu_backward_k(
            output, grad_output, input, grad_weight, dilation
        )
        # Bias
        grad_bias = t.zeros_like(bias)
        conv_relu_cuda.conv_relu_backward_bias(output, grad_output, grad_bias)

        return grad_input, grad_weight, grad_bias, None, None, None, None


conv_relu2dInPlace = ConvRelu2dInPlaceFunction.apply


class ConvRelu2dInPlaceModule(nn.Module):
    def __init__(self, output, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()

        w = t.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        self.weight = Parameter(w.cuda())
        self.bias = Parameter(t.Tensor(out_channels).cuda())
        self.output = output

        self.dilation = dilation
        self.stride = 1

    def forward(self, input):
        assert self.output.is_cuda
        return conv_relu2dInPlace(
            input, self.weight, self.bias, self.output, self.stride, self.dilation
        )
