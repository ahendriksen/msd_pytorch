import torch.nn as nn
import relu_cuda
from torch.autograd import Function


class ReLUInplaceFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = relu_cuda.forward(input)
        # Unfortunately, we cannot mark the input as dirty. Marking
        # the input dirty causes the pytorch execution engine to
        # realize that the input might be a view into another tensor
        # variable and this breaks the backward gradient computation.
        # ctx.mark_dirty(input)

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        return relu_cuda.backward(output, grad_output)


class ReLUInplaceModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return ReLUInplaceFunction.apply(input)
