import torch.nn as nn
import torch.utils.cpp_extension as cppe
from torch.autograd import Function
import os

os.environ['PATH'] = '/opt/sw/gcc-7.3.0/bin:' + os.environ['PATH']
os.environ['PATH'] = '/opt/gcc-7.3.0/bin:' + os.environ['PATH']


relu_inplace = cppe.load('relu_inplace',
                         sources=['msd_pytorch/relu_inplace.cpp',
                                  'msd_pytorch/relu_inplace_cuda.cu'],
                         extra_include_paths=cppe.include_paths(cuda=True),
                         verbose=True)


class ReLUInplaceFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = relu_inplace.forward(input)
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
        return relu_inplace.backward(output, grad_output)


class ReLUInplaceModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return ReLUInplaceFunction.apply(input)
