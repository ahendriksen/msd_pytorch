import torch.nn as nn
from torch.autograd import (Variable, Function)
import torch.utils.cpp_extension as cppe
import os


# This is a hack for CWI workstation, which have a too recent version
# of GCC installed.
os.environ['PATH'] = '/opt/sw/gcc-7.3.0/bin:' + os.environ['PATH']
os.environ['PATH'] = '/opt/gcc-7.3.0/bin:' + os.environ['PATH']


def recompile():
    ref_inplace = cppe.load('reflectionpad_inplace',
                            sources=['msd_pytorch/reflectionpad_inplace.cpp',
                                     'msd_pytorch/reflectionpad_inplace_cuda.cu'],
                            extra_cflags=['-Werror', '-Wfatal-errors', '-Wextra'],
                            # extra_cuda_cflags=[ '-Werror', '-Wfatal-errors', '-Wextra'],
                            extra_include_paths=cppe.include_paths(cuda=True),
                            verbose=True)
    return ref_inplace


reflectionpad_inplace = recompile()


class ReflectionPad2DInplaceFunction(Function):
    @staticmethod
    def forward(ctx, input, padding):
        ctx.padding = padding

        assert 3 * padding < input.shape[2], \
            "Too much padding for height"
        assert 3 * padding < input.shape[3], \
            "Too much padding for width"

        output = reflectionpad_inplace.forward(input, padding)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        padding = ctx.padding
        return reflectionpad_inplace.backward(gradOutput, padding), None


reflectionPad2DInplace = ReflectionPad2DInplaceFunction.apply


class ReflectionPad2DInplaceModule(nn.Module):
    def __init__(self, padding):
        super(ReflectionPad2DInplaceModule, self).__init__()
        self.padding = padding

    def forward(self, input):
        return reflectionPad2DInplace(input, self.padding)


class crop2dFunction(Function):
    @staticmethod
    def forward(ctx, input, crop_by=1):
        ctx.crop_by = crop_by

        output = input.clone()
        storage = output.storage()

        shape = output.shape

        storage_offset = crop_by * (1 + shape[3])
        stride = output.stride()
        size = shape[:2] + tuple(shape[i] - 2 * crop_by for i in [2, 3])

        output.set_(storage, storage_offset=storage_offset, size=size,
                    stride=stride)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        crop_by = ctx.crop_by

        shape = grad_output.data.shape
        size = shape[:2] + tuple(shape[i] + 2 * crop_by for i in [2, 3])

        grad_input = grad_output.data.new(*size).zero_()
        storage_offset = crop_by * (1 + size[3])
        stride = grad_input.stride()

        cropped = grad_input.new()
        cropped.set_(grad_input.storage(),
                     storage_offset=storage_offset, size=shape, stride=stride)
        cropped.copy_(grad_output.data)

        return Variable(grad_input), None


crop2d = crop2dFunction.apply


class Crop2DModule(nn.Module):
    def __init__(self, crop_by):
        super(Crop2DModule, self).__init__()
        self.crop_by = crop_by

    def forward(self, input):
        return crop2d(input, self.crop_by)
