import torch as t
import torch.nn as nn
from torch.autograd import (Variable, Function)
from torch.nn.modules.utils import (_ntuple)


class ReflectionPad2DInplaceFunction(Function):
    @staticmethod
    def forward(ctx, input, padding):
        ctx.padding = padding
        padL, padR, padT, padB = padding

        assert padT + padB + max(padT, padB) < input.shape[2], \
            "Too much padding for height"
        assert padL + padR + max(padL, padR) < input.shape[3], \
            "Too much padding for width"

        for i in range(padT):
            input[:, :, i, :] = input[:, :, 2 * padT - i, :]

        for i in range(padB):
            input[:, :, -(i + 1), :] = input[:, :, -(2 * padB - i + 1), :]

        for i in range(padL):
            input[:, :, :, i] = input[:, :, :, 2 * padL - i]

        for i in range(padR):
            input[:, :, :, -(i + 1)] = input[:, :, :, -(2 * padR - i + 1)]

        # This is necessary to convince pytorch that input was used in
        # the calculation. It notices that we output the same tensor
        # as was put in.
        ctx.mark_dirty(input)

        return input

    @staticmethod
    def backward(ctx, gradOutput):
        padL, padR, padT, padB = ctx.padding

        g = gradOutput.data.clone()
        for i in range(padT):
            g[:, :, 2 * padT - i, :] += g[:, :, i, :]
            g[:, :, i, :].fill_(0)

        for i in range(padB):
            g[:, :, -(2 * padB - i + 1), :] += g[:, :, -(i + 1), :]
            g[:, :, -(i + 1), :].fill_(0)

        for i in range(padL):
            g[:, :, :, 2 * padL - i] += g[:, :, :, i]
            g[:, :, :, i].fill_(0)

        for i in range(padR):
            g[:, :, :, -(2 * padR - i + 1)] = g[:, :, :, -(i + 1)]
            g[:, :, :, -(i + 1)].fill_(0)

        return Variable(g), None


reflectionPad2DInplace = ReflectionPad2DInplaceFunction.apply


class ReflectionPad2DInplaceModule(nn.Module):
    def __init__(self, padding):
        super(ReflectionPad2DInplaceModule, self).__init__()
        self.padding = _ntuple(4)(padding)

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
