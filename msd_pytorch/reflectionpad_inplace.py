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
            input[:, :, :, -(i + 1)] = input[:, :, :, -(2 * padB - i + 1)]

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
            g[:, :, :, -(2 * padB - i + 1)] = g[:, :, :, -(i + 1)]
            g[:, :, :, -(i + 1)].fill_(0)

        return Variable(g), None


reflectionPad2DInplace = ReflectionPad2DInplaceFunction.apply


class ReflectionPad2DInplaceModule(nn.Module):
    def __init__(self, padding):
        super(ReflectionPad2DInplaceModule, self).__init__()
        self.padding = _ntuple(4)(padding)

    def forward(self, input):
        return reflectionPad2DInplace(input, self.padding)
