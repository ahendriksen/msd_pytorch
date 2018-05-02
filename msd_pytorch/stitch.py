import torch.nn as nn
import torch as t
from torch.autograd import (Variable, Function)

# Stitching refers to the practice of copying and / or reusing shared
# buffers in a network to improve efficiency. It handles distributing
# the gradient transparently.

# In this module, we implement three types of stitching:

# 1) Slow stitching: concatenates to inputs in the forward pass and
#    distributes the gradient output in the backward
#    pass. Inefficient. Slow stitching is used for testing.
#
# 2) Copy Stitching: copies the input into a layer buffer L and returns
#    all layers up to and including the newly copied input. More
#    efficient than slow stitching, but preferably used sparingly.
#
# 3) Lazy Stitching: assumes that the input has already been copied in
#    the layer buffer L and returns all layers up to and including the
#    input. The gradient is accumulated in a gradient buffer G. Fast
#    and efficient.


class StitchSlowFunction(Function):
    # Naive stitching: concatenates two inputs in the channel
    # dimension.
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        return t.cat((input1, input2), 1)

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        nc1 = input1.shape[1]

        return grad_output[:, :nc1, :, :], grad_output[:, nc1:, :, :]


stitchSlow = StitchSlowFunction.apply


# Copy stitching:
# Has buffers L and G for the forward and backward pass.
#
# L is a tensor of dim B x C x ?
# where B is the minibatch size, and
#       C is the number of channels.
#
# G has the same dimension as L.
#
# i is an index in the C dimension and points to where the input (the
# output of the previous layer) must be copied.
#
# In the forward pass:
# - writes the input into L at channel i
# - return L up to and including channel i
#
# In the backward pass:
# - add the grad_output to G
# - return channel i of G
#
# NB: G must be zeroed before the backward pass!
class StitchCopyFunction(Function):
    @staticmethod
    def forward(ctx, input, L, G, i):
        width = input.shape[1]
        ctx.G, ctx.i, ctx.width = G, i, width
        # Decouple L as a variable from the input
        L_input = L.data.narrow(1, i, width)
        L_input.copy_(input)
        L_output = L.data.narrow(1, 0, i + width)
        return L_output

    @staticmethod
    def backward(ctx, grad_output):
        G, i, width = ctx.G, ctx.i, ctx.width
        G_output = G.narrow(1, 0, i + width)
        G_output.add_(grad_output.data)
        G_input = G.narrow(1, i, width)
        return G_input, None, None, None


stitchCopy = StitchCopyFunction.apply


class StitchCopyModule(nn.Module):
    def __init__(self, L, G, i):
        super(StitchCopyModule, self).__init__()
        self.L, self.G, self.i = L, G, i

    def forward(self, input):
        return stitchCopy(input, self.L, self.G, self.i)


class StitchLazyFunction(Function):
    # StitchLazy is similar to Stitch, but it does not copy the output
    # of the previous layer into L. Hence the name. StitchLazy
    # supposes that the output of the the previous layer has already
    # been copied into L. This can be accomplished with Conv2dInPlace,
    # for instance.
    @staticmethod
    def forward(ctx, input, L, G, i):
        width = input.shape[1]
        ctx.G, ctx.i, ctx.width = G, i, width
        return L.narrow(1, 0, i + width)

    @staticmethod
    def backward(ctx, grad_output):
        G, i, width = ctx.G, ctx.i, ctx.width

        G_output = G.narrow(1, 0, i + width)
        G_output.add_(grad_output.data)

        G_input = G.narrow(1, i, width)
        return Variable(G_input), None, None, None


stitchLazy = StitchLazyFunction.apply


class StitchLazyModule(nn.Module):
    def __init__(self, L, G, i):
        super(StitchLazyModule, self).__init__()
        self.L = L
        self.G = G
        self.i = i

    def forward(self, input):
        return stitchLazy(input, self.L, self.G, self.i)
