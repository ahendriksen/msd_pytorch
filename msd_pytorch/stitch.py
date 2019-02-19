"""Stitch Functions and Modules for threading the gradient

Stitching refers to the practice of copying and / or reusing shared
buffers in a network to improve efficiency. It handles distributing
the gradient transparently.

In this module, we implement three types of stitching:

1) Slow stitching: concatenates to inputs in the forward pass and
   distributes the gradient output in the backward
   pass. Inefficient. Slow stitching is used for testing.

2) Copy Stitching: copies the input into a layer buffer ``L`` and returns
   all layers up to and including the newly copied input. More
   efficient than slow stitching, but preferably used sparingly.

3) Lazy Stitching: assumes that the input has already been copied in
   the layer buffer ``L`` and returns all layers up to and including
   the input. The gradient is accumulated in a gradient buffer
   ``G``. This is fast and efficient.

"""

import torch.nn as nn
import torch as t
from torch.autograd import Variable, Function


class StitchBuffer:
    def __init__(self):
        """Holds the ``L`` and ``G`` buffers for a stitched module.

        The intermediate layers are stored in ``L`` for the forward
        pass. The gradients are stored in the ``G`` buffer.

        """
        self.L = t.zeros(1)
        self.G = t.zeros(1)

    def like_(self, tensor, new_shape):
        """Change the ``L`` and ``G`` buffers to match tensor.

        Matches the tensor's
        - data type
        - device (cpu, cuda, cuda:0, cuda:i)

        The shape is taken from the `new_shape` parameter.

        :param tensor: An input tensor
        :param new_shape: The new shape that the buffer should have.
        :returns: Nothing
        :rtype:

        """
        make_new = (
            new_shape != self.L.shape
            or tensor.dtype != self.L.dtype
            or tensor.device != self.L.device
        )

        options = {"dtype": tensor.dtype, "device": tensor.device}
        if make_new:
            self.L = tensor.new_zeros(new_shape, **options)
            self.G = tensor.new_zeros(new_shape, **options)

    def zero_(self):
        """Set buffers to zero.

        :returns:
        :rtype:

        """
        self.L.zero_()
        self.G.zero_()


class StitchSlowFunction(Function):
    """StitchSlowFunction

    Naive stitching: concatenates two inputs in the channel dimension.
    """

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


class StitchCopyFunction(Function):
    """Copy stitching:

    Stores output in buffer ``L`` in the forward pass and adds the
    ``grad_output`` to buffer ``G`` in the backward pass.

    The buffer ``L`` is a tensor of dimensions `B x C x ?` where

    * `B` is the minibatch size, and
    * `C` is the number of channels.

    The buffer ``G`` has the same dimension as ``L``.

    The parameter ``i`` is an index in the `C` dimension and points to
    where the input (the output of the previous layer) must be copied.

    In the forward pass:

    * write the input into ``L`` at channel ``i``
    * return ``L`` up to and including channel ``i``

    In the backward pass:

    * add the ``grad_output`` to ``G``
    * return channel ``i`` of ``G``

    It is good practice to zero the ``G`` buffer before the backward
    pass. Sometimes, this is not possible since some methods, such as
    ``torch.autograd.gradcheck``, repeatedly call ``.grad()`` on the
    output. Therefore, when ``grad_output`` is the same size as ``G``, the
    buffer ``G`` is zeroed in the ``backward`` function.

    """

    @staticmethod
    def forward(ctx, input, L, G, i):
        assert input.dtype == L.dtype, (
            f"Input type ({input.dtype}) and layer L type ({L.dtype}) "
            "should be the same. "
        )

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
        # If grad_output has the same size size as G, zero-fill the
        # gradient buffer G. This way, we make the gradient
        # computation reentrant -- it can be repeated multiple times.
        if i + width == G.size(1):
            G_output = G
            G_output.fill_(0)
        else:
            G_output = G.narrow(1, 0, i + width)

        G_output.add_(grad_output.data)
        G_input = G.narrow(1, i, width)
        return G_input, None, None, None


stitchCopy = StitchCopyFunction.apply


class StitchCopyModule(nn.Module):
    def __init__(self, buffer, i):
        """Make a new StitchCopyModule

        :param buffer: A StitchBuffer
        :param i: index of the output channel of the stitch
        :returns:
        :rtype:

        """
        super(StitchCopyModule, self).__init__()
        self.buffer = buffer
        self.i = i

    def forward(self, input):
        return stitchCopy(input, self.buffer.L, self.buffer.G, self.i)


class StitchLazyFunction(Function):
    """StitchLazyFunction


    ``StitchLazyFunction`` is similar to ``StitchCopyFunction``, but it
    does not copy the output of the previous layer into ``L``. Hence the
    name. ``StitchLazyFunction`` supposes that the output of the the
    previous layer has already been copied into ``L``. This can be
    accomplished with ``conv_cuda.conv2dInPlace``, for instance.

    The buffer ``L`` is a tensor of dimensions `B x C x ?` where

    * `B` is the minibatch size, and
    * `C` is the number of channels.

    The buffer ``G`` has the same dimension as ``L``.

    The parameter ``i`` is an index in the `C` dimension and points to
    where the input (the output of the previous layer) must be copied.

    In the forward pass:

    * write the input into ``L`` at channel ``i``

    In the backward pass:

    * add the ``grad_output`` to ``G``
    * return channel ``i`` of ``G``

    It is good practice to zero the ``G`` buffer before the backward
    pass. Sometimes, this is not possible since some methods, such as
    ``torch.autograd.gradcheck``, repeatedly call ``.grad()`` on the
    output. Therefore, when ``grad_output`` is the same size as ``G``, the
    buffer ``G`` is zeroed in the ``backward`` function.

    """

    @staticmethod
    def forward(ctx, input, L, G, i):
        width = input.shape[1]
        ctx.G, ctx.i, ctx.width = G, i, width
        return L.narrow(1, 0, i + width)

    @staticmethod
    def backward(ctx, grad_output):
        G, i, width = ctx.G, ctx.i, ctx.width
        # If grad_output has the same size size as G, zero-fill the
        # gradient buffer G. This way, we make the gradient
        # computation reentrant -- it can be repeated multiple times.
        if i + width == G.size(1):
            G_output = G
            G_output.fill_(0)
        else:
            G_output = G.narrow(1, 0, i + width)

        G_output.add_(grad_output.data)
        G_input = G.narrow(1, i, width)

        return Variable(G_input), None, None, None


stitchLazy = StitchLazyFunction.apply


class StitchLazyModule(nn.Module):
    def __init__(self, buffer, i):
        """Make a new StitchLazyModule

        :param buffer: A StitchBuffer
        :param i: index of the output channel of the stitch
        :returns:
        :rtype:

        """
        super(StitchLazyModule, self).__init__()
        self.buffer = buffer
        self.i = i

    def forward(self, input):
        return stitchLazy(input, self.buffer.L, self.buffer.G, self.i)
