import torch.nn as nn
import torch as t
from msd_pytorch.trp_conv_inplace import (
    Conv2dInPlaceModule, Conv3dInPlaceModule)
from msd_pytorch.stitch import (stitchLazy, StitchCopyModule)
from msd_pytorch.relu_inplace import (ReLUInplaceModule)
from math import sqrt
from functools import reduce
from operator import mul
import warnings

max_dilation = 10


def msd_dilation(i):
    return i % 10 + 1


def one_dilation(i):
    return 1


def product(xs):
    return reduce(mul, xs, 1)


def units_in_front(c_in, width, depth):
    return c_in + width * (depth - 1)


def init_convolution_weights(conv_weight, c_in, c_out, width, depth):
    # the number of parameters differs between 2d and 3d convolution (and
    # depends on kernel_size)
    np = product(conv_weight.shape[2:])
    nc = np * (c_in + width * (depth - 1)) + c_out
    std_dev = sqrt(2 / nc)
    conv_weight.normal_(0, std_dev)


class MSDLayerModule(nn.Module):
    def __init__(self, L, G, c_in, c_out, d, width, dilation, reflect=False,
                 conv3d=False):
        super(MSDLayerModule, self).__init__()

        in_front = units_in_front(c_in, width, d)
        self.L, self.G, self.in_front, self.width = L, G, in_front, width

        output = None

        if not reflect:
            warnings.warn("Zero-padding is not supported anymore. "
                          "Using reflection-padding instead.")
        self.reflect = None

        if conv3d:
            self.convolution = Conv3dInPlaceModule(
                output, in_front, width, kernel_size=3,
                dilation=dilation, padding=dilation)
        else:
            self.convolution = Conv2dInPlaceModule(
                output, in_front, width, kernel_size=3,
                dilation=dilation, padding=dilation)

        # Initialize the weights
        init_convolution_weights(self.convolution.weight.data,
                                 c_in, c_out, width, d)
        self.convolution.bias.data.zero_()

        # Add the relu to get a nice printout for the network from
        # pytorch.
        self.relu = ReLUInplaceModule()

    def forward(self, input):
        # Set output
        self.convolution.output = self.L.narrow(1, self.in_front, self.width)
        output = self.convolution(input)
        output = self.relu(output)
        output = stitchLazy(output, self.L, self.G, self.in_front)
        return output


class MSDFinalLayer(nn.Module):
    """Documentation for MSDFinalLayer

    Implements the final 1x1 multiplication and bias addition for all
    intermediate layers to get to the output layer.

    Initializes the weight and bias to zero.
    """
    def __init__(self, c_in, c_out):
        super(MSDFinalLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = nn.Conv1d(c_in, c_out, 1)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, input):
        b, c_in, *size = input.shape
        tmp_size = input[0, 0, ...].numel()

        # Put channels in last place in input shape
        output = input.view(b, c_in, tmp_size)
        output = self.linear(output)
        output = output.view(b, self.c_out, *size)
        return output


class MSDModule(nn.Module):
    def __init__(self, c_in, c_out, depth, width, dilation_function,
                 reflect=False, conv3d=False):
        """Create a msd module

        :param c_in: # of input channels
        :param c_out: # of output channels
        :param depth: # of layers
        :param width: # the width of the module
        :param dilation_function:
        :param reflect:
            Whether or not to use reflection padding instead of zero padding.
        :param conv3d:

        :param dilation_function: this fuction determines the dilation
        of the convolution in each layer. Usually, you will want to
        use `msd_dilation`.

        :param conv3d: Whether or not to use 3d convolutions (bool).
        :returns:
        :rtype:

        """
        super(MSDModule, self).__init__()
        #
        self.c_in = c_in
        self.c_out = c_out
        self.depth = depth
        self.width = width
        self.dilation_function = dilation_function
        self.conv3d = conv3d
        self.reflect = reflect
        #
        L = t.Tensor(1).cuda()
        G = L.clone().zero_()
        self.register_buffer('L', L)
        self.register_buffer('G', G)

        # The first layer copies input into the L buffer
        layers = [StitchCopyModule(L, G, 0)]

        layers += [MSDLayerModule(self.L, self.G, c_in, c_out, d, width,
                                  dilation_function(d),
                                  conv3d=conv3d)
                   for d in range(1, depth + 1)]

        in_front = units_in_front(c_in, width, depth + 1)
        self.c_final = MSDFinalLayer(in_front, c_out)

        self.net = nn.Sequential(*(layers + [self.c_final]))

        self.net.cuda()

    def forward(self, input):
        self.init_buffers(input.data.shape)
        return self.net(input)

    def init_buffers(self, input_shape):
        batch_sz, c_in, *shape = input_shape

        assert c_in == self.c_in, "Unexpected number of input channels"
        # Ensure that L and G are the correct size
        total_units = units_in_front(self.c_in, self.width, self.depth + 1)
        new_shape = (batch_sz, total_units, *shape)
        self.L.resize_(*new_shape)
        self.G.resize_(*new_shape)
        self.G.zero_()          # clear gradient cache

    def clear_buffers(self):
        # Clear the L and G buffers. Allocates a buffer containing one
        # item. Don't call this function between forward and
        # backward! The backward pass requires the L and G buffers.
        L_new, G_new = self.L.new(1), self.G.new(1)
        self.L.set_(L_new)      # This replaces the underlying storage
        self.G.set_(G_new)      # of L and G.
