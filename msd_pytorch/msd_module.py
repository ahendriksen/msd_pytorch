import torch.nn as nn
import torch as t
import torch.nn.init as init
from msd_pytorch.conv_inplace import (Conv2dInPlaceModule, Conv3dInPlaceModule)
from msd_pytorch.reflectionpad_inplace import (ReflectionPad2DInplaceModule, Crop2DModule)
from msd_pytorch.stitch import (stitchLazy, stitchCopy, StitchCopyModule)
from math import sqrt
from functools import reduce
from operator import mul

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

        if max_dilation < dilation:
            raise(RuntimeError("max_dilation (10) exceeded. " +
                               "Contact me to increase it manually."))

        if reflect and conv3d:
            raise(RuntimeError("3d Reflection padding not yet supported."))
        elif reflect and not conv3d:
            self.reflect = ReflectionPad2DInplaceModule(dilation)
        else:
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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        # Set output
        self.convolution.output = self.L.narrow(1, self.in_front, self.width)
        output = self.convolution(input)
        if self.reflect is not None:
            output = self.reflect(output)
        output = self.relu(output)
        output = stitchLazy(output, self.L, self.G, self.in_front)
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
        :param reflect: Whether or not to use reflection padding instead of zero padding.
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

        # Add margins to the layers:
        if reflect:
            layers = [nn.ReflectionPad2d(max_dilation),
                      StitchCopyModule(L, G, 0)]
        else:
            layers = [StitchCopyModule(L, G, 0)]

        layers += [MSDLayerModule(self.L, self.G, c_in, c_out, d, width,
                                  dilation_function(d),
                                  conv3d=conv3d)
                   for d in range(1, depth + 1)]

        in_front = units_in_front(c_in, width, depth + 1)
        if conv3d:
            self.c_final = nn.Conv3d(in_front, c_out, kernel_size=1)
        else:
            self.c_final = nn.Conv2d(in_front, c_out, kernel_size=1)

        self.c_final.weight.data.zero_()
        self.c_final.bias.data.zero_()
        self.net = nn.Sequential(*(layers + [self.c_final]))

        # Remove the margins at the end:
        if reflect:
            self.net = nn.Sequential(self.net, Crop2DModule(max_dilation))
        self.net.cuda()

    def forward(self, input):
        self.init_buffers(input.data.shape)
        return self.net(input)

    def init_buffers(self, input_shape):
        batch_sz, c_in, *shape = input_shape
        # If reflection padding is active, the intermediate buffers
        # must be bigger too.
        if self.reflect:
            shape = [s + 2 * max_dilation for s in shape]

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

    def grow(self, add_depth=1):
        new = MSDModule(self.c_in, self.c_out, self.depth + add_depth,
                        self.width, self.dilation_function, self.conv3d)

        cur_params = self.net.state_dict()
        new_params = new.net.state_dict()

        for k, cur_p in cur_params.items():
            new_p = new_params.get(k)
            if new_p is not None:
                new_p.copy_(cur_p)

        # Reinitialize the final convolution
        cur_w = self.c_final.weight.data
        new_w = new.c_final.weight.data
        new_w[:, :(cur_w.shape[1]), :, :].copy_(cur_w)
        cur_b = self.c_final.bias.data
        new_b = new.c_final.bias.data
        new_b.zero_()
        new_b[:(self.depth + 1)].copy_(cur_b)

        return new
