import torch.nn as nn
import torch as t
import torch.nn.init as init
from msd_pytorch.conv_inplace import (Conv2dInPlaceModule, Conv3dInPlaceModule)
from msd_pytorch.stitch import (stitchLazy, stitchCopy)


def msd_dilation(i):
    return i % 10 + 1


def one_dilation(i):
    return 1


class MSDLayerModule(nn.Module):
    def __init__(self, L, G, i, dilation, conv3d=False):
        super(MSDLayerModule, self).__init__()

        self.L, self.G, self.i = L, G, i

        output = None
        if conv3d:
            self.convolution = Conv3dInPlaceModule(
                output, i, 1, kernel_size=3,
                dilation=dilation, padding=dilation)
        else:
            self.convolution = Conv2dInPlaceModule(
                output, i, 1, kernel_size=3,
                dilation=dilation, padding=dilation)
        init.xavier_normal(self.convolution.weight)
        self.convolution.bias.data.zero_()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        # Set output
        self.convolution.output = self.L.narrow(1, self.i, 1)
        output = self.convolution(input)
        output = self.relu(output)
        output = stitchLazy(output, self.L, self.G, self.i)
        return output


class MSDModule(nn.Module):
    def __init__(self, depth, dilation_function, conv3d=False):
        super(MSDModule, self).__init__()
        #
        self.depth = depth
        self.dilation_function = dilation_function
        self.conv3d = conv3d
        #
        L = t.Tensor(1).cuda()
        G = L.clone().zero_()
        self.register_buffer('L', L)
        self.register_buffer('G', G)

        layers = [MSDLayerModule(self.L, self.G, i + 1,
                                 dilation_function(i),
                                 conv3d=conv3d)
                  for i in range(depth)]

        if conv3d:
            self.c_out = nn.Conv3d(depth + 1, 1, kernel_size=1)
        else:
            self.c_out = nn.Conv2d(depth + 1, 1, kernel_size=1)
        init.xavier_normal(self.c_out.weight.data)
        self.c_out.bias.data.zero_()
        final_relu = nn.ReLU(inplace=True)
        self.net = nn.Sequential(*(layers + [self.c_out, final_relu])).cuda()

    def forward(self, input):
        nb, nc, *shape = input.data.shape
        # Ensure that L and G are the correct size
        new_shape = (nb, (self.depth + 1) * nc, *shape)
        self.L.resize_(*new_shape)
        self.G.resize_(*new_shape)
        self.G.zero_()          # clear gradient cache

        input = stitchCopy(input, self.L, self.G, 0)
        return self.net(input)

    def clear_buffers(self):
        # Clear the L and G buffers.
        # Don't call this function between forward and backward! The
        # backward pass requires the L and G buffers.
        L_new, G_new = self.L.new(1), self.G.new(1)
        self.L.set_(L_new)      # This replaces the underlying storage
        self.G.set_(G_new)      # of L and G.

    def grow(self, add_depth=1):
        new = MSDModule(self.depth + add_depth, self.dilation_function,
                        self.conv3d)

        cur_params = self.net.state_dict()
        new_params = new.net.state_dict()

        for k, cur_p in cur_params.items():
            new_p = new_params.get(k)
            if new_p is not None:
                new_p.copy_(cur_p)

        # Reinitialize the final convolution
        cur_w = self.c_out.weight.data
        new_w = new.c_out.weight.data
        new_w[:, :(self.depth + 1), :, :].copy_(cur_w)
        cur_b = self.c_out.bias.data
        new_b = new.c_out.bias.data
        new_b.zero_()
        new_b[:(self.depth + 1)].copy_(cur_b)

        return new
