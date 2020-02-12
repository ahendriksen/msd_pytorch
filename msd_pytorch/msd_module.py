import torch.nn as nn
from msd_pytorch.conv import Conv2dInPlaceModule
from msd_pytorch.conv_relu import ConvRelu2dInPlaceModule
from msd_pytorch.stitch import stitchLazy, StitchCopyModule, StitchBuffer
from math import sqrt
import numpy as np


def units_in_front(c_in, width, layer_depth):
    """Calculate how many intermediate images are in front of current layer

    * The input channels count as intermediate images
    * The `layer_depth` index is zero-based: the first hidden layer
      has index zero.

    :param c_in: The number of input channels of the MSD module
    :param width: The width of the MSD module
    :param layer_depth:
        The depth of the layer for which we are calculating the units
        in front.  This index is zero-based: the first hidden layer
        has index zero.
    :returns:
    :rtype:

    """
    return c_in + width * layer_depth


def init_convolution_weights(conv_weight, c_in, c_out, width, depth):
    """Initialize MSD convolution kernel weights

    Based on:

    Pelt, Daniel M., & Sethian, J. A. (2017). A mixed-scale dense
    convolutional neural network for image analysis. Proceedings of
    the National Academy of Sciences, 115(2),
    254–259. http://dx.doi.org/10.1073/pnas.1715832114

    :param conv_weight:
        The kernel weight data
    :param c_in:
        Number of input channels of the MSD module
    :param c_out:
        Number of output channels of the MSD module
    :param width:
        The width of the MSD module
    :param depth:
        The depth of the MSD module. This is the number of hidden layers.
    :returns: Nothing
    :rtype:

    """
    # The number of parameters in the convolution depends on whether
    # the convolution is 2D or 3D. We multiply all non-channel
    # dimensions of the weight here to get the right answer.
    num_params = np.product(conv_weight.shape[2:])
    num_channels = num_params * (c_in + width * (depth - 1)) + c_out
    std_dev = sqrt(2 / num_channels)
    conv_weight.normal_(0, std_dev)


class MSDLayerModule(nn.Module):
    """A hidden layer of the MSD module.

    The primary responsibility of this module is to define the
    `forward()` method.

    This module is used by the `MSDModule`.

    This module is not responsible for

    * Buffer management
    * Weight initialization
    """

    def __init__(self, buffer, c_in, layer_depth, width, dilation):
        """Initialize the hidden layer.

        :param buffer: a StitchBuffer object for storing the L and G buffers.
        :param c_in: The number of input channels of the MSD module.
        :param layer_depth:
            The depth of this layer in the MSD module.  This index is
            zero-based: the first hidden layer has index zero.
        :param width: The width of the MSD module.
        :param dilation:
            An integer describing the dilation factor for the
            convolutions in this layer.
        :returns: A module for the MSD hidden layer.
        :rtype: MSDLayerModule

        """
        super(MSDLayerModule, self).__init__()

        in_front = units_in_front(c_in, width, layer_depth)
        self.buffer, self.in_front, self.width = buffer, in_front, width

        # Set output to None for the Conv2dInPlaceModule for now. We
        # set it in the forward pass.
        output = None
        self.convolution = ConvRelu2dInPlaceModule(
            output, in_front, width, kernel_size=3, dilation=dilation
        )

    def forward(self, input):
        # Set output
        self.convolution.output = self.buffer.L.narrow(1, self.in_front, self.width)
        output = self.convolution(input)
        output = stitchLazy(output, self.buffer.L, self.buffer.G, self.in_front)
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
        self.reset_parameters()

    def forward(self, input):
        b, c_in, *size = input.shape
        tmp_size = input[0, 0, ...].numel()

        # Put channels in last place in input shape
        output = input.view(b, c_in, tmp_size)
        output = self.linear(output)
        output = output.view(b, self.c_out, *size)
        return output

    def reset_parameters(self):
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()


class MSDModule(nn.Module):
    def __init__(
        self, c_in, c_out, depth, width, dilations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ):
        """Create a msd module

        :param c_in: # of input channels
        :param c_out: # of output channels
        :param depth: # of layers
        :param width: # the width of the module
        :param dilations: `list(int)`

        A list of dilations to use. Default is ``[1, 2, ..., 10]``.  A
        good alternative is ``[1, 2, 4, 8]``. The dilations are
        repeated.

        :returns: an MSD module
        :rtype: MSDModule

        """
        super(MSDModule, self).__init__()
        #
        self.c_in = c_in
        self.c_out = c_out
        self.depth = depth
        self.width = width
        self.dilations = dilations

        buffer = StitchBuffer()
        self.buffer = buffer

        # The first layer copies input into the L stitch buffer
        stitch_layer = StitchCopyModule(buffer, 0)

        # Then we have `depth` number of hidden layers:
        self.hidden_layers = [
            MSDLayerModule(buffer, c_in, d, width, dilations[d % len(dilations)])
            for d in range(depth)
        ]

        # Initialize weights for hidden layers:
        for m in self.hidden_layers:
            init_convolution_weights(
                m.convolution.weight.data, c_in, c_out, width, depth
            )
            m.convolution.bias.data.zero_()

        in_front = units_in_front(c_in, width, depth)
        self.c_final = MSDFinalLayer(in_front, c_out)

        self.net = nn.Sequential(stitch_layer, *self.hidden_layers, self.c_final)

        self.net.cuda()

    def forward(self, input):
        self.init_buffers(input.data)
        return self.net(input)

    def init_buffers(self, input):
        batch_sz, c_in, *shape = input.shape

        assert c_in == self.c_in, "Unexpected number of input channels"

        # Ensure that the stitch buffer is the correct shape
        total_units = units_in_front(self.c_in, self.width, self.depth)
        new_shape = (batch_sz, total_units, *shape)

        self.buffer.like_(input, new_shape)
