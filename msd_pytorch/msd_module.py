import torch
from math import sqrt
import numpy as np
from msd_pytorch.msd_block import MSDBlock2d


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


class MSDFinalLayer(torch.nn.Module):
    """Documentation for MSDFinalLayer

    Implements the final 1x1 multiplication and bias addition for all
    intermediate layers to get to the output layer.

    Initializes the weight and bias to zero.
    """

    def __init__(self, c_in, c_out):
        super(MSDFinalLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = torch.nn.Conv1d(c_in, c_out, 1)
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


class MSDModule(torch.nn.Module):
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
        repeated when the depth of the module exceeds the length of
        the list.

        :returns: an MSD module
        :rtype: MSDModule

        """
        super(MSDModule, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.depth = depth
        self.width = width
        self.dilations = [dilations[i % len(dilations)] for i in range(depth)]

        if depth < 1:
            raise ValueError(f"Depth must be at least 1. Got: {depth}.")
        if width < 1:
            raise ValueError(f"Width must be at least 1. Got: {width}.")

        self.msd_block = MSDBlock2d(self.c_in, self.dilations, self.width)
        self.final_layer = MSDFinalLayer(c_in=c_in + width * depth, c_out=c_out)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights for hidden layers:
        for w in self.msd_block.weights:
            init_convolution_weights(
                w.data, self.c_in, self.c_out, self.width, self.depth
            )

        self.msd_block.bias.data.zero_()
        self.final_layer.reset_parameters()

    def forward(self, input):
        output = self.msd_block(input)
        output = self.final_layer(output)
        return output
