import torch
from . import conv2d
from . import conv3d
import numpy as np


class MSDBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dilations, bias, *weights):
        depth = len(dilations)
        assert depth == len(weights), "number of weights does not match depth"

        if input.dim() == 4:
            conv_relu_forward = conv2d.conv2d_relu_forward
        elif input.dim() == 5:
            conv_relu_forward = conv3d.conv3d_relu_forward
        else:
            raise ValueError(
                "Input to MSDBlock must be 2D or 3D"
            )

        num_out_channels = sum(w.shape[0] for w in weights)
        assert (
            len(bias) == num_out_channels
        ), "number of biases does not match number of output channels from weights"

        ctx.dilations = dilations
        ctx.depth = depth

        result = input.new_empty(
            input.shape[0], input.shape[1] + num_out_channels, *input.shape[2:]
        )

        # Copy input into result buffer
        result[:, : input.shape[1]] = input

        result_start = input.shape[1]
        bias_start = 0

        for i in range(depth):
            # Extract variables
            sub_input = result[:, :result_start]
            sub_weight = weights[i]
            blocksize = sub_weight.shape[0]
            sub_bias = bias[bias_start : bias_start + blocksize]
            sub_result = result[:, result_start : result_start + blocksize]
            dilation = ctx.dilations[i]

            # Compute convolution. conv_relu_forward computes the
            # convolution and relu in one pass and stores the
            # output in sub_result.
            conv_relu_forward(
                sub_input, sub_weight, sub_bias, sub_result, dilation
            )

            # Update steps etc
            result_start += blocksize
            bias_start += blocksize

        ctx.save_for_backward(bias, result, *weights)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        bias, result, *weights = ctx.saved_tensors
        if grad_output.dim() == 4:
            conv_relu_backward_x = conv2d.conv2d_relu_backward_x
            conv_relu_backward_k = conv2d.conv2d_relu_backward_k
            conv_relu_backward_bias = conv2d.conv2d_relu_backward_bias
        elif grad_output.dim() == 5:
            conv_relu_backward_x = conv3d.conv3d_relu_backward_x
            conv_relu_backward_k = conv3d.conv3d_relu_backward_k
            conv_relu_backward_bias = conv3d.conv3d_relu_backward_bias
        else:
            assert False, "Mismatched tensor dimensions."


        depth = ctx.depth

        grad_bias = torch.zeros_like(bias)
        # XXX: Could we just overwrite grad_output instead of clone?
        gradients = grad_output.clone()
        grad_weights = []

        result_end = result.shape[1]
        bias_end = len(bias)

        for i in range(depth):
            idx = depth - 1 - i
            # Get subsets
            sub_weight = weights[idx]
            blocksize = sub_weight.shape[0]
            result_start = result_end - blocksize
            bias_start = bias_end - blocksize

            sub_grad_output = gradients[:, result_start:result_end]
            sub_grad_input = gradients[:, :result_start]
            sub_result = result[:, result_start:result_end]
            sub_input = result[:, :result_start]

            dilation = ctx.dilations[idx]

            # Gradient w.r.t. input: conv_relu_backward_x computes the
            # gradient wrt sub_input and adds the gradient to
            # sub_grad_input.
            conv_relu_backward_x(
                sub_result, sub_grad_output, sub_weight, sub_grad_input, dilation
            )

            # Gradient w.r.t weights
            IDX_WEIGHT_START = 3            # The first weight has index 3 in the forward pass.
            if ctx.needs_input_grad[i + IDX_WEIGHT_START]:
                sub_grad_weight = torch.zeros_like(sub_weight)
                conv_relu_backward_k(
                    sub_result, sub_grad_output, sub_input, sub_grad_weight, dilation
                )
                grad_weights.insert(0, sub_grad_weight)
            else:
                grad_weights.insert(0, None)
            # Gradient of Bias
            if ctx.needs_input_grad[2]:
                sub_grad_bias = grad_bias[bias_start:bias_end]
                conv_relu_backward_bias(
                    sub_result, sub_grad_output, sub_grad_bias
                )

            # Update positions etc
            result_end -= blocksize
            bias_end -= blocksize

        grad_input = gradients[:, : weights[0].shape[1]]

        return (grad_input, None, grad_bias, *grad_weights)


msd_block = MSDBlockFunction.apply


class MSDBlock(torch.nn.Module):
    def __init__(self, in_channels, dilations, width=1, ndim=2):
        """Multi-scale dense block

        Parameters
        ----------
        in_channels : int
            Number of input channels
        dilations : tuple of int
            Dilation for each convolution-block
        width : int
            Number of channels per convolution.

        Notes
        -----
        The number of output channels is in_channels + depth * width
        """
        super().__init__()
        if ndim == 2:
            self.kernel_size = (3, 3)
        elif ndim == 3:
            self.kernel_size = (3, 3, 3)
        else:
            raise ValueError("ndim must be 2 or 3")

        self.width = width
        self.dilations = dilations

        depth = len(self.dilations)

        self.bias = torch.nn.Parameter(torch.Tensor(depth * width))

        self.weights = []
        for i in range(depth):
            n_in = in_channels + width * i

            weight = torch.nn.Parameter(torch.Tensor(width, n_in, *self.kernel_size))

            self.register_parameter("weight{}".format(i), weight)
            self.weights.append(weight)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            torch.nn.init.kaiming_uniform_(weight, a=np.sqrt(5))

        if self.bias is not None:
            # TODO: improve
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # We need to obtain weights in this way, because self.weights
        # may become obsolete when used in multi-gpu settings when the
        # weights are automatically transferred (by, e.g.,
        # torch.nn.DataParallel). In that case, self.weights may
        # continue to point to the weight parameters on the original
        # device, even when the weight parameters have been
        # transferred to a different gpu.
        #
        # To be compatible with torch.nn.utils.prune, we obtain the
        # weights using attributes. Previously, we used
        # `self.parameters()`, but this returns the original
        # (unmasked) parameters.
        bias = self.bias
        weights = (getattr(self, "weight{}".format(i)) for i in range(len(self.weights)))

        return msd_block(input, self.dilations, bias, *weights)


class MSDBlock2d(MSDBlock):
    def __init__(self, in_channels, dilations, width=1):
        """Multi-scale dense block

        Parameters
        ----------
        in_channels : int
            Number of input channels
        dilations : tuple of int
            Dilation for each convolution-block
        width : int
            Number of channels per convolution.

        Notes
        -----
        The number of output channels is in_channels + depth * width
        """
        super().__init__(in_channels, dilations, width=width, ndim=2)


class MSDBlock3d(MSDBlock):
    def __init__(self, in_channels, dilations, width=1):
        """Multi-scale dense block

        Parameters
        ----------
        in_channels : int
            Number of input channels
        dilations : tuple of int
            Dilation for each convolution-block
        width : int
            Number of channels per convolution.

        Notes
        -----
        The number of output channels is in_channels + depth * width
        """
        super().__init__(in_channels, dilations, width=width, ndim=3)
