from msd_pytorch.msd_block import MSDModule2d
from torch.autograd import Variable
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim


def scaling_module(c_in, c_out, *, conv3d=False):
    """Make a Module that normalizes the input data.

    This part of the network can be used to renormalize the input
    data. Its parameters are

    * saved when the network is saved;
    * not updated by the gradient descent solvers.

    :param c_in: The number of input channels.
    :param c_out: The number of output channels.
    :param conv3d: Indicates that the input data is 3D instead of 2D.
    :returns: A scaling module.
    :rtype: torch.nn.ConvNd

    """
    if conv3d:
        c = nn.Conv3d(c_in, c_out, 1)
    else:
        c = nn.Conv2d(c_in, c_out, 1)
    c.bias.requires_grad = False
    c.bias.data.zero_()
    c.weight.requires_grad = False
    c.weight.data.fill_(1)
    return c


class MSDModel:
    """Base class for MSD models.

    This class provides methods for

    * training the network
    * calculating validation scores
    * loading and saving the network parameters to disk.
    * computing normalization for input and target data.

    .. note::
        Do not initialize MSDModel directly. Use
        :class:`~msd_pytorch.msd_segmentation_model.MSDSegmentationModel` or
        :class:`~msd_pytorch.msd_regression_model.MSDRegressionModel` instead.

    """

    def __init__(
        self, c_in, c_out, depth, width, dilations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ):
        """Create a new MSDModel base class.

        .. note::
            Do not initialize MSDModel directly. Use
            :class:`~msd_pytorch.msd_segmentation_model.MSDSegmentationModel` or
            :class:`~msd_pytorch.msd_regression_model.MSDRegressionModel` instead.


        :param c_in: The number of input channels.
        :param c_out: The number of output channels.
        :param depth: The depth of the MSD network.
        :param width: The width of the MSD network.
        :param dilations: `list(int)`

        A list of dilations to use. Default is ``[1, 2, ..., 10]``.  A
        good alternative is ``[1, 2, 4, 8]``. The dilations are
        repeated when there are more layers than supplied dilations.

        :returns:
        :rtype:

        """
        self.c_in, self.c_out = c_in, c_out
        self.depth, self.width = depth, width
        self.dilations = dilations

        # This part of the network can be used to renormalize the
        # input and output data. Its parameters are saved when the
        # network is saved.
        self.scale_in = scaling_module(c_in, c_in)
        self.scale_out = scaling_module(c_out, c_out)
        self.msd = MSDModule2d(c_in, c_out, depth, width, dilations)

        # It is the task of any subclass to initialize `self.net` and
        # call `init_optimizer` to set the trainable parameters.

    def init_optimizer(self, trainable_net):
        self.optimizer = optim.Adam(trainable_net.parameters())

    def set_normalization(self, dataloader):
        """Normalize input and target data.

        This function goes through all the training data to compute
        the mean and std of the training data.

        It modifies the network so that all future invocations of the
        network first normalize input data and target data to have
        mean zero and a standard deviation of one.

        These modified parameters are not updated after this step and
        are stored in the network, so that they are not lost when the
        network is saved to and loaded from disk.

        Normalizing in this way makes training more stable.

        :param dataloader: The dataloader associated to the training data.
        :returns:
        :rtype:

        """
        mean_in = square_in = mean_out = square_out = 0

        for (data_in, data_out) in dataloader:
            mean_in += data_in.mean()
            mean_out += data_out.mean()
            square_in += data_in.pow(2).mean()
            square_out += data_out.pow(2).mean()

        mean_in /= len(dataloader)
        mean_out /= len(dataloader)
        square_in /= len(dataloader)
        square_out /= len(dataloader)

        std_in = np.sqrt(square_in - mean_in ** 2)
        std_out = np.sqrt(square_out - mean_out ** 2)

        # The input data should be roughly normally distributed after
        # passing through scale_in. Note that the input is first
        # scaled and then recentered.
        self.scale_in.weight.data.fill_(1 / std_in)
        self.scale_in.bias.data.fill_(-mean_in / std_in)
        # The scale_out layer should rather 'denormalize' the network
        # output.
        self.scale_out.weight.data.fill_(std_out)
        self.scale_out.bias.data.fill_(mean_out)

    def set_input(self, data):
        """Set input data.

        :param data: `torch.Tensor`

        A ``BxCxHxW``-dimensional torch input tensor.

        :returns:
        :rtype:

        """
        assert self.c_in == data.shape[1], "Wrong number of input channels"
        self.input = Variable(data.cuda())

    def set_target(self, data):
        """Set target data.

        :param data: `torch.Tensor`

        A ``BxCxHxW``-dimensional torch target tensor.

        :returns:
        :rtype:

        """
        assert self.c_out == data.shape[1], "Wrong number of output channels"
        self.target = Variable(data.cuda())

    def forward(self, input=None, target=None):
        """Calculate the loss for a single input-target pair.

        Both ``input`` and ``target`` are optional. If one of these
        parameters is not set, a previous value of these parameters is
        used.

        :param input: `torch.Tensor`

        A ``BxCxHxW``-dimensional torch input tensor.

        :param target: `torch.Tensor`

        A ``BxCxHxW``-dimensional torch input tensor.

        :returns: The loss on target
        :rtype:

        """
        if input is not None:
            self.set_input(input)
        if target is not None:
            self.set_target(target)

        self.output = None
        self.loss = None
        self.output = self.net(self.input)
        self.loss = self.criterion(self.output, self.target)

        return self.loss.item()

    def learn(self, input=None, target=None):
        """Train on a single input-target pair.

        :param input: `torch.Tensor`

        A ``BxCxHxW``-dimensional torch input tensor.

        :param target: `torch.Tensor`

        A ``BxCxHxW``-dimensional torch input tensor.

        """
        loss = self.forward(input, target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        return loss

    def train(self, dataloader, num_epochs):
        """Train on a dataset.

        Trains the network for ``num_epochs`` epochs on the dataset
        supplied by ``dataloader``.

        :param dataloader: A dataloader for a dataset to train on.
        :param num_epochs: The number of epochs to train for.
        :returns:
        :rtype:

        """
        for epoch in range(num_epochs):
            for (input, target) in dataloader:
                self.learn(input, target)

    def validate(self, dataloader):
        """Calculate validation score for dataset.

        Calculates the mean loss per ``(input, target)`` pair in
        ``dataloader``. The loss function that is used depends on
        whether the model is doing regression or segmentation.

        :param dataloader: A dataloader for a dataset to calculate the loss on.
        :returns:
        :rtype:

        """
        validation_loss = 0
        for (input, target) in dataloader:
            validation_loss += self.forward(input, target)

        return validation_loss / len(dataloader)

    def apply(self, dataloader):
        """Calculate test score for dataset.

        Calculates the mean loss per ``(input, target)`` pair in
        ``dataloader``. The loss function that is used depends on
        whether the model is doing regression or segmentation.

        :param dataloader: A dataloader for a dataset to calculate the loss on.
        :returns:
        :rtype:

        """
        test_loss = 0
        for (input, target) in dataloader:
            test_loss += self.forward(input, target)

        return test_loss / len(dataloader)

    def print(self):
        """Print the network.
        """
        print(self.net)

    def get_loss(self):
        """Get the mean loss of the last forward calculation.

        Gets the mean loss of the last ``(input, target)`` pair. The
        loss function that is used depends on whether the model is
        doing regression or segmentation.

        :returns: The loss.
        :rtype: float

        """
        return self.loss.data.mean().item()

    def get_output(self):
        """Get the output of the network.

        .. note:: The output is only defined after a call to
           :func:`~forward`, :func:`~learn`, :func:`~train`,
           :func:`~validate`. If none of these methods has been
           called, ``None`` is returned.

        :returns: A torch tensor containing the output of the network or ``None``.
        :rtype: `torch.Tensor` or `NoneType`

        """
        return self.output

    def save(self, path, epoch):
        """Save network to disk.

        :param path: A filesystem path where the network parameters are stored.
        :param epoch: The number of epochs the network has trained for. This is useful for reloading!
        :returns: Nothing
        :rtype:

        """
        state = {
            "epoch": int(epoch),
            "state_dict": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        t.save(state, path)

    def load(self, path):
        """Load network parameters from disk.

        :param path: The filesystem path where the network parameters are stored.
        :returns: the number of epochs the network has trained for.
        :rtype: int

        """
        state = t.load(path)

        self.net.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.net.cuda()

        epoch = state["epoch"]

        return epoch
