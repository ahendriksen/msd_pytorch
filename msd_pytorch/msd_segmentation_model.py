from msd_pytorch.msd_model import MSDModel
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


class MSDSegmentationModel(MSDModel):
    """An MSD network for segmentation.

    This class provides helper methods for using the MSD network
    module for segmentation.

    Refer to the documentation of
    :class:`~msd_pytorch.msd_model.MSDModel` for more information on
    the helper methods and attributes.

    """

    def __init__(
        self,
        c_in,
        num_labels,
        depth,
        width,
        *,
        dilations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ):
        """Create a new MSD network for segmentation.

        :param c_in: The number of input channels.
        :param num_labels: The number of labels to divide the segmentation into.
        :param depth: The depth of the MSD network
        :param width: The width of the MSD network
        :param dilations: `list(int)`

        A list of dilations to use. Default is ``[1, 2, ..., 10]``.  A
        good alternative is ``[1, 2, 4, 8]``. The dilations are
        repeated when there are more layers than supplied dilations.

        :returns:
        :rtype:

        """
        self.num_labels = num_labels
        # Initialize msd network.
        super().__init__(c_in, num_labels, depth, width, dilations)

        self.criterion = nn.NLLLoss()

        # Initialize network
        net_trained = nn.Sequential(self.msd, nn.LogSoftmax(dim=1))
        self.net = nn.Sequential(self.scale_in, net_trained)
        self.net.cuda()

        # Train all parameters apart from self.scale_in.
        self.init_optimizer(net_trained)

    def set_normalization(self, dataloader):
        """Normalize input data.

        This function goes through all the training data to compute
        the mean and std of the training data. It modifies the network
        so that all future invocations of the network first normalize
        input data. The normalization parameters are saved.

        :param dataloader: The dataloader associated to the training data.
        :returns:
        :rtype:

        """
        mean = 0
        square = 0
        for (data_in, _) in dataloader:
            mean += data_in.mean()
            square += data_in.pow(2).mean()

        mean /= len(dataloader)
        square /= len(dataloader)
        std = np.sqrt(square - mean ** 2)

        # The input data should be roughly normally distributed after
        # passing through net_fixed.
        self.scale_in.bias.data.fill_(-mean / std)
        self.scale_in.weight.data.fill_(1 / std)

    def set_target(self, target):
        # The class labels must be of long data type
        target = target.long()

        min, max = target.min(), target.max()
        if min < 0 or self.num_labels <= max:
            raise ValueError(
                f"Target invalid: expected values in range {[0, self.num_labels - 1]}, but got {[min, max]}"
            )
        # The NLLLoss does not accept a channel dimension. So we
        # squeeze the target.
        target = target.squeeze(1)
        # The class labels must reside on the GPU
        target = target.cuda()
        self.target = Variable(target)
