from msd_pytorch.msd_model import (MSDModel)
from msd_pytorch.msd_module import (msd_dilation, one_dilation)
from sacred import Ingredient
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


msd_ingredient = Ingredient('msd')

dilation_functions = {
    'MSD': msd_dilation,
    'one': one_dilation
}


@msd_ingredient.config
def cfg():
    c_in = 1
    num_labels = 2
    depth = 30
    width = 1
    dilation = 'MSD'
    reflect = True
    conv3d = False


class MSDSegmentationModel(MSDModel):
    @msd_ingredient.capture()
    def __init__(self, c_in, num_labels, depth, width, dilation,
                 reflect, conv3d):
        # We don't support 3d segmentation yet.
        assert not conv3d, "3d segmentation is not yet supported"
        # Allow supplying a list of labels instead of just the number
        # of labels.
        if isinstance(num_labels, list):
            c_out = len(num_labels)
            self.labels = num_labels
        else:
            c_out = num_labels

        # Initialize msd network.
        super().__init__(c_in, c_out, depth, width, dilation,
                         reflect, conv3d)

        # TODO: implement NLLLoss3d.
        # LogSoftmax + NLLLoss is equivalent to a Softmax activation
        # with Cross-entropy loss.
        self.criterion = nn.NLLLoss2d()

        # Initialize network
        net_trained = nn.Sequential(
            self.msd,
            nn.LogSoftmax(dim=1))
        self.net = nn.Sequential(self.scale_in,
                                 net_trained)
        self.net.cuda()

        # Train all parameters apart from self.scale_in.
        self.init_optimizer(net_trained)

    def set_normalization(self, dataloader):
        """Normalize input data.

           This function goes through all the training data to compute
           the mean and std of the training data. It modifies the
           network so that all future invocations of the network first
           normalize input data. The normalization parameters are saved.

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
        self.scale_in.bias.data.fill_(- mean)
        self.scale_in.weight.data.fill_(1 / std)

    def set_target(self, data):
        # relabel if necessary:
        if self.labels:
            for i, label in enumerate(self.labels):
                data[data == label] = i

        # The class labels must be of long data type
        data = data.long()
        # The NLLLoss does not accept a channel dimension. So we
        # squeeze the target.
        data = data.squeeze(1)
        # The class labels must reside on the GPU
        data = data.cuda()
        self.target = Variable(data)
