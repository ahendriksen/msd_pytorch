from msd_pytorch.msd_module import (MSDModule, msd_dilation, one_dilation)
from sacred import Ingredient
from torch.autograd import Variable
import numpy as np
import os
import os.path
import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tvu

msd_ingredient = Ingredient('msd')

loss_functions = {'L1': nn.L1Loss(),
                  'L2': nn.MSELoss()}


dilation_functions = {
    'MSD': msd_dilation,
    'one': one_dilation
}


@msd_ingredient.config
def cfg():
    c_in = 1
    c_out = 1
    depth = 30
    width = 1
    loss_function = "L1"
    dilation = 'MSD'
    reflect = True
    save_dir = 'saved_networks'
    conv3d = False


class MSDSegmentationModel():
    @msd_ingredient.capture()
    def __init__(self, c_in, num_labels, depth, width, dilation,
                 reflect, conv3d):
        self.c_in, self.num_labels = c_in, num_labels
        self.depth, self.width = depth, width

        assert not conv3d, "3d segmentation is not yet supported"

        # TODO: implement NLLLoss3d.
        # LogSoftmax + NLLLoss is equivalent to a Softmax activation
        # with Cross-entropy loss.
        self.criterion = nn.NLLLoss2d()

        self.dilation = dilation
        dilation_function = dilation_functions[dilation]
        assert(dilation_function is not None)

        self.reflect = reflect

        # This part of the network can be used to renormalize the
        # input data. Its parameters are saved when the network is
        # saved.
        net_fixed = nn.Conv2d(c_in, c_in, 1)
        net_fixed.bias.requires_grad = False
        net_fixed.bias.data.zero_()
        net_fixed.weight.requires_grad = False
        net_fixed.weight.data.fill_(1)
        self.net_fixed = net_fixed

        # The rest of the network has parameters that are updated
        # during training.
        self.msd = MSDModule(c_in, num_labels, depth, width,
                             dilation_function, reflect)

        net_trained = nn.Sequential(
            self.msd,
            nn.Conv2d(num_labels, num_labels, 1),
            nn.LogSoftmax(dim=1))

        self.net = nn.Sequential(net_fixed,
                                 net_trained)
        self.net.cuda()
        self.optimizer = optim.Adam(net_trained.parameters())

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
        self.net_fixed.bias.data.fill_(- mean)
        self.net_fixed.weight.data.fill_(1 / std)

    def set_input(self, data):
        assert self.c_in == data.shape[1], "Wrong number of input channels"
        self.input = Variable(data.cuda())

    def set_target(self, data):
        # The class labels must be of long data type
        data = data.long()
        # The class labels must reside on the GPU
        data = data.cuda()
        self.target = Variable(data)

    def forward(self, input=None, target=None):
        if input is not None:
            self.set_input(input)
        if target is not None:
            self.set_target(target)

        self.output = self.net(self.input)
        # The NLLLoss does not accept a channel dimension. So we
        # squeeze the target.
        self.loss = self.criterion(self.output,
                                   self.target.squeeze(1))

    def learn(self, input=None, target=None):
        self.forward(input, target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for (input, target) in dataloader:
                self.learn(input, target)

    def validate(self, dataloader):
        validation_loss = 0
        for (input, target) in dataloader:
            self.forward(input, target)
            validation_loss += self.get_loss()

        return validation_loss / len(dataloader)

    def print(self):
        print(self.net)

    def get_loss(self):
        return self.loss.data.sum()

    def get_output(self):
        return self.output

    def get_network_path(self, save_dir, name, label):
        filename = "{}_{}.pytorch".format(name, label)
        save_path = os.path.join(save_dir, filename)
        return save_path

    def save_network(self, save_dir, name, label):
        save_path = self.get_network_path(save_dir, name, label)
        os.makedirs(save_dir, exist_ok=True)
        # Clear the L and G buffers before saving:
        self.msd.clear_buffers()

        t.save(self.net.state_dict(), save_path)
        return save_path

    def load_network(self, save_dir='.', name='MSD', label='0',
                     save_file=None):
        """Load network parameters from storage.

        :param save_dir: directory to save files in.
        :param name: name of the network.
        :param label: a label (such as current epoch) to add to the filename.
        :param save_file: a file path or stream-like object that overrides the default filename structure.
        :returns:
        :rtype:

        """
        if save_file is None:
            save_file = self.get_network_path(save_dir, name, label)
        self.net.load_state_dict(t.load(save_file))
        self.net.cuda()

    def save_output(self, filename):
        tvu.save_image(self.output.data, filename)

    def save_input(self, filename):
        tvu.save_image(self.input.data, filename)

    def save_target(self, filename):
        tvu.save_image(self.target.data, filename)

    def save_diff(self, filename):
        tvu.save_image(t.abs(self.target - self.output).data, filename)

    def save_heatmap(self, filename):
        ''' Make a heatmap of the absolute sum of the convolution kernels
        '''

        # heatmap = t.zeros(self.depth, self.depth)

        # conv_ws = [w for k, w in self.net.state_dict().items()
        #            if 'convolution.weight' in k]

        # for i, w in enumerate(conv_ws):
        #     for j in range(w.shape[1]):
        #         heatmap[j, i] = w[:, j, :, :].abs().sum()
        L = self.net.L.clone()
        C = self.net.c_final.weight.data

        for i, c in enumerate(C.squeeze().tolist()):
            L[:, i, :, :].mul_(c)

        tvu.save_image(L[:, 1:, :, :].transpose(0, 1),
                       filename,
                       nrow=10)

    def save_g(self, filename):
        tvu.save_image(self.net.G[:, 1:, :, :].transpose(0, 1),
                       filename,
                       nrow=10)
