from msd_pytorch.msd_module import (MSDModule, msd_dilation, one_dilation)
from sacred import Ingredient
from torch.autograd import Variable
import numpy as np
import os
import os.path
import torch as t
import torch.nn as nn
import torch.optim as optim


msd_ingredient = Ingredient('msd')

dilation_functions = {
    'MSD': msd_dilation,
    'one': one_dilation
}


def scaling_module(c_in, c_out, conv3d=False):
    # This part of the network can be used to renormalize the
    # input data. Its parameters are saved when the network is
    # saved.
    if conv3d:
        c = nn.Conv3d(c_in, c_out, 1)
    else:
        c = nn.Conv2d(c_in, c_out, 1)
    c.bias.requires_grad = False
    c.bias.data.zero_()
    c.weight.requires_grad = False
    c.weight.data.fill_(1)
    return c


class MSDModel():
    def __init__(self, c_in, c_out, depth, width, dilation, reflect,
                 conv3d):
        self.c_in, self.c_out = c_in, c_out
        self.depth, self.width = depth, width

        self.conv3d = conv3d
        self.dilation = dilation

        dilation_function = dilation_functions[dilation]
        assert(dilation_function is not None)
        self.reflect = reflect

        # This part of the network can be used to renormalize the
        # input and output data. Its parameters are saved when the
        # network is saved.
        self.scale_in = scaling_module(c_in, c_in, conv3d)
        self.scale_out = scaling_module(c_out, c_out, conv3d)

        self.msd = MSDModule(c_in, c_out, depth, width,
                             dilation_function, reflect,
                             conv3d=conv3d)

    def init_optimizer(self, trainable_net):
        self.optimizer = optim.Adam(trainable_net.parameters())

    def set_normalization(self, dataloader):
        """Normalize input and target data.

           This function goes through all the training data to compute
           the mean and std of the training data. It modifies the
           network so that all future invocations of the network first
           normalize input data and output data. The normalization
           parameters are saved.

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
        self.scale_in.bias.data.fill_(- mean_in / std_in)
        # The scale_out layer should rather 'denormalize' the network
        # output.
        self.scale_out.weight.data.fill_(std_out)
        self.scale_out.bias.data.fill_(mean_out)

    def set_input(self, data):
        assert self.c_in == data.shape[1], "Wrong number of input channels"
        self.input = Variable(data.cuda())

    def set_target(self, data):
        assert self.c_out == data.shape[1], "Wrong number of output channels"
        self.target = Variable(data.cuda())

    def forward(self, input=None, target=None):
        if input is not None:
            self.set_input(input)
        if target is not None:
            self.set_target(target)

        self.output = self.net(self.input)
        self.loss = self.criterion(self.output, self.target)

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
