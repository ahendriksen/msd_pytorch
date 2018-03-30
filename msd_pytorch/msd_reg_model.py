import torch.nn as nn
import torch as t
import torch.optim as optim
import torchvision.utils as tvu
from sacred import Ingredient
from torch.autograd import Variable
from msd_pytorch.msd_module import (MSDModule, msd_dilation, one_dilation)
import os
import os.path

msd_ingredient = Ingredient('msd')

loss_functions = {'L1': nn.L1Loss(),
                  'L2': nn.MSELoss()}


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


class MSDRegressionModel():
    @msd_ingredient.capture()
    def __init__(self, c_in, c_out, depth, width, loss_function,
                 dilation, reflect, conv3d):
        self.c_in, self.c_out = c_in, c_out
        self.depth, self.width = depth, width

        self.loss_function = loss_function
        self.conv3d = conv3d
        self.criterion = loss_functions[loss_function]
        assert(self.criterion is not None)

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

        # Train only MSD parameters:
        net_trained = nn.Sequential(self.msd, nn.ReLU())
        self.optimizer = optim.Adam(net_trained.parameters())

        # Define the whole network:
        self.net = nn.Sequential(self.scale_in, net_trained, self.scale_out)
        self.net.cuda()

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

        std_in = square_in - mean_in ** 2
        std_out = square_out - mean_out ** 2

        # The input data should be roughly normally distributed after
        # passing through net_fixed.
        self.scale_in.bias.data.fill_(- mean_in)
        self.scale_in.weight.data.fill_(1 / std_in)
        # The scale_out layer should rather 'denormalize' the network
        # output.
        self.scale_out.bias.data.fill_(mean_in)
        self.scale_out.weight.data.fill_(std_out)

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

    def load_network(self, save_dir, name, label):
        save_path = self.get_network_path(save_dir, name, label)
        self.net.load_state_dict(t.load(save_path))
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
        L = self.msd.L.clone()
        C = self.msd.c_final.weight.data

        for i, c in enumerate(C.squeeze().tolist()):
            L[:, i, :, :].mul_(c)

        tvu.save_image(L[:, 1:, :, :].transpose(0, 1),
                       filename,
                       nrow=10)

    def save_g(self, filename):
        tvu.save_image(self.msd.G[:, 1:, :, :].transpose(0, 1),
                       filename,
                       nrow=10)
