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


@msd_ingredient.config
def cfg():
    c_in = 1
    c_out = 1
    depth = 30
    width = 1
    loss_function = "L1"
    dilation = 'MSD'
    save_dir = 'saved_networks'
    conv3d = False


class MSDModel():
    @msd_ingredient.capture()
    def __init__(self, c_in, c_out, depth, width, loss_function, dilation,
                 conv3d):
        self.c_in, self.c_out = c_in, c_out
        self.depth, self.width = depth, width

        self.loss_function = loss_function
        self.conv3d = conv3d
        self.criterion = loss_functions[loss_function]
        assert(self.criterion is not None)

        self.dilation = dilation
        dilation_function = dilation_functions[dilation]
        assert(dilation_function is not None)
        self.net = MSDModule(c_in, c_out, depth, width, dilation_function,
                             conv3d=conv3d)

        self.optimizer = optim.Adam(self.net.parameters())

    def unsqueeze(self, data):
        assert len(data.shape) >= 2, "Must supply at least 2-dimensional data"

        assert len(data.shape) >= 3 or not self.conv3d, \
            "Must supply at least 3-dimensional data"

        desired_shape_len = 5 if self.conv3d else 4

        while len(data.shape) < desired_shape_len:
            data = data.unsqueeze(0)

        for shape_dim in data.shape[2:]:
            assert shape_dim > 10, \
                "Size too small: convolutions break because there is not enough\
                padding"

        return data

    def set_input(self, data):
        data = self.unsqueeze(data)
        assert self.c_in == data.shape[1], "Wrong number of input channels"

        self.input = Variable(data.cuda())

    def set_target(self, data):
        data = self.unsqueeze(data)
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

    def grow(self, add_depth=1):
        new = MSDModel(self.depth + add_depth, self.loss_function,
                       self.dilation, self.conv3d)
        new.net = self.net.grow(add_depth)
        return new

    def print(self):
        print(self.net)

    def get_loss(self):
        return self.loss.data.sum()

    def get_output(self):
        return self.output

    def save_network(self, save_dir, name, label):
        filename = "{}_{}.pytorch".format(name, label)
        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        # Clear the L and G buffers before saving:
        self.net.clear_buffers()

        t.save(self.net.state_dict(), save_path)
        return save_path

    def load_network(self, save_dir, name, label):
        filename = "{}_{}.pytorch".format(name, label)
        save_path = os.path.join(save_dir, filename)
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
