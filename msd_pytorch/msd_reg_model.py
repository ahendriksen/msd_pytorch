from msd_pytorch.msd_model import (MSDModel)
from msd_pytorch.msd_module import (msd_dilation, one_dilation)
from sacred import Ingredient
import torch.nn as nn

msd_ingredient = Ingredient('msd')

loss_functions = {'L1': nn.L1Loss(),
                  'L2': nn.MSELoss()}


dilation_functions = {
    'MSD': msd_dilation,
    'one': one_dilation
}


@msd_ingredient.config
def cfg():
    c_in = 1                    # noqa
    c_out = 1                   # noqa
    depth = 30                  # noqa
    width = 1                   # noqa
    loss_function = "L1"        # noqa
    dilation = 'MSD'            # noqa
    reflect = True              # noqa
    conv3d = False              # noqa


class MSDRegressionModel(MSDModel):
    @msd_ingredient.capture()
    def __init__(self, c_in, c_out, depth, width, loss_function,
                 dilation, reflect, conv3d):

        super().__init__(c_in, c_out, depth, width, dilation, reflect,
                         conv3d)

        self.loss_function = loss_function
        self.criterion = loss_functions[loss_function]
        assert(self.criterion is not None)

        # Define the whole network:
        self.net = nn.Sequential(self.scale_in, self.msd, self.scale_out)
        self.net.cuda()

        # Train only MSD parameters:
        self.init_optimizer(self.msd)
