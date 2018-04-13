import torch as t
from torch.autograd import (Variable)
from torch.utils.data import (TensorDataset, DataLoader)
from msd_pytorch.msd_reg_model import MSDRegressionModel
import unittest


class TestMSDRegressionModel(unittest.TestCase):
    def test_regression_model(self):
        # Make sure that learning updates all parameters.
        t.manual_seed(1)        # make test repeatable

        for conv3d in [False, True]:
            c_in, c_out, depth, width = 1, 1, 11, 1
            model = MSDRegressionModel(c_in, c_out, depth, width,
                                       'L1', 'MSD', False, conv3d)
            shape = (11, 11, 11) if conv3d else (11, 11)
            input = t.randn(1, c_in, *shape)    # batch size is one.
            target = t.randn(1, c_out, *shape)  #
            model.set_input(input)
            model.set_target(target)
            for i in range(10):
                model.learn(input, input)

    def test_normalization(self):
        means = [-1, 0, 1]
        stds = [1, 10, 100]

        mean = -1
        std = 10

        for mean in means:
            for std in stds:
                sample_size, num_channels, shape = 1, 1, (1000, 1000)

                t_in = mean + t.randn(sample_size, num_channels, *shape) * std
                t_out = mean + t.randn(sample_size, num_channels, *shape) * std

                ds = TensorDataset(t_in, t_out)

                dl = DataLoader(ds, sample_size)

                model = MSDRegressionModel(num_channels, num_channels, 0, 1,
                                           'L1', 'MSD', False, conv3d=False)
                model.set_normalization(dl)

                (input, target), *_=  dl
                output = model.forward(input, target)

                # Check input layer scaling
                l0 = model.scale_in(Variable(input).cuda())
                self.assertAlmostEqual(l0.data.mean(), 0, delta=1e-2)
                self.assertAlmostEqual(l0.data.std(), 1, delta=1e-2)

                # Check output layer scaling
                l1 = model.scale_out(l0)
                self.assertAlmostEqual(l1.data.mean(), target.mean(), delta=1e-2)
                self.assertAlmostEqual(l1.data.std(), target.std(), delta=1e-2)

if __name__ == '__main__':
    unittest.main()
