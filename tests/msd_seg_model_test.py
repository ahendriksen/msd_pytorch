import torch as t
from torch.autograd import (Variable)
from torch.utils.data import (TensorDataset, DataLoader)
from msd_pytorch.msd_seg_model import MSDSegmentationModel
import unittest


class TestMSDSegmentationModel(unittest.TestCase):
    def test_segmentation_model(self):
        # Make sure that learning updates all parameters.
        t.manual_seed(1)        # make test repeatable


        for conv3d in [True, False]:
            c_in, num_labels, depth, width = 1, 2, 11, 2
            model = MSDSegmentationModel(c_in, num_labels, depth, width,
                                         'MSD', reflect=False,
                                         conv3d=conv3d)
            shape = (11, 11, 11) if conv3d else (11, 11)
            input = t.randn(1, c_in, *shape)   # batch size is one.
            target = t.rand(1, 1, *shape).bernoulli()
            model.set_input(input)
            model.set_target(target)
            for i in range(10):
                model.learn(input, target)


    def test_normalization(self):
        means = [-1, 0, 1]
        stds = [1, 10, 100]

        mean = -1
        std = 10

        for mean in means:
            for std in stds:
                sample_size, num_channels, shape = 1, 1, (1000, 1000)
                num_labels = 2
                t_in = mean + t.randn(sample_size, num_channels, *shape) * std
                t_out = t.bernoulli(t.rand(sample_size, *shape))

                ds = TensorDataset(t_in, t_out)

                dl = DataLoader(ds, sample_size)

                model = MSDSegmentationModel(num_channels, num_labels, 0, 1,
                                             'MSD', False, conv3d=False)
                model.set_normalization(dl)

                (input, target), *_=  dl
                output = model.forward(input, target)

                # Check input layer scaling
                l0 = model.scale_in(Variable(input).cuda())
                self.assertAlmostEqual(l0.data.mean(), 0, delta=1e-2)
                self.assertAlmostEqual(l0.data.std(), 1, delta=1e-2)

if __name__ == '__main__':
    unittest.main()
