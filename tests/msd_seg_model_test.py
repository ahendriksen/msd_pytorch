import torch as t
from msd_pytorch.msd_seg_model import MSDSegmentationModel
import unittest


class TestMSDSegmentationModel(unittest.TestCase):
    def test_segmentation_model(self):
        # Make sure that learning updates all parameters.
        t.manual_seed(1)        # make test repeatable


        # 3d convolution is not yet supported:
        for conv3d in [False]:
            c_in, num_labels, depth, width = 1, 2, 11, 2
            model = MSDSegmentationModel(c_in, num_labels, depth, width,
                                         'MSD', conv3d)
            shape = (11, 11, 11) if conv3d else (11, 11)
            input = t.randn(1, c_in, *shape)   # batch size is one.
            target = t.rand(1, 1, *shape).bernoulli()
            model.set_input(input)
            model.set_target(target)
            for i in range(10):
                model.learn(input, target)

if __name__ == '__main__':
    unittest.main()
