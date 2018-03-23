import torch as t
from msd_pytorch.msd_reg_model import MSDRegressionModel
import unittest


class TestMSDRegressionModel(unittest.TestCase):
    def test_regression_model(self):
        # Make sure that learning updates all parameters.
        t.manual_seed(1)        # make test repeatable

        for conv3d in [False, True]:
            c_in, c_out, depth, width = 1, 1, 11, 1
            model = MSDRegressionModel(c_in, c_out, depth, width,
                                       'L1', 'MSD', conv3d)
            shape = (11, 11, 11) if conv3d else (11, 11)
            input = t.randn(1, c_in, *shape)    # batch size is one.
            target = t.randn(1, c_out, *shape)  #
            model.set_input(input)
            model.set_target(target)
            for i in range(10):
                model.learn(input, input)


if __name__ == '__main__':
    unittest.main()
