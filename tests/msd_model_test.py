import torch as t
from msd_pytorch.msd_model import MSDModel
import unittest


class TestMSDModel(unittest.TestCase):
    def test_parameters_change(self):
        # Make sure that learning updates all parameters.
        t.manual_seed(1)        # make test repeatable

        for conv3d in [False, True]:
            c_in, c_out, depth, width = 1, 1, 11, 1
            model = MSDModel(c_in, c_out, depth, width, 'L1', 'MSD', conv3d)
            shape = (11, 11, 11) if conv3d else (11, 11)
            input = t.randn(c_in, *shape)    # intentionally leave out
            target = t.randn(c_out, *shape)  # the batch size
            model.set_input(input)
            model.set_target(target)
            ps0 = [p.data.clone() for p in list(model.net.parameters())]
            for i in range(10):
                model.learn(input, input)
            ps1 = [p.data.clone() for p in list(model.net.parameters())]

            for p0, p1 in zip(ps0, ps1):
                self.assertTrue(t.sum(t.abs(p0 - p1)) > 0)

            if not conv3d:
                model.save_heatmap('test.png')


if __name__ == '__main__':
    unittest.main()
