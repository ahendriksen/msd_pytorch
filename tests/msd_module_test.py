import torch.nn as nn
import torch as t
import torch.optim as optim
from torch.autograd import (Variable)
from msd_pytorch.msd_module import (MSDModule, msd_dilation)
import unittest


class MSDModuleTest(unittest.TestCase):

    def test_clear_buffers(self):
        t.manual_seed(2)

        size = (30, 30)
        x = Variable(t.randn(1, 1, *size)).cuda()
        target = Variable(t.randn(1, 1, *size)).cuda()

        net = MSDModule(10, msd_dilation, False)
        y1 = net(x)
        criterion = nn.L1Loss()
        loss = criterion(y1, target)
        optimizer = optim.Adam(net.parameters())
        optimizer.zero_grad()
        loss.backward()

        net_new = net.grow(1)
        net_new.clear_buffers()
        y2 = net(x)
        self.assertAlmostEqual(0, (y1 - y2).abs().data.sum())

    def test_3d(self):
        batch_sz = 1
        in_channels = 1
        size = (20,) * 3
        x = t.randn(batch_sz, in_channels, *size).cuda()

        net = MSDModule(10, msd_dilation, True)

        output = net(Variable(x))

        self.assertNotAlmostEqual(0, output.data.abs().sum())

    def test_parameters_change(self):
        t.manual_seed(1)

        size = (30, 30)
        for batch_sz in [1, 2, 5]:
            for depth in range(20):
                x = Variable(t.randn(batch_sz, 1, *size)).cuda()
                target = Variable(t.randn(batch_sz, 1, *size)).cuda()
                self.assertTrue(x.data.is_cuda)

                net = MSDModule(depth, msd_dilation, False)

                self.assertIsNotNone(net)

                y = net(x)
                self.assertIsNotNone(y)
                criterion = nn.L1Loss()
                loss = criterion(y, target)
                optimizer = optim.Adam(net.parameters())
                optimizer.zero_grad()

                params0 = [p.data.clone() for p in net.parameters()]
                loss.backward()
                optimizer.step()
                params1 = [p.data for p in net.parameters()]

                diff = [t.abs(p - q).sum() for p, q in zip(params0, params1)]

                self.assertEqual(len(list(net.named_parameters())),
                                 len(params0))
                nparams1 = net.named_parameters()
                # Check that all parameters change
                for d, (n, p), p0 in zip(diff, nparams1, params0):
                    msg = "param {}\nValue initial: {}\nValue after: {}\n" + \
                          "Grad: {}\nin Net \n{}"
                    msg = msg.format(n, p0, p.data, p.grad, net)
                    self.assertGreater(d, 0.0, msg=msg)
                # Check that the loss is not zero
                self.assertNotAlmostEqual(0, loss.data.abs().sum())


if __name__ == "__main__":
    unittest.main()
