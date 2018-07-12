import torch.nn as nn
import torch as t
from torch.autograd import (Variable)
from msd_pytorch.stitch import (StitchCopyModule, stitchSlow, stitchCopy)
import unittest


class stitchTest(unittest.TestCase):

    def test_compare_slow(self):
        # This test compares the slow stitching with normal stitching
        # and ensures they give the same results.
        size = (20, 20)
        depth = 5
        batch_sz = 3
        # Prepare buffers
        L = t.zeros(batch_sz, depth, *size)
        G = t.zeros(batch_sz, depth, *size)
        # Prepare inputs and targets
        input = Variable(t.randn(batch_sz, 1, *size), requires_grad=True)
        input_slow = Variable(input.data.clone(), requires_grad=True)
        target = Variable(t.randn(batch_sz, 1, *size))
        # Prepare operations
        cs = [nn.Conv2d(i + 1, 1, 1) for i in range(depth)]
        fs = [StitchCopyModule(L, G, i) for i in range(depth)]
        # Compute output
        net = nn.Sequential(*[val for pair in zip(fs, cs) for val in pair])
        output = net(input)
        # Compute output slow
        output_slow = input_slow
        for c in cs[:-1]:
            output_slow = stitchSlow(output_slow, c(output_slow))
        output_slow = cs[-1](output_slow)
        # Determine loss for both outputs
        criterion = nn.L1Loss()
        loss = criterion(output, target)
        loss.backward()
        loss_slow = criterion(output_slow, target)
        loss_slow.backward()

        def diff(x, y):
            x = x.data if isinstance(x, Variable) else x
            y = y.data if isinstance(y, Variable) else y
            return (x - y).abs().sum()
        # Assert outputs are equal, losses are equal
        self.assertAlmostEqual(0, diff(output, output_slow))
        self.assertAlmostEqual(0, diff(loss, loss_slow))
        # Assert gradients of the input are equal
        self.assertAlmostEqual(0, diff(input.grad, input_slow.grad))

    def test_module(self):
        size = (2, 2)
        depth = 3
        batch_sz = 2

        L = t.zeros(batch_sz, depth, *size)
        G = t.zeros(batch_sz, depth, *size)

        input, target = (Variable(t.randn(batch_sz, 1, *size))
                         for i in range(2))

        cs = [nn.Conv2d(i + 1, 1, 1) for i in range(depth)]
        fs = [StitchCopyModule(L, G, i) for i in range(depth)]

        net = nn.Sequential(*[val for pair in zip(fs, cs) for val in pair])
        output = net(input)

        criterion = nn.L1Loss()
        loss = criterion(output, target)
        loss.backward()

    def test_module_cuda(self):
        size = (2, 2)
        depth = 3
        batch_sz = 2

        L = t.zeros(batch_sz, depth, *size).cuda()
        G = t.zeros(batch_sz, depth, *size).cuda()

        input, target = (Variable(t.randn(batch_sz, 1, *size).cuda())
                         for i in range(2))

        cs = [nn.Conv2d(i + 1, 1, 1) for i in range(depth)]
        fs = [StitchCopyModule(L, G, i) for i in range(depth)]

        net = nn.Sequential(*[val for pair in zip(fs, cs) for val in pair])
        net.cuda()
        output = net(input)

        criterion = nn.L1Loss()
        target = target.cuda()

        loss = criterion(output, target)
        loss.backward()

    def test_identity(self):
        # Prepare layer and gradient storage
        L = t.zeros(1, 3, 1)
        G = t.zeros(1, 3, 1)

        x = Variable(t.Tensor(1, 1, 1).fill_(5), requires_grad=True)

        l0 = stitchCopy(x, L, G, 0)

        c1 = nn.Conv1d(1, 1, 1)
        c1.weight.data.fill_(2)
        c1.bias.data.fill_(0)

        l1 = stitchCopy(c1(l0), L, G, 1)

        c2 = nn.Conv1d(2, 1, 1)
        c2.weight.data.fill_(4)
        c2.bias.data.fill_(0)

        c2_out = c2(l1)
        l2 = stitchCopy(c2_out, L, G, 2)

        c3 = nn.Conv1d(3, 1, 1)
        c3.weight.data.fill_(2)
        c3.bias.data.fill_(0)

        l3 = c3(l2)

        l3.backward()

    def test2d(self):
        d = 3
        input_size = (2, 2)
        # Prepare layer and gradient storage
        L = t.zeros(1, d, *input_size)
        G = t.zeros(1, d, *input_size)

        x = Variable(t.Tensor(1, 1, *input_size).fill_(2), requires_grad=True)

        cs = [nn.Conv2d(i + 1, 1, 1) for i in range(d)]
        for i, c in enumerate(cs):
            c.weight.data.fill_(i + 1)
            c.bias.data.fill_(i % 2)

        output = x
        outputs = [x]
        for i in range(d - 1):
            output = stitchCopy(output, L, G, i)
            output = cs[i](output)
            outputs.append(output)

        output = output.sum()
        output.backward()


if __name__ == '__main__':
    unittest.main()
