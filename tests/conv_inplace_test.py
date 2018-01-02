import torch.nn as nn
import torch as t
from torch.autograd import Variable
from torch.autograd import gradcheck
from itertools import product as cartesianp
from msd_pytorch.conv_inplace import (conv2dInPlace, Conv2dInPlaceModule,
                                      Conv3dInPlaceModule)
import unittest


class TestConv2dInPlace(unittest.TestCase):

    def test_simple(self):
        batch_sz = 5
        in_channels = 5
        size = (20,) * 2
        xi = t.ones(batch_sz, in_channels, *size).cuda()
        xc = t.ones(batch_sz, in_channels, *size).cuda()

        output = t.ones(batch_sz, 1, *size).cuda()
        ci = Conv2dInPlaceModule(output, in_channels, 1).cuda()
        cc = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1).cuda()

        ci.weight.data.fill_(1)
        cc.weight.data.fill_(1)
        ci.bias.data.zero_()
        cc.bias.data.zero_()

        xiv = Variable(xi, requires_grad=True)
        xcv = Variable(xc, requires_grad=True)

        yi = ci(xiv)
        yc = cc(xcv)

        self.assertAlmostEqual(0, (yi.data - output).abs().sum())
        self.assertAlmostEqual(0, (yi - yc).data.abs().sum())

        ls = [('conv_ip', xiv, ci), ('conv2d', xcv, cc)]
        for i in range(2):
            for (n, x, c) in ls:
                t.cuda.synchronize()
                if x.grad is not None:
                    x.grad.data.zero_()

                y = c(x)
                c.zero_grad()
                y.sum().backward()

                t.cuda.synchronize()

    def test_3d(self):
        batch_sz = 5
        in_channels = 5
        size = (20,) * 3
        xi = t.ones(batch_sz, in_channels, *size).cuda()
        xc = t.ones(batch_sz, in_channels, *size).cuda()

        output = t.ones(batch_sz, 1, *size).cuda()
        ci = Conv3dInPlaceModule(output, in_channels, 1).cuda()
        cc = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1).cuda()

        ci.weight.data.fill_(1)
        cc.weight.data.fill_(1)
        ci.bias.data.zero_()
        cc.bias.data.zero_()

        xiv = Variable(xi, requires_grad=True)
        xcv = Variable(xc, requires_grad=True)

        yi = ci(xiv)
        yc = cc(xcv)

        self.assertAlmostEqual(0, (yi.data - output).abs().sum())
        self.assertAlmostEqual(0, (yi - yc).data.abs().sum())

        ls = [('conv_ip', xiv, ci), ('conv2d', xcv, cc)]
        for i in range(2):
            for (n, x, c) in ls:
                t.cuda.synchronize()
                if x.grad is not None:
                    x.grad.data.zero_()

                y = c(x)
                c.zero_grad()
                y.sum().backward()

                t.cuda.synchronize()

    def test_grad(self):
        # gradcheck takes a tuple of tensor as input, check if your gradient
        # evaluated with these tensors are close enough to numerical
        # approximations and returns True if they all verify this condition.
        t.manual_seed(1)

        input = t.randn(1, 2, 20, 20)
        output = t.randn(1, 1, 20, 20)
        weight = t.randn(1, 2, 3, 3)
        bias = t.randn(1)
        input = Variable(input.double().cuda(), requires_grad=True)
        output = Variable(output.double().cuda(), requires_grad=True)
        weight = Variable(weight.double().cuda(), requires_grad=True)
        bias = Variable(bias.double().cuda(), requires_grad=True)
        dilation = padding = (2, 2)
        stride = (1, 1)

        test_input = (input, weight, bias, output, padding, stride, dilation)
        test = gradcheck(conv2dInPlace, test_input, eps=1e-4, atol=1e-4)
        self.assertTrue(test)

    def test_vs_conv(self):

        t.manual_seed(1)
        for batch_sz, n_channels, dil in cartesianp(
                [1, 2, 3, 5], [1, 2, 5], [1, 2, 10]):

            size = (20,) * 2
            L = t.ones(batch_sz, n_channels + 1, *size).cuda()
            Ln = L.narrow(1, 0, n_channels)
            x = Variable(Ln, requires_grad=True)
            x1 = Variable(Ln.clone(), requires_grad=True)

            # Create conventional convolution
            c1 = nn.Conv2d(n_channels, 1, kernel_size=3,
                           stride=1, padding=dil,
                           dilation=dil, groups=1, bias=True).cuda()

            # Create in place version
            output = L.narrow(1, n_channels, 1)
            c = Conv2dInPlaceModule(output, n_channels, 1,
                                    kernel_size=3, dilation=dil,
                                    padding=dil)

            # Make sure conventional and in place version have the
            # same parameters.
            c.weight.data.copy_(c1.weight.data)
            c.bias.data.copy_(c1.bias.data)

            # Execute conventional convolution
            self.assertIsNone(c1.weight.grad)
            self.assertIsNone(c1.bias.grad)
            y1 = c1(x1)
            c1.zero_grad()
            y1.sum().backward()

            t.cuda.synchronize()

            # Execute in place convolution
            self.assertIsNone(c.weight.grad)
            self.assertIsNone(c.bias.grad)
            y = c(x)
            c.zero_grad()
            y.sum().backward()

            # L1 difference between to tensors (or vars).
            def diff(x, y):
                x = x.data if isinstance(x, Variable) else x
                y = y.data if isinstance(y, Variable) else y
                return (x - y).abs().sum()

            # Assert parameter sizes are the same.
            self.assertEqual(c.weight.data.shape, c1.weight.data.shape)
            self.assertEqual(c.bias.data.shape, c1.bias.data.shape)
            # Assert output sizes are the same.
            self.assertEqual(y1.data.shape, y.data.shape)
            # Assert outputs are equal
            self.assertAlmostEqual(0, diff(y1, y))
            # Assert that all gradients are equal.
            self.assertAlmostEqual(0, diff(x.grad, x1.grad), delta=1e-5)
            self.assertAlmostEqual(0, diff(c1.weight.grad, c.weight.grad))
            self.assertAlmostEqual(0, diff(c1.bias.grad, c.bias.grad))


if __name__ == "__main__":
    unittest.main()
