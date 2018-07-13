import torch.nn as nn
import torch as t
from torch.autograd import (Variable, Function)
from msd_pytorch.reflectionpad_inplace import (
    ReflectionPad2DInplaceModule, crop2d,
    reflectionPad3DInplace,
    reflectionPad2DInplace,
    ReflectionPad3DInplaceModule)
from torch.nn.modules.utils import (_ntuple)
import unittest


class IdentityFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput.clone()


identity = IdentityFunction.apply


class reflectionpadTest(unittest.TestCase):

    def test_reflection_inplace(self):
        t.manual_seed(1)

        size = (10, 9)
        padding = 5
        padL, padR, padT, padB = _ntuple(4)(padding)

        x = Variable(t.randn(2, 3, *size), requires_grad=True)
        # Create a zero padded big X
        X = Variable(x.data.clone())
        X = nn.ConstantPad2d(padding, 0)(x)
        X = Variable(X.data, requires_grad=True)  # Retain gradient

        # Apply reflection padding to x and inplace reflection padding
        # to X.
        y = nn.ReflectionPad2d(padding)(x)
        # An additional copy is necessary to keep pytorch happy. We
        # avoid 'RuntimeError: a leaf Variable that requires grad
        # has been used in an in-place operation.'
        X_ = identity(X)
        Y = ReflectionPad2DInplaceModule(padding)(X_)

        # Y and y should be identical.
        self.assertAlmostEqual(0, (y - Y).data.abs().sum())

        # Test backwards.
        g = y.data.clone().normal_()
        y.backward(g)
        Y.backward(g)

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(X.grad)

        # Resize X.grad and compare with x.grad.
        X_g = X.grad.data[:, :, padL:(-padR), padT:-padB]
        self.assertEqual(x.grad.data.shape, X_g.shape)

    def test_crop2d(self):
        # Test forward and backward.
        f = crop2d

        for use_cuda in [True, False]:
            # x has size 1 x 1 x 10 x 10
            x = t.arange(0, 100).unfold(0, 10, 10).unsqueeze(0).unsqueeze(0)
            if use_cuda:
                x.cuda()
            x = Variable(x, requires_grad=True)

            y = f(x)
            z = y.sum()
            z.backward()

            x_grad = t.ones(10, 10)
            x_grad[:, 0] = 0
            x_grad[:, 9] = 0
            x_grad[0, :] = 0
            x_grad[9, :] = 0
            x_grad.unsqueeze_(0).unsqueeze_(0)

            if use_cuda:
                x_grad.cuda()

            testdiff = abs((x.grad.data - x_grad).sum())

            self.assertAlmostEqual(testdiff, 0)


    def test_reflection_inplace2d(self):
        t.manual_seed(1)

        size = (9, 9)
        padding2d = (2, 3, 4, 1)

        # We take some 3D data x0 and want to check that 3d inplace
        # padding works.
        x0 = t.randn(2, 3, *size)
        x1 = Variable(x0.clone(), requires_grad=True)
        x2 = nn.ReflectionPad2d(padding2d)(x1)
        x3 = reflectionPad2DInplace(x2, padding2d)

        self.assertAlmostEqual(0, (x2 - x3).data.abs().sum())
        # Test backwards.
        g = x3.data.clone().normal_()

        x3.backward(g)
        self.assertIsNotNone(x1.grad)

        y0 = x0.clone()
        y1 = Variable(y0.clone(), requires_grad=True)
        y2 = nn.ReflectionPad2d(padding2d)(y1)
        # Test forward equal
        self.assertEqual(y2.shape, x3.shape)
        self.assertAlmostEqual(0, (y2 - x3).data.abs().sum())
        # Test grad
        y2.backward(g)

        self.assertIsNotNone(y1.grad)
        self.assertEqual(y1.grad.shape, x1.grad.shape)
        self.assertAlmostEqual(0, (y1.grad - x1.grad).data.abs().sum(),
                               places=4)


    def test_reflection_inplace3d_depthwise(self):
        t.manual_seed(1)

        size = (10, 9, 9)
        padding3d = (1, 2, 3, 4, 0, 0)
        padding2d = (1, 2, 3, 4)
        # padL, padR, padT, padB, pad0, pad1 = _ntuple(6)(padding)

        # We take some 3D data x0 and want to check that 3d inplace
        # padding works.
        x0 = t.randn(2, 3, *size)
        x1 = Variable(x0.clone(), requires_grad=True)
        x2 = nn.ConstantPad3d(padding3d, 1)(x1)
        x3 = reflectionPad3DInplace(x2, padding3d)

        # Test backwards.
        g = x3.data.clone().normal_()
        x3.backward(g)
        self.assertIsNotNone(x1.grad)

        # Pytorch does not have a 3d reflection pad. So we have
        # to make do with the 2d version. We check for every 'row' if
        # the pytorch 2d pad and our 3d reflection pad coincide. We do
        # the same for the gradient.
        for i in range(size[0]):
            y0 = x0[:, :, i, :, :].clone()
            y1 = Variable(y0.clone(), requires_grad=True)
            y2 = nn.ReflectionPad2d(padding2d)(y1)
            # Test forward equal
            self.assertEqual(y2.shape, x3[:, :, i, :, :].shape)
            self.assertAlmostEqual(
                0, (y2 - x3[:, :, i, :, :]).data.abs().sum())
            # Test grad
            y2.backward(g[:, :, i, :, :])
            self.assertIsNotNone(y1.grad)
            self.assertEqual(y1.grad.shape, x1.grad[:, :, i, :, :].shape)
            self.assertAlmostEqual(0,
                                   (y1.grad - x1.grad[:, :, i, :, :])
                                   .data.abs().sum(),
                                   places=4)

    def test_reflection_inplace3d_heightwise(self):
        t.manual_seed(1)

        size = (10, 9, 9)
        padding3d = (1, 2, 0, 0, 3, 4)
        padding2d = (1, 2, 3, 4)

        # We take some 3D data x0 and want to check that 3d inplace
        # padding works.
        x0 = t.randn(2, 3, *size)
        x1 = Variable(x0.clone(), requires_grad=True)
        x2 = nn.ConstantPad3d(padding3d, 1)(x1)
        x3 = reflectionPad3DInplace(x2, padding3d)

        # Test backwards.
        g = x3.data.clone().normal_()
        x3.backward(g)
        self.assertIsNotNone(x1.grad)

        # Pytorch does not have a 3d reflection pad. So we have
        # to make do with the 2d version. We check for every 'row' if
        # the pytorch 2d pad and our 3d reflection pad coincide. We do
        # the same for the gradient.
        for i in range(size[1]):
            y0 = x0[:, :, :, i, :].clone()
            y1 = Variable(y0.clone(), requires_grad=True)
            y2 = nn.ReflectionPad2d(padding2d)(y1)
            # Test forward equal
            self.assertEqual(y2.shape, x3[:, :, :, i, :].shape)
            self.assertAlmostEqual(
                0, (y2 - x3[:, :, :, i, :]).data.abs().sum())
            # Test grad
            y2.backward(g[:, :, :, i, :])
            self.assertIsNotNone(y1.grad)
            self.assertEqual(y1.grad.shape, x1.grad[:, :, :, i, :].shape)
            self.assertAlmostEqual(0,
                                   (y1.grad - x1.grad[:, :, :, i, :])
                                   .data.abs().sum(),
                                   places=4)


    def test_reflection_inplace3d_widthwise(self):
        t.manual_seed(1)

        size = (10, 9, 9)
        padding3d = (0, 0, 1, 2, 3, 4)
        padding2d = (1, 2, 3, 4)

        # We take some 3D data x0 and want to check that 3d inplace
        # padding works.
        x0 = t.randn(2, 3, *size)
        x1 = Variable(x0.clone(), requires_grad=True)
        x2 = nn.ConstantPad3d(padding3d, 1)(x1)
        x3 = reflectionPad3DInplace(x2, padding3d)

        # Test backwards.
        g = x3.data.clone().normal_()
        x3.backward(g)
        self.assertIsNotNone(x1.grad)

        # Pytorch does not have a 3d reflection pad. So we have
        # to make do with the 2d version. We check for every 'row' if
        # the pytorch 2d pad and our 3d reflection pad coincide. We do
        # the same for the gradient.
        for i in range(size[2]):
            y0 = x0[:, :, :, :, i].clone()
            y1 = Variable(y0.clone(), requires_grad=True)
            y2 = nn.ReflectionPad2d(padding2d)(y1)
            # Test forward equal
            self.assertEqual(y2.shape, x3[:, :, :, :, i].shape)
            self.assertAlmostEqual(
                0, (y2 - x3[:, :, :, :, i]).data.abs().sum())
            # Test grad
            y2.backward(g[:, :, :, :, i])
            self.assertIsNotNone(y1.grad)
            self.assertEqual(y1.grad.shape, x1.grad[:, :, :, :, i].shape)
            self.assertAlmostEqual(0,
                                   (y1.grad - x1.grad[:, :, :, :, i])
                                   .data.abs().sum(),
                                   places=4)


if __name__ == '__main__':
    unittest.main()
