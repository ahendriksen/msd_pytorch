import torch.nn as nn
import torch as t
from torch.autograd import (Variable, Function)
from msd_pytorch.reflectionpad_inplace import (ReflectionPad2DInplaceModule, crop2d)
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


if __name__ == '__main__':
    unittest.main()
