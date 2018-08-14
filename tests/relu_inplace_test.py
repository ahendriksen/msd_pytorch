from itertools import product as cartesianp
from torch.autograd import (Variable, Function)
from torch.nn import Parameter
import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.cpp_extension as cppe
import unittest
from msd_pytorch.relu_inplace import (ReLUInplaceFunction, ReLUInplaceModule)


class TestReLUInPlace(unittest.TestCase):

    def test_out_of_bounds_writes(self):
        batch_sz = 3
        channels = [1,2,5]
        shapes = [11, 20, 253]

        do3d = [False, True]

        for chan, shape, conv3d in cartesianp(channels, shapes, do3d):

            relu1 = nn.ReLU(inplace=True)
            relu2 = ReLUInplaceModule()

            ndim = 3 if conv3d else 2
            shape = (shape,) * ndim

            orig1 = t.randn(batch_sz, chan, *shape).cuda()
            orig2 = orig1.data.clone()
            self.assertAlmostEqual(0, (orig1 - orig2).abs().sum().item())
            input1 = orig1[1:2, ...]
            input2 = orig2[1:2, ...]
            grad_output = t.randn_like(input1)

            output1 = relu1(input1)
            output2 = relu2(input2)

            self.assertAlmostEqual(0, (output1 - output2).abs().sum().item())
            self.assertAlmostEqual(0, (orig1 - orig2).abs().sum().item())

    def test_backward(self):
        batch_sz = 3
        channels = [1, 2, 5]
        shapes = [11, 20, 63]
        dtype = t.double
        do3d = [False, True]

        for chan, shape, conv3d in cartesianp(channels, shapes, do3d):

            relu1 = nn.ReLU(inplace=False)
            relu2 = ReLUInplaceModule()

            ndim = 3 if conv3d else 2
            shape = (shape,) * ndim

            orig1 = t.randn(batch_sz, chan, *shape, dtype=dtype).cuda()
            orig2 = orig1.data.clone()
            input1 = orig1[1:2, ...]
            input2 = orig2[1:2, ...]
            grad_output = t.randn_like(input1)

            input1.requires_grad, input2.requires_grad = True, True

            output1 = relu1(input1)
            output2 = relu2(input2)

            self.assertAlmostEqual(0, (output1 - output2).abs().sum().item())

            # # Test backward step
            output1.backward(grad_output)
            output2.backward(grad_output)

            grad_diff = (input1.grad - input2.grad).abs().sum().item()
            self.assertAlmostEqual(0, grad_diff)
