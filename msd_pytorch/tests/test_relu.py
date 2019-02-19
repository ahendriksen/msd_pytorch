from . import torch_equal
from itertools import product as cartesianp
import torch as t
import torch.nn as nn
from msd_pytorch.relu_inplace import ReLUInplaceModule


def test_out_of_bounds_writes():
    batch_sz = 3
    channels = [1, 2, 5]
    shapes = [(11, 41), (253, 531), (11, 31, 41)]

    for chan, shape in cartesianp(channels, shapes):
        relu1 = nn.ReLU(inplace=True)
        relu2 = ReLUInplaceModule()

        orig1 = t.randn(batch_sz, chan, *shape).cuda()
        orig2 = orig1.data.clone()

        input1 = orig1[1:2, ...]
        input2 = orig2[1:2, ...]

        output1 = relu1(input1)
        output2 = relu2(input2)

        assert torch_equal(output1, output2)
        assert torch_equal(orig1, orig2)


def test_backward():
    batch_sz = 3
    channels = [1, 2, 5]
    shapes = [(11, 41), (253, 131), (11, 31, 41)]
    dtype = t.double

    for chan, shape in cartesianp(channels, shapes):
        relu1 = nn.ReLU(inplace=False)
        relu2 = ReLUInplaceModule()

        orig1 = t.randn(batch_sz, chan, *shape, dtype=dtype).cuda()
        orig2 = orig1.data.clone()
        input1 = orig1[1:2, ...]
        input2 = orig2[1:2, ...]
        grad_output = t.randn_like(input1)

        input1.requires_grad, input2.requires_grad = True, True

        output1 = relu1(input1)
        output2 = relu2(input2)

        assert torch_equal(output1, output2)

        # # Test backward step
        output1.backward(grad_output)
        output2.backward(grad_output)

        assert torch_equal(input1.grad, input2.grad)
