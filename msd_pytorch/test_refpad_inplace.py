import unittest
from itertools import product as cartesianp
import torch.nn as nn
import torch.utils.cpp_extension as cppe
import torch as t
from torch.autograd import (Variable, Function)
from torch.nn import Parameter
from timeit import default_timer as timer
from msd_pytorch.reflectionpad_inplace import reflectionPad2DInplace
import os

os.environ['PATH'] = '/opt/sw/gcc-5.4.0/bin:' + os.environ['PATH']


def recompile():
    ref_inplace = cppe.load('reflectionpad_inplace',
                             sources=['msd_pytorch/reflectionpad_inplace.cpp', 'msd_pytorch/reflectionpad_inplace_cuda.cu'],
                             extra_cflags=[ '-Werror', '-Wfatal-errors', '-Wextra'],
                             # extra_cuda_cflags=[ '-Werror', '-Wfatal-errors', '-Wextra'],
                             extra_include_paths=cppe.include_paths(cuda=True),
                             verbose = True)
    return ref_inplace


reflectionpad_inplace = recompile()



class NewReflectionPad2DInplaceFunction(Function):
    @staticmethod
    def forward(ctx, input, padding):
        ctx.padding = padding

        assert 3 * padding < input.shape[2], \
            "Too much padding for height"
        assert 3 * padding < input.shape[3], \
            "Too much padding for width"

        output = reflectionpad_inplace.forward(input, padding)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        padding = ctx.padding
        return reflectionpad_inplace.backward(gradOutput, padding), None

class IdentityFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return input.clone()
    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput.clone();


def show_grads(grad1, grad2, gradoutput):
    grad1 = grad1[0, 0, ...].cpu().numpy()
    grad2 = grad2[0, 0, ...].cpu().numpy()
    gradoutput = gradoutput[0,0, ...].cpu().numpy()

    opts = {'vmin' : -10, 'vmax' : 10}

    plt.figure()
    plt.title("gradOutput")
    plt.imshow(gradoutput, **opts)

    plt.figure()
    plt.title("input1.grad")
    plt.imshow(grad1, **opts)

    plt.figure()
    plt.title("input2.grad")
    plt.imshow(grad2, **opts)

    plt.figure()
    plt.title("Difference")
    plt.imshow(grad1 - grad2, **opts)

    plt.show()

class TestReflectionPadInPlace(unittest.TestCase):

    def test_out_of_bounds_writes(self):
        batch_sz = 3
        channels = [1,2,5]
        shapes = [30, 57, 253]
        pads = [1, 2, 3, 4]

        do3d = [False]

        for chan, shape, conv3d, padding in cartesianp(channels, shapes, do3d, pads):

            # chan = 1
            # shape = 10
            # conv3d = False

            ref1 = reflectionPad2DInplace
            ref2 = NewReflectionPad2DInplaceFunction.apply

            ndim = 3 if conv3d else 2
            shape = (shape, shape)
            pad_t = (padding, ) * (ndim * 2)

            orig1 = t.randn(batch_sz, chan, *shape).cuda()
            orig1 = t.bernoulli(t.Tensor(batch_sz, chan, *shape).uniform_(0, 1)).cuda()
            orig1[:, :, :, 0:2] = 5.0
            orig1[:, :, :, 2] = 2.0
            orig1[:, :, :, -1:-2] = 2.0
            orig1[:, :, :, -3] = 6.0
            orig2 = orig1.data.clone()
            self.assertAlmostEqual(0, (orig1 - orig2).abs().sum().item())
            input1 = orig1[1:2, ...]
            input2 = orig2[1:2, ...]
            grad_output = t.randn_like(input1)

            output1 = ref1(input1, pad_t)
            output2 = ref2(input2, padding)

            self.assertAlmostEqual(0, (output1 - output2).abs().sum().item())
            self.assertAlmostEqual(0, (orig1 - orig2).abs().sum().item())

    def test_backward(self):
        batch_sz = 3
        channels = [1,2,5]
        shapes = [30, 57, 253]
        pads = [1, 2, 3, 4]

        do3d = [False]

        type = t.float32
        for chan, shape, conv3d, padding in cartesianp(channels, shapes, do3d, pads):

            # chan = 1
            # shape = 10
            # conv3d = False
            # padding = 1

            ref1 = reflectionPad2DInplace
            ref2 = NewReflectionPad2DInplaceFunction.apply

            ndim = 3 if conv3d else 2
            shape = (shape, shape)
            pad_t = (padding, ) * (ndim * 2)

            orig1 = t.randn(batch_sz, chan, *shape, dtype=type).cuda()
            orig1 = t.bernoulli(t.zeros(batch_sz, chan, *shape, dtype=t.float64).uniform_(0, 1)).cuda()
            orig1[:, :, :, 0:2] = 5.0
            orig1[:, :, :, 2] = 2.0
            orig1[:, :, :, -1:-2] = 2.0
            orig1[:, :, :, -3] = 6.0
            orig2 = orig1.data.clone()
            self.assertAlmostEqual(0, (orig1 - orig2).abs().sum().item())
            input1 = orig1[1:2, ...]
            input2 = orig2[1:2, ...]
            grad_output = t.randn_like(input1)

            input1.requires_grad = input2.requires_grad = True

            output1 = ref1(IdentityFunction.apply(input1), pad_t)
            output2 = ref2(IdentityFunction.apply(input2), padding)

            self.assertAlmostEqual(0, (output1 - output2).abs().sum().item())
            self.assertAlmostEqual(0, (orig1 - orig2).abs().sum().item())

            output1.backward(grad_output)
            output2.backward(grad_output)

            self.assertIsNotNone(input1.grad)
            self.assertIsNotNone(input2.grad)

            self.assertAlmostEqual(0, (input1.grad - input2.grad).abs().sum().item())

unittest.main()
