from msd_pytorch.trp_conv_inplace import (Conv2dInPlaceModule)
from msd_pytorch.msd_module import (MSDModule, msd_dilation, MSDFinalLayer)
from msd_pytorch.relu_inplace import (ReLUInplaceFunction)
from msd_pytorch.stitch import (stitchCopy)
from torch.autograd import (Variable)
import msd_pytorch.stitch as stitch
import torch as t
import torch.nn as nn
import torch.optim as optim
import unittest


class MSDModuleTest(unittest.TestCase):

    def test_final_layer(self):
        """Test MSDFinalLayer module

        We check that the msd_final module does exactly the same as a
        1x1 convolution.

        """
        conv3d = True
        shape = (25, ) * (3 if conv3d else 2)
        k_shape = (1,) * (3 if conv3d else 2)
        device = t.device("cuda:0")
        dtype = t.double

        batch_size = 3
        c_in = 10
        c_out = 2

        input = t.randn(batch_size, c_in, *shape, dtype=dtype, device=device)
        bias = t.randn(c_out, dtype=dtype, device=device)
        weight = t.randn(c_out, c_in, 1, dtype=dtype, device=device)

        msd_final = MSDFinalLayer(c_in, c_out)
        msd_final.linear.bias.data = bias
        msd_final.linear.weight.data = weight

        if conv3d:
            conv = nn.Conv3d(c_in, c_out, 1)
        else:
            conv = nn.Conv2d(c_in, c_out, 1)
        conv.bias.data = bias
        conv.weight.data = weight.view(c_out, c_in, *k_shape)

        # Check that outputs have the same shape
        output1 = conv(input)
        output2 = msd_final(input)
        self.assertEqual(output1.shape, output2.shape)
        # And have the same values.
        diff = (output1 - output2).abs().sum().item()
        self.assertAlmostEqual(diff, 0)

    def test_clear_buffers(self):
        t.manual_seed(2)

        c_in, c_out = 2, 3
        depth, width = 11, 5
        size = (30, 30)

        x = Variable(t.randn(1, c_in, *size)).cuda()
        target = Variable(t.randn(1, c_out, *size)).cuda()

        net = MSDModule(c_in, c_out, depth, width, msd_dilation, conv3d=False)
        y1 = net(x)
        net.clear_buffers()
        y2 = net(x)
        self.assertAlmostEqual(0, (y1 - y2).abs().data.sum())

    def test_3d(self):
        batch_sz = 1
        c_in, c_out = 2, 3
        depth, width = 11, 3
        size = (20,) * 3
        x = t.randn(batch_sz, c_in, *size).cuda()

        net = MSDModule(c_in, c_out, depth, width, msd_dilation, conv3d=True)

        output = net(Variable(x))

        # The final 1x1 convolution has weight initialized to zero, so
        # output should be zero.
        self.assertAlmostEqual(0, output.data.abs().sum())

    def test_reflect(self):
        batch_sz = 1
        c_in, c_out = 2, 3
        depth, width = 11, 3
        size = (20,) * 2
        x = t.randn(batch_sz, c_in, *size).cuda()
        target = t.randn(batch_sz, c_out, *size).cuda()

        net = MSDModule(c_in, c_out, depth, width, msd_dilation,
                        reflect=True, conv3d=False)

        output = net(Variable(x))

        self.assertEqual(target.shape, output.data.shape)

        loss = nn.MSELoss()(output, Variable(target))
        loss.backward()

        self.assertAlmostEqual(0, output.data.abs().sum())

    def test_with_tail(self):
        batch_sz = 1
        c_in, c_out = 2, 3
        depth, width = 11, 3
        size = (20,) * 3
        x = t.randn(batch_sz, c_in, *size).cuda()
        target = Variable(t.randn(batch_sz, 1, *size).cuda())

        net = nn.Sequential(
            MSDModule(c_in, c_out, depth, width, msd_dilation, conv3d=True),
            nn.Conv3d(3, 1, 1))
        net.cuda()

        output = net(Variable(x))
        loss = nn.MSELoss()(output, target)
        loss.backward()

        self.assertNotAlmostEqual(0, output.abs().sum().item())

    def test_backward(self):
        dbg = False

        def dbg_print(*args):
            if (dbg):
                print(*args)

        d = 3
        input_size = (20, 20)
        # Prepare layer and gradient storage
        L = t.zeros(1, d + 1, *input_size).cuda()
        G = t.zeros(1, d + 1, *input_size).cuda()

        cs = [Conv2dInPlaceModule(None, i + 1, 1, kernel_size=3, dilation=1)
              for i in range(d)]
        relu = nn.ReLU(inplace=True)
        relu = ReLUInplaceFunction.apply

        for i, c in enumerate(cs):
            c.weight.data.fill_(i + 1)
            c.bias.data.fill_(i % 2)

        x = t.Tensor(1, 1, *input_size).fill_(2).cuda()
        x.requires_grad = True

        dbg_print("A", L._version)
        output = stitchCopy(x, L, G, 0)
        dbg_print("B", L._version, output._version, output.grad_fn)

        conv = cs[0]
        conv.output = L.narrow(1, 1, 1)  # narrow(L, 1, 1, 1)
        dbg_print("C", L._version, output._version, output.grad_fn)
        output = conv(x)
        dbg_print("D", L._version, output._version, output.grad_fn)
        output = relu(output)
        dbg_print("E", L._version, output._version, output.grad_fn)
        output = stitch.stitchLazy(output, L, G, 1)
        dbg_print("F", L._version, output._version, output.grad_fn)

        conv = cs[1]
        conv.output = L.narrow(1, 2, 1)  # narrow(L, 1, 2, 1)
        dbg_print("G", L._version, output._version, output.grad_fn)
        output = conv(output)
        dbg_print("H", L._version, output._version, output.grad_fn)
        output = relu(output)
        dbg_print("I", L._version, output._version, output.grad_fn)
        output = stitch.stitchLazy(output, L, G, 2)
        dbg_print("J", L._version, output._version, output.grad_fn)

        conv = cs[2]
        dbg_print("K", L._version, output._version, output.grad_fn)
        conv.output = L.narrow(1, 3, 1)  #
        dbg_print("L", L._version, output._version, output.grad_fn)
        output = conv(output)
        dbg_print("M", L._version, output._version, output.grad_fn)
        output = relu(output)
        dbg_print("N", L._version, output._version, output.grad_fn)
        output = stitch.stitchLazy(output, L, G, 3)
        dbg_print("O", L._version, output._version, output.grad_fn)

        output.backward(t.ones_like(output))
        dbg_print(x.grad.shape)

    def test_parameters_change(self):
        # This test ensures that all parameters are updated after an
        # update step.
        t.manual_seed(1)

        size = (30, 30)
        for batch_sz in [1]:
            for depth in range(0, 20, 5):
                width = c_in = c_out = batch_sz
                x = Variable(t.randn(batch_sz, c_in, *size)).cuda()
                target = Variable(t.randn(batch_sz, c_out, *size)).cuda()
                self.assertTrue(x.data.is_cuda)

                net = MSDModule(c_in, c_out, depth, width, msd_dilation,
                                conv3d=False)

                self.assertIsNotNone(net)

                params0 = [p.data.clone() for p in net.parameters()]
                # Train for two iterations. The convolution weights in
                # the MSD layers are not updated after the first
                # training step because the final 1x1 convolution
                # weights are zero.
                for i in [1, 2]:
                    y = net(x)
                    self.assertIsNotNone(y)
                    criterion = nn.L1Loss()
                    loss = criterion(y, target)
                    optimizer = optim.Adam(net.parameters())
                    optimizer.zero_grad()

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
                self.assertNotAlmostEqual(0, loss.abs().item())


if __name__ == "__main__":
    unittest.main()
