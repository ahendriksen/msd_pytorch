from pytest import approx
import pytest
from msd_pytorch.msd_module import MSDModule, MSDFinalLayer
from torch.autograd import Variable
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.autograd import gradcheck


def test_msd_module_3D():
    net = MSDModule(2, 3, 5, 1, ndim=3).cuda()
    x = t.ones(1, 2, 7, 7, 7).cuda()
    # The final layer is initialized with zeros. Therefore the output
    # of an untrained network must always be zero.
    assert net(x).sum().item() == 0.0
    assert net(x).shape == x.shape


def test_msd_gradients():
    t.manual_seed(1)

    dtype = t.double
    size = (11, 13)
    batch_sz = 2

    for depth in [9]:
        print(f"Depth: {depth}")
        width = c_in = c_out = batch_sz
        x = Variable(t.randn(batch_sz, c_in, *size, dtype=dtype)).cuda()
        x.requires_grad = True

        net = MSDModule(c_in, c_out, depth, width).cuda()
        net.double()

        # The weights of the final layer are initialized to zero by
        # default. This makes it trivial to pass gradcheck. Therefore,
        # we reinitialize all weights randomly.
        for p in net.parameters():
            p.data = t.randn_like(p.data)

        gradcheck(net, [x], raise_exception=True, atol=1e-4, rtol=1e-3)


def test_final_layer():
    """Test MSDFinalLayer module

    We check that the msd_final module does exactly the same as a
    1x1 convolution.

    """
    for conv3d in [False, True]:

        shape = (25,) * (3 if conv3d else 2)
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
        assert output1.shape == output2.shape

        # And have the same values.
        diff = (output1 - output2).abs().sum().item()
        assert diff == approx(0)


def test_parameters_change():
    # This test ensures that all parameters are updated after an
    # update step.
    t.manual_seed(1)

    size = (30, 30)
    for batch_sz in [1]:
        for depth in range(1, 20, 6):
            width = c_in = c_out = batch_sz
            x = Variable(t.randn(batch_sz, c_in, *size)).cuda()
            target = Variable(t.randn(batch_sz, c_out, *size)).cuda()
            assert x.data.is_cuda

            net = MSDModule(c_in, c_out, depth, width).cuda()

            params0 = dict((n, p.data.clone()) for n, p in net.named_parameters())
            # Train for two iterations. The convolution weights in
            # the MSD layers are not updated after the first
            # training step because the final 1x1 convolution
            # weights are zero.
            optimizer = optim.Adam(net.parameters())
            optimizer.zero_grad()
            for _ in range(2):
                y = net(x)
                assert y is not None
                criterion = nn.L1Loss()
                loss = criterion(y, target)
                loss.backward()
                optimizer.step()

            params1 = dict(net.named_parameters())

            for name in params1.keys():
                p0, p1 = params0[name], params1[name]
                d = abs(p0 - p1.data.clone()).sum().item()
                assert 0.0 < d, (
                    f"Parameter {name} left unchanged: \n"
                    f"Initial value: {p0}\n"
                    f"Current value: {p1}\n"
                    f"Gradient: {p1.grad}\n"
                )

            # Check that the loss is not zero
            assert loss.abs().item() != approx(0.0)


def test_zero_depth_network():
    with pytest.raises(ValueError):
        MSDModule(1, 1, depth=0, width=1)
    with pytest.raises(ValueError):
        MSDModule(1, 1, depth=1, width=0)
