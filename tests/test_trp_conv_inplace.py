#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for trp_conv_inplace."""


import unittest
import msd_pytorch.trp_conv_inplace as trp
import torch
from torch.autograd import Variable
import torch.nn as nn


class Testtrp_conv_inplace(unittest.TestCase):
    """Tests for trp_conv_inplace."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_conv2d(self):
        """Test trp conv 2d."""

        batch_sz = 5
        in_channels = 5
        size = (45,) * 2
        xi = torch.ones(batch_sz, in_channels, *size).cuda()
        xc = torch.ones(batch_sz, in_channels, *size).cuda()

        output = torch.ones(batch_sz, 1, *size).cuda()
        ci = trp.Conv2dInPlaceModule(output, in_channels, 1).cuda()
        cc = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1).cuda()

        ci.weight.data.fill_(1)
        cc.weight.data.fill_(1)
        ci.bias.data.zero_()
        cc.bias.data.zero_()

        xiv = Variable(xi, requires_grad=True)
        xcv = Variable(xc, requires_grad=True)

        yi = ci(xiv)
        yc = cc(xcv)
        self.assertEqual(yi.shape, yc.shape)

        # Check center of output, where the output should be equal.
        d = 1
        yi_ = yi[:, :, d:-d, d:-d]
        yc_ = yc[:, :, d:-d, d:-d]

        # Check that pytorch and own convolution agree in the center:
        self.assertAlmostEqual(0, (yi_ - yc_).data.abs().sum())
        # Check that the result of in-place convolution has ended up
        # in the 'output' buffer.
        self.assertAlmostEqual(0, (yi.data - output).abs().sum())

    def test_conv3d(self):
        """Test trp conv 3d."""

        batch_sz = 5
        in_channels = 5
        size = (23,) * 3
        k_shape = (3,) * 3
        dtype = torch.double

        xi = torch.randn(batch_sz, in_channels, *size, dtype=dtype).cuda()
        xc = xi.clone()
        weight = torch.randn(1, in_channels, *k_shape, dtype=dtype).cuda()
        bias = torch.randn(1, dtype=dtype).cuda()
        output = torch.ones(batch_sz, 1, *size, dtype=dtype).cuda()

        ci = trp.Conv3dInPlaceModule(output, in_channels, 1).cuda()
        cc = nn.Conv3d(in_channels, 1, kernel_size=3, padding=1).cuda()

        ci.weight.data = weight
        cc.weight.data = weight
        ci.bias.data = bias
        cc.bias.data = bias

        xiv = Variable(xi, requires_grad=True)
        xcv = Variable(xc, requires_grad=True)

        yi = ci(xiv)
        yc = cc(xcv)
        self.assertEqual(yi.shape, yc.shape)

        # Check center of output, where the output should be equal.
        d = 2
        yi_ = yi[:, :, d:-d, d:-d, d:-d]
        yc_ = yc[:, :, d:-d, d:-d, d:-d]

        # Check that pytorch and own convolution agree in the center:
        self.assertAlmostEqual(yi_.sum().item(), yc_.sum().item())
        self.assertAlmostEqual(0, (yi_ - yc_).data.abs().sum().item())
        # Check that the result of in-place convolution has ended up
        # in the 'output' buffer.
        self.assertAlmostEqual(0, (yi.data - output).abs().sum().item())
