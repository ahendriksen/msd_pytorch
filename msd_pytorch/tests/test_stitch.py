from . import torch_equal
import torch.nn as nn
import torch as t
from torch.autograd import Variable
from msd_pytorch.stitch import (
    StitchCopyModule,
    stitchSlow,
    StitchBuffer,
)
from torch.autograd.gradcheck import get_analytical_jacobian


def test_compare_slow():
    """Compare the slow stitching with normal stitching
    """
    size = (20, 20)
    depth = 5
    batch_sz = 3

    # Prepare inputs and targets
    input = Variable(t.randn(batch_sz, 1, *size), requires_grad=True)
    input_slow = Variable(input.data.clone(), requires_grad=True)
    target = Variable(t.randn(batch_sz, 1, *size))

    # Prepare buffers
    buffer = StitchBuffer()
    buffer.like_(input, (batch_sz, depth, *size))

    # Prepare operations
    cs = [nn.Conv2d(i + 1, 1, 1) for i in range(depth)]
    fs = [StitchCopyModule(buffer, i) for i in range(depth)]

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

    # Assert outputs are equal, losses are equal
    assert torch_equal(output, output_slow)
    assert torch_equal(loss, loss_slow)

    # Assert gradients of the input are equal
    assert torch_equal(input.grad, input_slow.grad)


def test_stitch_grad():
    size = (10, 10)
    depth = 10
    batch_sz = 1
    device = t.device("cuda")

    buffer = StitchBuffer()
    buffer_shape = (batch_sz, depth, *size)

    fs = [StitchCopyModule(buffer, i) for i in range(depth)]
    cs = [nn.Conv2d(i + 1, 1, 1) for i in range(depth)]

    net = nn.Sequential(*[val for pair in zip(fs, cs) for val in pair])
    net.to(device)
    net.double()

    def f(x):
        buffer.like_(x, buffer_shape)
        buffer.zero_()
        output = net(x)
        return output

    input = t.randn(batch_sz, 1, *size, device=device, dtype=t.double)
    input = input.double()
    input.requires_grad = True

    o = f(input)
    analytical, reentrant, correct_grad_sizes = get_analytical_jacobian((input,), o)
    assert reentrant

    t.autograd.gradcheck(f, [input], raise_exception=True)


def test_stitch_copy_module():
    size = (2, 2)
    depth = 3
    batch_sz = 2

    for device in [t.device("cpu"), t.device("cuda")]:
        input, target = (
            Variable(t.randn(batch_sz, 1, *size, device=device)) for i in range(2)
        )

        buffer = StitchBuffer()
        buffer_shape = (batch_sz, depth, *size)
        buffer.like_(input, buffer_shape)

        cs = [nn.Conv2d(i + 1, 1, 1) for i in range(depth)]
        fs = [StitchCopyModule(buffer, i) for i in range(depth)]
        net = nn.Sequential(*[val for pair in zip(fs, cs) for val in pair])

        net.to(device)

        output = net(input)

        criterion = nn.L1Loss()
        loss = criterion(output, target)
        loss.backward()
