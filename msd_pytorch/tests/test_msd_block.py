import pytest
from . import torch_equal
import torch
import msd_pytorch.msd_block as msd_block
import msd_pytorch.msd_module as msd_module
from torch.autograd import gradcheck


def test_msd_block():
    # Test forward evaluation
    img = torch.nn.Parameter(torch.ones((1, 1, 1, 5))).cuda()

    model = msd_block.MSDBlock2d(1, [1, 1, 1], width=1).cuda()
    result = model(img)

    print(result)

    # Backward evaluation
    loss = torch.mean(result)
    loss.backward()

    for weight in model.weights:
        print('grad', weight.grad)


@pytest.mark.slow
def test_msd_block_2d_grad_check():
    """Test using gradcheck provided by Torch.

    Because there is no reference implementation to compare to, we
    check if the gradient calculations are consistent with numerical
    calculations.

    """
    dtype = torch.double
    B = 2                   # Batch size
    C_IN = 3                # Input channels
    H = 13                  # Height
    W = 21                  # Width
    dilations = [1, 2, 3]   # Dilation

    net = msd_block.MSDBlock2d(C_IN, dilations).cuda().to(dtype=dtype)
    x = torch.randn(B, C_IN, H, W, dtype=dtype, requires_grad=True).cuda()

    gradcheck(net, [x], raise_exception=True)


@pytest.mark.slow
def test_msd_block_3d_grad_check():
    """Test using gradcheck provided by Torch.

    Because there is no reference implementation to compare to, we
    check if the gradient calculations are consistent with numerical
    calculations.

    """
    dtype = torch.double
    B = 2                   # Batch size
    C_IN = 3                # Input channels
    D, H, W = (5, 7, 11)
    dilations = [1, 2, 3]   # Dilation

    net = msd_block.MSDBlock3d(C_IN, dilations).cuda().to(dtype=dtype)
    x = torch.randn(B, C_IN, D, H, W, dtype=dtype, requires_grad=True).cuda()

    gradcheck(net, [x], raise_exception=True)
