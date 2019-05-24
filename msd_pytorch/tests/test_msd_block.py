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


def copy_weights(module, module2d):
    width = module.width
    assert module.width == module2d.width
    block = module2d.msd_block

    for i, hl in enumerate(module.hidden_layers):
        block.weights[i].data[:] = hl.convolution.weight
        block.bias.data[i * width:(i + 1) * width] = hl.convolution.bias

    module2d.final_layer.linear.weight.data[:] = module.c_final.linear.weight.data
    module2d.final_layer.linear.bias.data[:] = module.c_final.linear.bias.data


def init_weights_for_testing(module):
    for i, hl in enumerate(module.hidden_layers):
        hl.convolution.weight.data.normal_(0, 1)
        hl.convolution.bias.data.normal_(0, 1)
        module.c_final.linear.weight.data.normal_(0, 1)
        module.c_final.linear.bias.data.normal_(0, 1)


def assert_grads_equal(module, module2d):
    block = module2d.msd_block
    width = module.width
    for i, hl in enumerate(module.hidden_layers):
        assert torch_equal(block.weights[i].grad, hl.convolution.weight.grad)
        assert torch_equal(block.bias.grad[i * width:(i + 1) * width], hl.convolution.bias.grad)

    assert torch_equal(module2d.final_layer.linear.weight.grad, module.c_final.linear.weight.grad)
    assert torch_equal(module2d.final_layer.linear.bias.grad, module.c_final.linear.bias.grad)


def test_compare_msd_module():
    dtype = torch.float    #
    device = torch.device("cuda")
    B = 2                   # Batch size
    C_IN = 3                # Input channels
    C_OUT = 2               # Output channels
    H = 13                  # Height
    W = 21                  # Width
    dilations = [1, 5, 3]   # Dilation
    depth = 10
    width = 2

    # Input
    with_grad = dict(requires_grad=True, device=device, dtype=dtype)
    no_grad = dict(requires_grad=False, device=device, dtype=dtype)
    x1 = torch.randn(B, C_IN, H, W, **with_grad)
    x2 = x1.clone()
    tgt = torch.randn(B, C_OUT, H, W, **no_grad)

    # Models
    m1 = msd_module.MSDModule(C_IN, C_OUT, depth, width, dilations).to(device)
    m2 = msd_block.MSDModule2d(C_IN, C_OUT, depth, width, dilations).to(device)

    # Output
    init_weights_for_testing(m1)
    copy_weights(m1, m2)

    o1 = m1(x1)
    o2 = m2(x2)
    l1 = torch.nn.MSELoss()(o1, tgt)
    l2 = torch.nn.MSELoss()(o2, tgt)
    l1.backward(torch.ones_like(o1))
    l2.backward(torch.ones_like(o2))

    assert torch_equal(o1, o2)
    assert_grads_equal(m1, m2)


def test_grad_check():
    torch.manual_seed(1)

    dtype = torch.double
    size = (11, 13)
    batch_sz = 2

    for depth in [9]:
        print(f"Depth: {depth}")
        width = c_in = c_out = batch_sz
        x = torch.randn(batch_sz, c_in, *size, dtype=dtype).cuda()
        x.requires_grad = True

        net = msd_block.MSDModule2d(c_in, c_out, depth, width)
        net.cuda()
        net.double()

        for p in net.parameters():
            p.data = torch.randn_like(p.data)

        assert net is not None
        gradcheck(net, [x], raise_exception=True, atol=1e-4, rtol=1e-3)
