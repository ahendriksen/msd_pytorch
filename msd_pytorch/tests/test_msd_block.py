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
