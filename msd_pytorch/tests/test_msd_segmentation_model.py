import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from msd_pytorch.msd_segmentation_model import MSDSegmentationModel
from . import torch_equal


def test_params_change():
    """Ensure that learning updates all parameters.
    """
    # make test repeatable
    torch.manual_seed(1)

    c_in, num_labels, depth, width = 1, 3, 11, 1
    model = MSDSegmentationModel(c_in, num_labels, depth, width, dilations=[1, 2, 3])
    shape = (13, 14)

    input = torch.randn(1, c_in, *shape)  # batch size is one.
    target = torch.randint(low=0, high=num_labels, size=(1, 1, *shape))

    params0 = [p.data.clone() for p in model.optimizer.param_groups[0]["params"]]

    model.set_input(input)
    model.set_target(target)
    for i in range(10):
        model.learn(input, target)

    params1 = [p.data.clone() for p in model.optimizer.param_groups[0]["params"]]

    for p0, p1 in zip(params0, params1):
        assert not torch_equal(p0, p1)


def test_api_surface(tmp_path):
    ###########################################################################
    #                              Create network                             #
    ###########################################################################
    batch_size = 1
    c_in, depth, width = 1, 11, 1
    num_labels = 4
    model = MSDSegmentationModel(c_in, num_labels, depth, width)

    # Data
    N, shape = 20, (30, 30)
    d_in = torch.randn(N, c_in, *shape)
    d_out = torch.randint(low=0, high=num_labels, size=(N, 1, *shape))
    ds = TensorDataset(d_in, d_out)
    dl = DataLoader(ds, batch_size)

    ###########################################################################
    #                            low-level methods                            #
    ###########################################################################
    (inp, tgt), *_ = dl
    model.set_input(inp)
    model.set_target(tgt)
    with pytest.raises(ValueError):
        model.set_target(tgt + 1)
    with pytest.raises(ValueError):
        model.set_target(tgt * -1)

    model.forward(inp, tgt)
    model.learn(inp, tgt)

    ###########################################################################
    #                         Methods with dataloader                         #
    ###########################################################################
    # High-level methods using the dataloader:
    model.set_normalization(dl)
    model.train(dl, 1)

    ###########################################################################
    #                        Get floating point losses                        #
    ###########################################################################
    assert isinstance(model.validate(dl), float)
    assert isinstance(model.get_loss(), float)

    ###########################################################################
    #                         Test saving and loading                         #
    ###########################################################################
    params0 = dict((n, p.data.clone()) for n, p in model.net.named_parameters())
    epoch_saved = 94
    model.save(tmp_path / "network.torch", epoch_saved)

    for p in model.net.parameters():
        p.data.fill_(18.74)

    model = MSDSegmentationModel(c_in, num_labels, depth, width)
    epoch_loaded = model.load(tmp_path / "network.torch")
    params1 = dict((n, p.data.clone()) for n, p in model.net.named_parameters())

    for n in params0.keys():
        p0, p1 = params0[n], params1[n]
        assert torch_equal(p0, p1)

    assert epoch_saved == epoch_loaded
