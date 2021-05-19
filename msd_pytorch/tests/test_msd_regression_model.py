import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from msd_pytorch.msd_regression_model import MSDRegressionModel
from . import torch_equal


def test_params_change_3d():
    """Ensure that learning updates all parameters.
    """
    # make test repeatable
    torch.manual_seed(1)

    c_in, c_out, depth, width = 1, 1, 11, 1
    model = MSDRegressionModel(
        c_in, c_out, depth, width, loss="L1", dilations=[1, 2, 4, 8], ndim=3,
    )
    shape = (17, 11, 13)

    input = torch.randn(1, c_in, *shape)  # batch size is one.
    target = torch.randn(1, c_out, *shape)

    params0 = [p.data.clone() for p in model.optimizer.param_groups[0]["params"]]

    model.set_input(input)
    model.set_target(target)
    for i in range(10):
        model.learn(input, target)

    params1 = [p.data.clone() for p in model.optimizer.param_groups[0]["params"]]

    for p0, p1 in zip(params0, params1):
        assert not torch_equal(p0, p1)


def test_params_change():
    """Ensure that learning updates all parameters.
    """
    # make test repeatable
    torch.manual_seed(1)

    c_in, c_out, depth, width = 1, 1, 11, 1
    model = MSDRegressionModel(
        c_in, c_out, depth, width, loss="L1", dilations=[1, 2, 4, 8]
    )
    shape = (13, 14)

    input = torch.randn(1, c_in, *shape)  # batch size is one.
    target = torch.randn(1, c_out, *shape)

    params0 = [p.data.clone() for p in model.optimizer.param_groups[0]["params"]]

    model.set_input(input)
    model.set_target(target)
    for i in range(10):
        model.learn(input, target)

    params1 = [p.data.clone() for p in model.optimizer.param_groups[0]["params"]]

    for p0, p1 in zip(params0, params1):
        assert not torch_equal(p0, p1)


def test_data_parallel():
    """Check that msd_model is compatible with multi-GPU approaches

    Specifically, `torch.nn.DataParallel`.
    """

    shape = (100, 100)
    inp = torch.zeros(4, 1, *shape, dtype=torch.float32, device=torch.device("cuda:0"))
    tgt = torch.zeros(4, 1, *shape, dtype=torch.float32, device=torch.device("cuda:0"))

    model = MSDRegressionModel(1, 1, 11, 1, parallel=True)
    model.forward(inp, tgt)
    model.learn(inp, tgt)


def test_api_surface(tmp_path):
    ###########################################################################
    #                              Create network                             #
    ###########################################################################
    batch_size = 1
    c_in, c_out, depth, width = 1, 1, 11, 1
    model = MSDRegressionModel(
        c_in, c_out, depth, width, dilations=[1, 2, 3], loss="L1"
    )

    with pytest.raises(ValueError):
        model = MSDRegressionModel(
            c_in, c_out, depth, width, dilations=[1, 2, 3], loss="invalid"
        )

    # Data
    N, shape = 20, (30, 30)
    d_in = torch.randn(N, c_in, *shape)
    d_out = torch.randn(N, c_out, *shape)
    ds = TensorDataset(d_in, d_out)
    dl = DataLoader(ds, batch_size)

    ###########################################################################
    #                            low-level methods                            #
    ###########################################################################
    (inp, tgt), *_ = dl
    model.set_input(inp)
    model.set_target(tgt)
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
    epoch_saved = 93
    model.save(tmp_path / "network.torch", epoch_saved)

    for p in model.net.parameters():
        p.data.fill_(18.74)

    model = MSDRegressionModel(
        c_in, c_out, depth, width, dilations=[1, 2, 3], loss="L2"
    )
    epoch_loaded = model.load(tmp_path / "network.torch")
    params1 = dict((n, p.data.clone()) for n, p in model.net.named_parameters())

    for n in params0.keys():
        p0, p1 = params0[n], params1[n]
        assert torch_equal(p0, p1)

    assert epoch_saved == epoch_loaded
