import torch
from msd_pytorch.msd_model import MSDModel
from torch.utils.data import TensorDataset, DataLoader
from pytest import approx


def test_normalization():
    torch.manual_seed(1)

    means = [-10, 0, 10]
    stds = [1, 10]

    for mean in means:
        for std in stds:
            # Network
            width = 1
            depth = 10
            sample_size, num_channels, shape = 1, 1, (1000, 1000)
            model = MSDModel(num_channels, num_channels, depth, width)

            # Data
            d_in = mean + torch.randn(sample_size, num_channels, *shape) * std
            d_out = mean + torch.randn(sample_size, num_channels, *shape) * std
            ds = TensorDataset(d_in, d_out)
            dl = DataLoader(ds, sample_size)

            # Calculate normalization:
            model.set_normalization(dl)

            (input, _), *_ = dl

            # Check input layer scaling
            l0 = model.scale_in(input)
            assert l0.mean().item() == approx(0, abs=1e-4, rel=1e-4)
            assert l0.std().item() == approx(1.0, rel=1e-4)

            # Check output layer scaling
            l1 = model.scale_out(l0)
            assert l1.mean().item() == approx(mean, abs=1e-2, rel=1e-2)
            assert l1.std().item() == approx(std, rel=1e-2)
