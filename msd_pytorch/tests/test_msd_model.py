import torch
from msd_pytorch.msd_model import MSDModel
from torch.utils.data import TensorDataset, DataLoader


def test_normalization():
    torch.manual_seed(1)

    means = [-10, 0, 10]
    stds = [1, 10]

    for mean in means:
        for std in stds:
            # Network
            width = 1
            depth = 10
            sample_size, num_channels, shape = 1, 3, (100, 100)
            model = MSDModel(num_channels, num_channels, depth, width)

            # Data
            # We make the input and target dataset equal, which makes
            # their mean and std equal. This should ensure that the
            # scale_in and scale_out module are each others' inverse.
            d_in = mean + torch.randn(sample_size, num_channels, *shape) * std
            d_out = d_in.clone()
            ds = TensorDataset(d_in, d_out)
            dl = DataLoader(ds, sample_size)

            # Calculate normalization:
            model.set_normalization(dl)

            # Ensure that scale_out is the inverse of scale_in:
            x = torch.randn_like(d_in)
            y = model.scale_out(model.scale_in(x))
            assert abs(x - y).mean() < 1e-4
