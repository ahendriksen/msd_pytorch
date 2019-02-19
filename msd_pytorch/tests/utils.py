from pytest import approx


def torch_equal(x, y):
    return abs(to_numpy(x)).sum() == approx(abs(to_numpy(y)).sum())


def to_numpy(x):
    return x.detach().cpu().numpy()
