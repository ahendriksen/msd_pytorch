import msd_pytorch.image_dataset as img_ds
import logging
import tifffile
import imageio
import numpy as np
import pytest
from msd_pytorch.errors import InputError
from . import torch_equal


def test_convert_to_integral():
    img = np.ones(10)

    assert np.mean(img_ds._convert_to_integral(img.astype(np.int8))) == 1
    assert np.mean(img_ds._convert_to_integral(img.astype(np.bool))) == 1
    np.mean(img_ds._convert_to_integral((-img).astype(np.int8)))

    for t in [np.float16, np.float32, np.float64]:
        with pytest.raises(InputError):
            img_ds._convert_to_integral(img.astype(t))


def test_relabel_image():
    img0 = np.ones(10, dtype=np.uint8)
    assert img_ds._relabel_image(img0, 2).mean() == 1

    img = img0 - 2
    with pytest.raises(InputError):
        img_ds._relabel_image(img, 2)

    img = img0 + 2
    with pytest.raises(InputError):
        img_ds._relabel_image(img, 2)
    with pytest.raises(InputError):
        img_ds._relabel_image(img, [2]).mean()

    assert img_ds._relabel_image(img, [3]).mean() == 0
    assert img_ds._relabel_image(img, 4).mean() == 3


def test_empty_stack(tmp_path, caplog):
    """Test that image stack with non existing path logs a warning.
    """

    img_ds.ImageStack(tmp_path / "non_existing.tif")
    img_ds.ImageStack(tmp_path / "non_existing*.tif")

    assert len(caplog.record_tuples) == 2
    assert np.all(
        [
            level == logging.WARNING and "empty" in msg
            for (_, level, msg) in caplog.record_tuples
        ]
    )


def test_stack_no_glob(tmp_path):
    tifffile.imsave(tmp_path / "image.tif", np.ones((10, 10)))
    stack = img_ds.ImageStack(tmp_path)
    assert len(stack) == 1


def test_image_stack(tmp_path):
    num_channels = 3
    for i in range(10):
        img = np.random.randn(num_channels, 10, 10)
        imageio.imsave(tmp_path / f"imio_{i}.png", img.swapaxes(0, 2))
        imageio.imsave(tmp_path / f"imio_{i}.jpeg", img.swapaxes(0, 2))
        tifffile.imsave(tmp_path / f"tiff_{i}.tif", img)
        tifffile.imsave(tmp_path / f"tiff2d_{i}.tif", img[0])
        tifffile.imsave(tmp_path / f"tiff0_{i:05}.tif", img)

    pngs = img_ds.ImageStack(tmp_path / "imio_*.png")
    pngs_c = img_ds.ImageStack(tmp_path / "imio_*.png", collapse_channels=True)
    jpegs = img_ds.ImageStack(tmp_path / "imio_*.jpeg")
    jpegs_c = img_ds.ImageStack(tmp_path / "imio_*.jpeg", collapse_channels=True)
    utiffs = img_ds.ImageStack(tmp_path / "tiff_*.tif")
    utiffs_2d = img_ds.ImageStack(tmp_path / "tiff2d_*.tif")
    otiffs = img_ds.ImageStack(tmp_path / "tiff0_*.tif")

    # This image stack should have no labels:
    with pytest.raises(RuntimeError):
        pngs.num_labels

    n = len(pngs)
    assert n == len(pngs_c)
    assert n == len(jpegs)
    assert n == len(jpegs_c)
    assert n == len(utiffs)
    assert n == len(otiffs)
    for i in range(n):
        assert pngs[i].dim() == 3
        assert pngs[i].shape[0] == num_channels
        assert pngs_c[i].shape[0] == 1
        assert jpegs[i].dim() == 3
        assert jpegs[i].shape[0] == num_channels
        assert jpegs_c[i].shape[0] == 1

        assert utiffs_2d[i].dim() == 3
        assert torch_equal(otiffs[i], utiffs[i])


def test_image_stack_labels(tmp_path):
    num_channels = 1
    num_labels = 10
    for i in range(10):
        img = np.random.randint(num_labels, size=(num_channels, 10, 10), dtype=np.uint8)
        tifffile.imsave(tmp_path / f"tiff_{i}.tif", img)
        tifffile.imsave(tmp_path / f"tiff2d_{i}.tif", img[0])
        tifffile.imsave(tmp_path / f"tiff0_{i:05}.tif", img)

    # Check for errors when there are missing labels:
    with pytest.raises(InputError):
        list(img_ds.ImageStack(tmp_path / "tiff0_*.tif", labels=range(9)))
    with pytest.raises(InputError):
        list(img_ds.ImageStack(tmp_path / "tiff0_*.tif", labels=9))

    # Check that all stacks have exactly the same values
    utiffs = img_ds.ImageStack(tmp_path / "tiff_*.tif", labels=num_labels)
    utiffs_2d = img_ds.ImageStack(tmp_path / "tiff2d_*.tif", labels=num_labels)
    otiffs = img_ds.ImageStack(tmp_path / "tiff0_*.tif", labels=range(num_labels))
    stacks = [utiffs, utiffs_2d, otiffs]

    n = len(utiffs)
    for s in stacks:
        assert n == len(s)

    for i in range(n):
        for s in stacks:
            assert s[i].dim() == 3
            assert s[i].shape[0] == num_channels
            assert torch_equal(otiffs[i], s[i])
            assert s.num_labels == num_labels


def test_image_dataset(tmp_path):
    num_channels = 3
    for i in range(10):
        for purpose in ["train", "val"]:
            img = np.random.randn(num_channels, 10, 10)
            imageio.imsave(tmp_path / f"{purpose}_imio_{i}.png", img.swapaxes(0, 2))
            imageio.imsave(tmp_path / f"{purpose}_imio_{i}.jpeg", img.swapaxes(0, 2))
            tifffile.imsave(tmp_path / f"{purpose}_tiff_{i}.tif", img)
            tifffile.imsave(tmp_path / f"{purpose}_tiff2d_{i}.tif", img[0])
            tifffile.imsave(tmp_path / f"{purpose}_tiff0_{i:05}.tif", img)

    pngs = img_ds.ImageDataset(
        tmp_path / "train_imio_*.png", tmp_path / "val_imio_*.png"
    )
    pngs_c = img_ds.ImageDataset(
        tmp_path / "train_imio_*.png",
        tmp_path / "val_imio_*.png",
        collapse_channels=True,
    )
    jpegs = img_ds.ImageDataset(
        tmp_path / "train_imio_*.jpeg", tmp_path / "val_imio_*.jpeg"
    )
    jpegs_c = img_ds.ImageDataset(
        tmp_path / "train_imio_*.jpeg",
        tmp_path / "val_imio_*.jpeg",
        collapse_channels=True,
    )
    utiffs = img_ds.ImageDataset(
        tmp_path / "train_tiff_*.tif", tmp_path / "val_tiff_*.tif"
    )
    utiffs_2d = img_ds.ImageDataset(
        tmp_path / "train_tiff2d_*.tif", tmp_path / "val_tiff2d_*.tif"
    )
    otiffs = img_ds.ImageDataset(
        tmp_path / "train_tiff0_*.tif", tmp_path / "val_tiff0_*.tif"
    )

    # check that image stack has no labels:
    with pytest.raises(RuntimeError):
        pngs.num_labels

    n = len(pngs)
    assert n == len(pngs_c)
    assert n == len(jpegs)
    assert n == len(jpegs_c)
    assert n == len(utiffs)
    assert n == len(otiffs)
    for i in range(n):
        for j in [0, 1]:
            assert pngs[i][j].dim() == 3
            assert pngs[i][j].shape[0] == num_channels
            assert pngs_c[i][j].shape[0] == 1
            assert jpegs[i][j].dim() == 3
            assert jpegs[i][j].shape[0] == num_channels
            assert jpegs_c[i][j].shape[0] == 1

            assert utiffs_2d[i][j].dim() == 3
            assert torch_equal(otiffs[i][j], utiffs[i][j])
