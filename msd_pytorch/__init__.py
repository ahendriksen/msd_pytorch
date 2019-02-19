# -*- coding: utf-8 -*-

"""Top-level package for Mixed-scale Dense Networks for PyTorch."""

__author__ = """Allard Hendriksen"""
__email__ = "allard.hendriksen@cwi.nl"


def __get_version():
    import os.path

    version_filename = os.path.join(os.path.dirname(__file__), "VERSION")
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()


import msd_pytorch.image_dataset
import msd_pytorch.errors
from .image_dataset import ImageDataset
from .msd_regression_model import MSDRegressionModel
from .msd_segmentation_model import MSDSegmentationModel
