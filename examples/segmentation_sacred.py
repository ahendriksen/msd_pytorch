"""This example shows how to construct a Sacred experiment with the
   MSD network for segmentation.
"""
import re
from msd_pytorch.TiffDataset import TiffDataset
from msd_pytorch.msd_seg_model import (MSDSegmentationModel, msd_ingredient)
from os import environ
from sacred import Experiment
from sacred.observers import MongoObserver
from timeit import default_timer as timer
from torch.utils.data import DataLoader
import glob
import os.path
import numpy as np

ex = Experiment('MSD_pytorch segmentation example',
                ingredients=[msd_ingredient])

mongo_user = environ.get('MONGO_SACRED_USER')
mongo_pass = environ.get('MONGO_SACRED_PASS')
mongo_host = environ.get('MONGO_SACRED_HOST')

assert mongo_user, 'Setting $MONGO_USER is required'
assert mongo_pass, 'Setting $MONGO_PASS is required'
assert mongo_host, 'Setting $MONGO_HOST is required'

mongo_url = 'mongodb://{0}:{1}@{2}:27017/sacred?authMechanism=SCRAM-SHA-1'.format(
    mongo_user, mongo_pass, mongo_host)

ex.observers.append(MongoObserver.create(url=mongo_url, db_name='sacred'))

# The default dataset has 5 labels. Make sure that this example can be
# run without extra command line parameters.


@msd_ingredient.config
def update_msd_config():
    c_in = 1
    num_labels = 5


@ex.config
def config(msd):
    # Set parameters
    epochs = 1                      # The number of epochs to train for
    batch_size = 1                  # The mini-batch size
    # (1 is strongly recommended for msd_pytorch)

    dataset_dir = "~/datasets/"     # Dataset directory (may contain a '~')
    # Training directory (relative to dataset directory)
    train_dir = "MLTestData/train/"
    # Validation directory (relative to dataset directory)
    val_dir = "MLTestData/val/"
    # Glob for input images (relative to training directory)
    train_inp_glob = "image/*.tiff"
    # Glob for target images (relative to training directory)
    train_tgt_glob = "label/*.tiff"
    # Glob for input images (relative to validation directory)
    val_inp_glob = "image/*.tiff"
    # Glob for target images (relative to validation directory)
    val_tgt_glob = "label/*.tiff"


def split_by_number(x):
    # https://code.activestate.com/recipes/135435-sort-a-string-using-numeric-order/
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]


def load_dataset(dataset_dir, sub_dir, input_glob, target_glob):
    dataset_dir = os.path.expanduser(dataset_dir)
    dataset_dir = os.path.realpath(dataset_dir)
    sub_dir = os.path.join(dataset_dir, sub_dir)
    # https://code.activestate.com/recipes/135435-sort-a-string-using-numeric-order/
    inp_imgs = sorted(glob.glob(os.path.join(
        sub_dir, input_glob)), key=split_by_number)
    tgt_imgs = sorted(glob.glob(os.path.join(
        sub_dir, target_glob)), key=split_by_number)

    return TiffDataset('', '', input_imgs=inp_imgs, target_imgs=tgt_imgs)


@ex.command
@ex.capture()
def stats(batch_size, dataset_dir, train_dir, val_dir, train_inp_glob,
          train_tgt_glob, val_inp_glob, val_tgt_glob, msd):
    # Load datasets
    train_ds = load_dataset(dataset_dir, train_dir,
                            train_inp_glob, train_tgt_glob)
    val_ds = load_dataset(dataset_dir, val_dir, val_inp_glob, val_tgt_glob)

    # Create dataloaders, which batch and shuffle the data:
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)

    print('Statistics for training dataset')
    print_stats(msd, train_dl)
    print('Statistics for validation dataset')
    print_stats(msd, val_dl)


def print_stats(msd, dl):
    print("  Sample size: {}".format(len(dl)))
    print("  Label statistics: ")

    if isinstance(msd['num_labels'], list):
        labels = msd['num_labels']
    else:
        labels = range(msd['num_labels'])

    for label in labels:
        density = 0
        for (_, target) in dl:
            density += (target == label).sum() / target.numel()
        density /= len(dl)
        print("    {:03}: {:02.2f}%".format(label, density * 100))
    print("  Input statistics: ")
    mean_in = 0
    square_in = 0
    for (input, _) in dl:
        mean_in += input.mean()
        square_in += input.pow(2).mean()
    mean_in /= len(dl)
    square_in /= len(dl)
    std_in = np.sqrt(square_in - mean_in ** 2)
    print("    mean:    {}".format(mean_in))
    print("    std dev: {}".format(std_in))


@ex.automain
def main(msd, epochs, batch_size, dataset_dir,
         train_dir, val_dir, train_inp_glob, train_tgt_glob, val_inp_glob,
         val_tgt_glob):
    # Load datasets
    train_ds = load_dataset(dataset_dir, train_dir, train_inp_glob,
                            train_tgt_glob)
    val_ds = load_dataset(dataset_dir, val_dir, val_inp_glob, val_tgt_glob)

    # Create dataloaders, which batch and shuffle the data:
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)

    # Create model:
    model = MSDSegmentationModel(msd['c_in'], msd['num_labels'])

    # The network works best if the input data has mean zero and has a
    # standard deviation of 1. To achieve this, we get a rough estimate of
    # correction parameters from the training data. These parameters are
    # not updated after this step and are stored in the network, so that
    # they are not lost when the network is saved to and loaded from disk.
    model.set_normalization(train_dl)

    # train for some epochs:
    print("Training...")
    best_validation_error = model.validate(val_dl)
    print("Initial validation error: {}".format(best_validation_error))

    for epoch in range(epochs):
        start = timer()
        # Train
        model.train(train_dl, 1)
        # Compute training error
        ex.log_scalar("Training error", model.validate(train_dl))
        # Compute validation error
        validation_error = model.validate(val_dl)
        print("Validation error: {}".format(validation_error))
        ex.log_scalar("Validation error", validation_error)
        # Save network if worthwile
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            model.save_network('.', 'segnet', epoch)
            ex.add_artifact(model.get_network_path('.', 'segnet', epoch))

        end = timer()
        ex.log_scalar("Iteration time", end - start)

    # Always save final network parameters
    model.save_network('.', 'regnet', epoch)
    ex.add_artifact(model.get_network_path('.', 'regnet', epoch))
