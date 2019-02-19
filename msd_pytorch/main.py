import logging
from os import environ
from sacred import Experiment, Ingredient
from timeit import default_timer as timer
from timeit import Timer
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn import MSELoss
import msd_pytorch as mp
from .bench import bench

msd_ingredient = Ingredient("msd")
ex = Experiment("MSD", ingredients=[msd_ingredient])

mongo_enabled = environ.get("MONGO_SACRED_ENABLED")
mongo_user = environ.get("MONGO_SACRED_USER")
mongo_pass = environ.get("MONGO_SACRED_PASS")
mongo_host = environ.get("MONGO_SACRED_HOST")

if mongo_enabled == "true":
    from sacred.observers import MongoObserver

    assert mongo_user, "Setting $MONGO_USER is required"
    assert mongo_pass, "Setting $MONGO_PASS is required"
    assert mongo_host, "Setting $MONGO_HOST is required"

    mongo_url = "mongodb://{0}:{1}@{2}:27017/sacred?authMechanism=SCRAM-SHA-1".format(
        mongo_user, mongo_pass, mongo_host
    )

    ex.observers.append(MongoObserver.create(url=mongo_url, db_name="sacred"))


@msd_ingredient.config
def msd_config():
    c_in = 1                     # Number of input channels
    c_out = 1                    # Number of output channels (for regression; see `labels` for segmentation)
    depth = 10                   # The depth of the network
    width = 1                    # The width of the network
    dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # The dilation-scheme that is used in the network
    loss = 'L2'                  # Which loss to use for regression (options: "L1" or "L2")


@ex.config
def ex_config():
    epochs = 1             # The number of epochs to train for
    batch_size = 1         # The mini-batch size
    train_input_glob = ""  # The glob pattern for the training set input data
    train_target_glob = "" # The glob pattern for the training set target data
    val_input_glob = ""    # The glob pattern for the validation set input data
    val_target_glob = ""   # The glob pattern for the validation set input data

    # The labels that you expect in your segmentation targets (if you are doing segmentation)
    labels = [0, 1]
    # Benchmark specific
    input_size = 1024


@ex.command
def segmentation(
    msd,
    epochs,
    labels,
    batch_size,
    train_input_glob,
    train_target_glob,
    val_input_glob,
    val_target_glob,
):
    logging.info("Load training dataset")
    # Create train (always) and validation (only if specified) datasets.
    train_ds = mp.ImageDataset(train_input_glob, train_target_glob, labels=labels)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    if val_input_glob:
        logging.info("Load validation set")
        val_ds = mp.ImageDataset(val_input_glob, val_target_glob, labels=labels)
        val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    else:
        logging.info("No validation set loaded")
        val_dl = None

    logging.info("Create network model")
    model = mp.MSDSegmentationModel(num_labels=train_ds.num_labels)
    train(model, epochs, train_dl, val_dl)


@ex.command
def regression(
    msd,
    epochs,
    batch_size,
    train_input_glob,
    train_target_glob,
    val_input_glob,
    val_target_glob,
):
    logging.info("Load training dataset")
    # Create train (always) and validation (only if specified) datasets.
    train_ds = mp.ImageDataset(train_input_glob, train_target_glob)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    if val_input_glob:
        logging.info("Load validation set")
        val_ds = mp.ImageDataset(val_input_glob, val_target_glob)
        val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    else:
        logging.info("No validation set loaded")
        val_dl = None

    logging.info("Create network model")
    model = mp.MSDRegressionModel()
    train(model, epochs, train_dl, val_dl)


@ex.command
def benchmark(msd, batch_size, input_size):
    inp = torch.zeros(batch_size, msd["c_in"], input_size, input_size).cuda()
    tgt = torch.zeros(batch_size, msd["c_out"], input_size, input_size).cuda()

    model = mp.MSDRegressionModel()
    t = Timer(stmt="model.net(inp)", globals={"inp": inp, "model": model})
    print(bench("Forward", t))

    mse = MSELoss()
    t = Timer(
        stmt=("result = model.net(inp);" "loss = mse(result, tgt);" "loss.backward()"),
        globals={"inp": inp, "tgt": tgt, "mse": mse, "model": model},
    )
    print(bench("Gradient", t))


def train(model, epochs, train_dl, val_dl):
    # The network works best if the input data has mean zero and has a
    # standard deviation of 1. To achieve this, we get a rough estimate of
    # correction parameters from the training data. These parameters are
    # not updated after this step and are stored in the network, so that
    # they are not lost when the network is saved to and loaded from disk.
    logging.info("Start estimating normalization parameters")
    model.set_normalization(train_dl)
    logging.info("Done estimating normalization parameters")

    logging.info("Starting training...")
    best_validation_error = np.inf
    validation_error = 0.0

    for epoch in range(epochs):
        start = timer()
        # Train
        model.train(train_dl, 1)
        # Compute training error
        train_error = model.validate(train_dl)
        ex.log_scalar("Training error", train_error)
        logging.info(f"{epoch:05} Training error: {train_error: 0.6f}")
        # Compute validation error
        if val_dl is not None:
            validation_error = model.validate(val_dl)
            ex.log_scalar("Validation error", validation_error)
            logging.info(f"{epoch:05} Validation error: {validation_error: 0.6f}")
        # Save network if worthwile
        if validation_error < best_validation_error or val_dl is None:
            best_validation_error = validation_error
            model.save(f"msd_network_epoch_{epoch}.torch", epoch)
            ex.add_artifact(f"msd_network_epoch_{epoch}.torch")

        end = timer()
        ex.log_scalar("Iteration time", end - start)
        logging.info(f"{epoch:05} Iteration time: {end-start: 0.6f}")

    # Always save final network parameters
    model.save(f"msd_network_epoch_{epoch}.torch", epoch)
    ex.add_artifact(f"msd_network_epoch_{epoch}.torch")


@ex.main
def experiment_main():
    print(
        """
MSD
---

If you have input and target images in directories
`./train/input/` and `./train/target/`, then you can train a network
to do regression with the following command in your terminal:

> msd regression -p with train_input_glob='./train/input/*' train_target_glob='./train/target/*' epochs=10 msd.depth=30

Similarly, segmentation is possible using the following command:

> msd segmentation -p with train_input_glob='./train/input/*' train_target_glob='./train/target/*' epochs=10 msd.depth=30 labels=[0,1,2,3]

More command-line arguments are available

epochs = 1                         # The number of epochs to train for
labels = [0, 1]                    # The labels that you expect in your segmentation targets (if you are doing segmentation)
train_input_glob = ''              # The glob pattern for the training set input data
train_target_glob = ''             # The glob pattern for the training set target data
val_input_glob = ''                # The glob pattern for the validation set input data
val_target_glob = ''               # The glob pattern for the validation set input data
msd:
  c_in = 1                         # Number of input channels
  c_out = 1                        # Number of output channels (for regression; see `labels` for segmentation)
  depth = 10                       # The depth of the network
  width = 1                        # The width of the network
  dilations = [1, 2, ..., 10]      # The dilation-scheme that is used in the network
  loss = 'L2'                      # Which loss to use for regression (options: "L1" or "L2")


The path specification for the images is a path with optional glob
pattern describing the image file paths. Tildes and other HOME
directory specifications are expanded with `os.path.expanduser` and
symlinks are resolved.

If the path points to a directory, then all files in the directory are
included. If the path points to file, then that single file is
included.

Alternatively, one may specify a "glob pattern" to match
specific files in the directory.

Examples:

* "~/train_images/"
* "~/train_images/cats*.png"
* "~/train_images/*.tif"
* "~/train_images/scan*"
* "~/train_images/just_one_image.jpeg"
"""
    )


def main_function():
    mp.MSDRegressionModel = msd_ingredient.capture(mp.MSDRegressionModel)
    mp.MSDSegmentationModel = msd_ingredient.capture(mp.MSDSegmentationModel)

    ex.run_commandline()
