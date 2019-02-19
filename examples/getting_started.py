"""
A short example to get you started with Mixed-scale Dense Networks for PyTorch
"""
import msd_pytorch as mp
from torch.utils.data import DataLoader
import numpy as np

###############################################################################
#                              Network parameters                             #
###############################################################################
# The number of input channels of the MSD network
c_in = 1
# The depth of the MSD network. Good values range between 30 and 200.
depth = 30
# The width of the MSD network. A value of 1 is recommended.
width = 1
# The dilation scheme to use for the MSD network. The default is [1,
# 2, ..., 10], but [1, 2, 4, 8] is good too.
dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# If using the MSD network for regression, set these parameters too.

# The loss function to use. Can be "L1" or "L2".
loss = "L2"
# The number of output channels of the network.
c_out = 1

###############################################################################
#                              Dataset parameters                             #
###############################################################################

# The glob pattern for the training set input data. For instance: "~/train/input*.tif".
train_input_glob = "~/datasets/MLTestData/train/noisy/*1.tiff"
# The glob pattern for the training set target data. For instance: "~/train/target/img*.png"
train_target_glob = "~/datasets/MLTestData/train/label/*1.tiff"
# The glob pattern for the validation set input data. Can be left
# empty if there is no validation data.
val_input_glob = ""
# The glob pattern for the validation set input data. Can be left
# empty if there is no validation data.
val_target_glob = ""

# If you are doing segmentation, set the values of the label you are
# expecting in the target dataset. You can either set this value to an
# integer `k`, indicating that the label set is {0, 1, ..., k-1}, or
# explicitly set the expected label values using a list, as is done
# below.
labels = [0, 1, 2, 3, 4]

###############################################################################
#                             Training parameters                             #
###############################################################################
# The number of epochs to train for
epochs = 10
# Which task to perform. Can be "regression" or "segmentation"
task = "segmentation"
# The mini-batch size used in training.
batch_size = 3

###############################################################################
#                                 Loading data                                #
###############################################################################

print("Load training dataset")
if task == "regression":
    train_ds = mp.ImageDataset(train_input_glob, train_target_glob)
else:
    train_ds = mp.ImageDataset(train_input_glob, train_target_glob, labels=labels)

train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Load Validation dataset (if specified)
if val_input_glob:
    print("Load validation set")
    if task == "regression":
        val_ds = mp.ImageDataset(val_input_glob, val_target_glob)
    else:
        val_ds = mp.ImageDataset(val_input_glob, val_target_glob, labels=labels)

    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
else:
    print("No validation set loaded")
    val_dl = None


print(f"Create {task} network model")
if task == "regression":
    model = mp.MSDRegressionModel(
        c_in, c_out, depth, width, dilations=dilations, loss=loss
    )
else:
    model = mp.MSDSegmentationModel(
        c_in, train_ds.num_labels, depth, width, dilations=dilations
    )

# The network works best if the input data has mean zero and has a
# standard deviation of 1. To achieve this, we get a rough estimate of
# correction parameters from the training data. These parameters are
# not updated after this step and are stored in the network, so that
# they are not lost when the network is saved to and loaded from disk.
print("Start estimating normalization parameters")
model.set_normalization(train_dl)
print("Done estimating normalization parameters")

print("Starting training...")
best_validation_error = np.inf
validation_error = 0.0

for epoch in range(epochs):
    # Train
    model.train(train_dl, 1)
    # Compute training error
    train_error = model.validate(train_dl)
    print(f"{epoch:05} Training error: {train_error: 0.6f}")
    # Compute validation error
    if val_dl is not None:
        validation_error = model.validate(val_dl)
        print(f"{epoch:05} Validation error: {validation_error: 0.6f}")
    # Save network if worthwile
    if validation_error < best_validation_error or val_dl is None:
        best_validation_error = validation_error
        model.save(f"msd_network_epoch_{epoch}.torch", epoch)

# Save final network parameters
model.save(f"msd_network_epoch_{epoch}.torch", epoch)

# The parameters can be reloaded again:
epoch = model.load(f"msd_network_epoch_{epoch}.torch")
