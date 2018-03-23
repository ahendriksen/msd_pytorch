"""This example shows how to train a network to do regression with
   msd_pytorch. We train a network to denoise 2d images.
"""


from msd_pytorch.TiffDataset import TiffDataset
from msd_pytorch.msd_reg_model import (MSDRegressionModel)
from timeit import default_timer as timer
from torch.utils.data import DataLoader
import glob
import os.path

# Set parameters
in_channels = 1                 # number of input channels
out_channels = 1                # Number of output channels
depth = 30                      # Depth of the MSD network
width = 1                       # Width of the MSD network
epochs = 1                      # The number of epochs to train for
batch_size = 1                  # The mini-batch size
                                # (1 is strongly recommended for msd_pytorch)

# Training dataset
dataset_dir = os.path.expanduser("~/datasets")
train_dir = os.path.join(dataset_dir, "MLTestData/train/")

inp_imgs = sorted(glob.glob(train_dir + "noisy/*.tiff"))
tgt_imgs = sorted(glob.glob(train_dir + "image/*.tiff"))
train_ds = TiffDataset('', '', input_imgs=inp_imgs, target_imgs=tgt_imgs)

# Validation dataset
val_dir = os.path.join(dataset_dir, "MLTestData/val/")
inp_imgs = sorted(glob.glob(val_dir + "noisy/*.tiff"))
tgt_imgs = sorted(glob.glob(val_dir + "image/*.tiff"))
val_ds = TiffDataset('', '', input_imgs=inp_imgs, target_imgs=tgt_imgs)

# Create dataloaders, which batch and shuffle the data:
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size, shuffle=True)

# Create model:
model = MSDRegressionModel(in_channels, out_channels, depth, width,
                           "L2", "MSD", conv3d=False)

# The network works best if the input data has mean zero and has a
# standard deviation of 1. To achieve this, we get a rough estimate of
# correction parameters from the training data. These parameters are
# not updated after this step and are stored in the network, so that
# they are not lost when the network is saved to and loaded from disk.
model.set_normalization(train_dl)

# Print how a random network performs on the validation dataset:
print("Initial validation loss: {}".format(model.validate(val_dl)))

# Try loading a precomputed network:
try:
    model.load_network('.', 'regnet', 1)
    print("Loaded validation loss: {}".format(model.validate(val_dl)))
except:
    pass

# train for some epochs:
print("Training...")
start = timer()
model.train(train_dl, epochs)
end = timer()

print("Validation loss: {}".format(model.validate(val_dl)))
print("Total time:      {}".format(end - start))

# Save network:
model.save_network('.', 'regnet', 1)
