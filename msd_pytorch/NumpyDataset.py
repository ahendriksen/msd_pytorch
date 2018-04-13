from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch as t
import os
import os.path


def get_image_names(dir, allowed_extensions):
    # This does not work when part of the path is a symbolic link.
    # assert os.path.isdir(dir), "{} is not a valid directory".format(dir)

    images = set()
    for root, _, fs in sorted(os.walk(dir)):
        for f in fs:
            is_image = any([f.endswith(ext) for ext in allowed_extensions])

            if is_image:
                images.add(f)

    return images


def to_tensor(img_path):
    npa = np.load(img_path)
    # uint16 is not supported by pytorch. We must cast the array to
    # float32.
    if npa.dtype == np.dtype('uint16'):
        npa = npa.astype(np.float32, copy=False)
    tensor = t.from_numpy(npa)
    if len(tensor.shape) < 3:
        tensor = tensor.unsqueeze(0)

    return tensor


class NumpyDataset(Dataset):
    r"""A dataset of numpy files.

    You can either supply the paths of the images directly (through
    input_imgs and target_imgs) or supply an input and target
    directory where the images will be matched if they have the same
    filename.

    :param input_dir: The directory of the input images.
    :param target_dir: The directory of the target images.
    :param allowed_extensions: The extensions that we filter for
                               (default: ['.npy'])
    :param input_imgs: An optional list of image paths for the
                       input images. Path will be joined with input_dir.
    :param target_imgs: An optional list of image paths for the
                        target images. Path will be joined with target_dir.
    :returns:
    :rtype: NumpyDataset

    """

    def __init__(self, input_dir, target_dir,
                 allowed_extensions=['.npy'],
                 input_imgs=None,
                 target_imgs=None):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.allowed_extensions = allowed_extensions

        self.input_imgs = input_imgs
        self.target_imgs = target_imgs

        self._initialize_paths()

    def _initialize_paths(self):
        if self.input_imgs is not None:
            assert len(self.input_imgs) == len(self.target_imgs), \
                "Uneven # of input and target images"

            self.input_imgs = [os.path.join(
                self.input_dir, f) for f in self.input_imgs]
            self.target_imgs = [os.path.join(
                self.target_dir, f) for f in self.target_imgs]

        else:
            input_imgs = get_image_names(
                self.input_dir, self.allowed_extensions)
            target_imgs = get_image_names(
                self.target_dir, self.allowed_extensions)

            # assume that input and target images have the same
            # filename. Discard any images that don't have a corresponding
            # input or target.
            imgs = input_imgs & target_imgs

            self.input_imgs = [os.path.join(self.input_dir, f) for f in imgs]
            self.target_imgs = [os.path.join(self.target_dir, f) for f in imgs]

        if(len(self.input_imgs) == 0 or len(self.target_imgs) == 0):
            msg = 'Could not find any images in "{}" or "{}".'
            msg = msg.format(self.input_dir, self.target_dir)
            raise(RuntimeError(msg))

    def __getitem__(self, index):
        input_f = self.input_imgs[index]
        target_f = self.target_imgs[index]

        input = to_tensor(input_f)
        target = to_tensor(target_f)

        return (input, target)

    def __len__(self):
        return len(self.input_imgs)
