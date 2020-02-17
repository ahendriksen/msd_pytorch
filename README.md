[![DOI](https://zenodo.org/badge/188229038.svg)](https://zenodo.org/badge/latestdoi/188229038)

# Mixed-scale Dense Networks for PyTorch

An implementation of Mixed-Scale Dense networks in PyTorch.

* Free software: GNU General Public License v3
* Documentation: [https://ahendriksen.github.io/msd_pytorch]

## Getting Started

It takes a few steps to setup Mixed-scale Dense Networks for PyTorch
on your machine. We recommend installing [Anaconda package
manager](https://www.anaconda.com/download/) for Python 3.

### Requirements

This package requires

- Linux x64
- CUDA 9.0 and/or 10.0 compatible graphics card
- [Anaconda package manager](https://www.anaconda.com/download/)

This package is compatible with python 3.6 and 3.7.

### Installing with Conda

Simply install with either cuda version 9.0 or version 10.0:
```
conda install -c conda-forge -c aahendriksen -c pytorch msd_pytorch cudatoolkit=9.0
# or
conda install -c conda-forge -c aahendriksen -c pytorch msd_pytorch cudatoolkit=10.0
```

### Installing from source

To install msd_pytorch from source, you need to have the CUDA toolkit
installed. Specifically, you need `nvcc` and a compatible C++
compiler. Moreover, you need to have a working installation of
PyTorch.

To install PyTorch, we recommend using conda. Install PyTorch with
either of these versions of cudatoolkit:
``` shell
conda install pytorch=1.1.0 torchvision cudatoolkit=10.0 -c pytorch
conda install pytorch=1.1.0 torchvision cudatoolkit=9.0  -c pytorch
```

To get the source code, simply clone this GitHub project.
``` shell
git clone https://github.com/ahendriksen/msd_pytorch.git
cd msd_pytorch
```

Using pip to install the package automatically triggers the
compilation of the native C++ and CUDA code. So you need to direct the
installer to a CUDA-compatible C++ compiler in this way:
``` shell
GXX=/path/to/compatible/cpp/compiler pip install -e .[dev]
```
Or, if the standard C++ compiler is compatible with CUDA:
``` shell
pip install -e .[dev]
```

### Using the tools

The msd_pytorch package ships with some command-line tools to make
your life easier. If you have input and target images in directories
`./train/input/` and `./train/target/`, then you can train a network
to do regression with the following command in your terminal:

``` shell
msd regression -p with train_input_glob='./train/input/*' train_target_glob='./train/target/*' epochs=10 msd.depth=30
```

Similarly, segmentation is possible using the following command:
``` shell
msd segmentation -p with train_input_glob='./train/input/*' train_target_glob='./train/target/*' epochs=10 msd.depth=30 labels=[0,1,2,3]
```

More command-line arguments are available

``` yaml
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
```

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

* `"~/train_images/"`
* `"~/train_images/cats*.png"`
* `"~/train_images/*.tif"`
* `"~/train_images/scan*"`
* `"~/train_images/just_one_image.jpeg"`

### Running the examples

To learn more about the functionality of the package check out our
examples folder.


## Cite
If you find our work useful, please cite as:

```
@software{hendriksen-2019-msd-pytor,
  author       = {Hendriksen, Allard A.},
  title        = {ahendriksenh/msd\_pytorch: v0.7.2},
  month        = dec,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v0.7.2},
  doi          = {10.5281/zenodo.3560114},
  url          = {https://doi.org/10.5281/zenodo.3560114}
}
```

## Authors and contributors

* **Allard Hendriksen** - *Initial work*
* **Jonas Adler** - *Discussions and code*
* **Richard Schoonhoven** - *Testing and patches*

See also the list of [contributors](https://github.com/ahendriksen/msd_pytorch/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `master` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.
