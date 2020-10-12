[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3560114.svg)](https://doi.org/10.5281/zenodo.3560114)

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

Simply install with either CUDA version 9.0, 10.1, or 10.2:
```
conda install msd_pytorch cudatoolkit=9.0 -c aahendriksen -c pytorch -c defaults -c conda-forge
conda install msd_pytorch cudatoolkit=10.1 -c aahendriksen -c pytorch -c defaults -c conda-forge
conda install msd_pytorch cudatoolkit=10.2 -c aahendriksen -c pytorch -c defaults -c conda-forge
```

**Note**: The order of the channels is important. If you install pytorch from
the default conda channel or from conda-forge, installation might fail.

### Installing from source

To install msd_pytorch from source, you need to have the CUDA toolkit
installed. Specifically, you need `nvcc` and a compatible C++
compiler. Moreover, you need to have a working installation of
PyTorch.

To install PyTorch, we recommend using conda. Install PyTorch with
either of these versions of cudatoolkit:
``` shell
conda install pytorch=1.1.0 cudatoolkit=10.0 -c pytorch
conda install pytorch=1.1.0 cudatoolkit=9.0  -c pytorch
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
