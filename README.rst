===========
MSD-pytorch
===========


.. image:: https://img.shields.io/pypi/v/msd_pytorch.svg
        :target: https://pypi.python.org/pypi/msd_pytorch

.. image:: https://img.shields.io/travis/ahendriksen/msd_pytorch.svg
        :target: https://travis-ci.org/ahendriksen/msd_pytorch

.. image:: https://readthedocs.org/projects/msd-pytorch/badge/?version=latest
        :target: https://msd-pytorch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/ahendriksen/msd_pytorch/shield.svg
     :target: https://pyup.io/repos/github/ahendriksen/msd_pytorch/
     :alt: Updates


A pytorch implementation of the MSD network structure.


* Free software: MIT license
* Documentation: https://msd-pytorch.readthedocs.io. (WIP)


Installation
------------

Create a conda environment

.. code:: bash

    export CONDA_ENV_NAME=msd
    conda create -y -n $CONDA_ENV_NAME python=3.6.1 anaconda

Then activate the environment:

.. code:: bash

    source activate $CONDA_ENV_NAME

Install PyTorch:

.. code:: bash

    conda install pytorch=0.3.1 torchvision -c pytorch

Now install `msd_pytorch`:

.. code:: bash

    git clone https://gitlab.com/tomo-ml/msd_pytorch
    cd msd_pytorch
    pip install .

You can check if everything works by executing:

.. code:: bash

    make test


Features
--------

* TODO

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
