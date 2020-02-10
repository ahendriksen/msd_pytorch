#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

with open(os.path.join('msd_pytorch', 'VERSION')) as version_file:
    version = version_file.read().strip()

requirements = [
    # Add your project's requirements here, e.g.,
    'sacred>=0.7.2',
    "tifffile",
    "imageio"
]

setup_requirements = [
    'pytest-runner'
]

test_requirements = [
    'pytest'
]

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'recommonmark',
    # Other
    'watchdog',
    'coverage',
    'pytest',
    'pytest-runner'
]


def __nvcc_args():
    gxx = os.environ.get('GXX')
    if gxx is not None:
        return ['-ccbin', gxx]
    else:
        return []


setup(
    author="Allard Hendriksen",
    author_email='allard.hendriksen@cwi.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="An implementation of Mixed-Scale Dense networks in PyTorch. ",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='msd_pytorch',
    name='msd_pytorch',
    entry_points='''
        [console_scripts]
        msd=msd_pytorch.main:main_function
    ''',
    packages=find_packages(include=['msd_pytorch']),
    setup_requires=setup_requirements,
    test_suite='msd_pytorch.tests',
    tests_require=test_requirements,
    extras_require={'dev': dev_requirements},
    url='https://github.com/ahendriksen/msd_pytorch',
    version=version,
    zip_safe=False,
    ext_modules=[
        CUDAExtension(
            name='conv_cuda',
            sources=[
                'msd_pytorch/conv.cpp',
                'msd_pytorch/conv_cuda.cu',
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': __nvcc_args(),
            },
        ),
        CUDAExtension(
            name='conv_relu_cuda',
            sources=[
                'msd_pytorch/conv_relu.cpp',
                'msd_pytorch/conv_relu_cuda.cu',
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': __nvcc_args(),
            },
        ),
        CUDAExtension(
            name='relu_cuda',
            sources=[
                'msd_pytorch/relu.cpp',
                'msd_pytorch/relu_cuda.cu'
            ],
            extra_compile_args={
                'cxx': [],
                'nvcc': __nvcc_args(),
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
