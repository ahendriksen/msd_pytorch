#!/bin/sh

echo "------------------------------------------------------------"
echo "Build msd-pytorch Environment: "
echo "------------------------------------------------------------"

env

echo "------------------------------------------------------------"

if [ -f $CUDA_HOME/bin/nvcc ] ; then
    echo "-- CUDA already installed";
else
    echo "Using version ${cudatoolkit} "
    export CUDA_HOME=/usr/local/cuda-${cudatoolkit}
    echo "  set CUDA_HOME=${CUDA_HOME}"
fi


$PYTHON setup.py clean
$PYTHON setup.py install --single-version-externally-managed --record record.txt || exit 1
