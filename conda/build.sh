#!/bin/sh

echo "------------------------------------------------------------"
echo "Build msd-pytorch Environment: "
echo "------------------------------------------------------------"

env

echo "------------------------------------------------------------"


# Inspired by  astra-toolbox/python/conda/linux_release/release.sh
if [ -z $TMP_NVCC_LOCATION ]; then
    export TMP_NVCC_LOCATION=/tmp/cuda_toolkits ;
fi

case $cudatoolkit in
    9.0)  export CUDA_URL="https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run";;
    9.2)  export CUDA_URL="https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux";;
    10.0) export CUDA_URL="https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux";;
    *) echo -n "cudatoolkit version $cudatoolkit not recognized. Exiting."; exit 1;;
esac

echo "------------------------------------------------------------"
echo "Installing CUDA version: $cudatoolkit"
echo "-- "

mkdir -p $TMP_NVCC_LOCATION
[ -f $TMP_NVCC_LOCATION/`basename $CUDA_URL` ] || (cd $TMP_NVCC_LOCATION; wget $CUDA_URL )

export CUDA_INSTALLER=$TMP_NVCC_LOCATION/`basename $CUDA_URL`

export CUDA_HOME=$TMP_NVCC_LOCATION/cuda-$cudatoolkit

if [ -f $CUDA_HOME/bin/nvcc ] ; then
    echo "-- CUDA already installed";
else
    echo "Installing version ${cudatoolkit} "
    echo "  with ${CUDA_INSTALLER}"
    echo "  into ${CUDA_HOME}"
    echo "  with temp space ${TMP_NVCC_LOCATION}"
    sh $CUDA_INSTALLER --silent --toolkit --toolkitpath=$CUDA_HOME --tmpdir=$TMP_NVCC_LOCATION --override
fi

echo "-- Setting CUDA_HOME: $CUDA_HOME"
echo "------------------------------------------------------------"

$PYTHON setup.py clean
# PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH CUDA_HOME=$CUDA_HOME $PYTHON setup.py install --single-version-externally-managed --record record.txt || exit 1
CUDA_HOME=$CUDA_HOME $PYTHON setup.py install --single-version-externally-managed --record record.txt || exit 1
