#!/usr/bin/env bash
set -euo pipefail

mkdir -p conda_docker_out
mkdir -p release_out

export MAX_JOBS=4

MOUNTS="-v ${PWD}/conda_docker_out/:/opt/conda/conda-bld/linux-64/ -v ${PWD}/:/msd_build_dir/ -v pkgs:/opt/conda/pkgs"


export cudatoolkit=10.2
export CUDA_HOME=/usr/local/cuda-${cudatoolkit}/
# For more information about TORCH_CUDA_ARCH_LIST, see:
# - https://github.com/pytorch/pytorch/commit/cd207737017db8c81584763207df20bc6110ed75
# - https://en.wikipedia.org/wiki/CUDA#GPUs_supported
# - https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-gpu-architecture

export TORCH_CUDA_ARCH_LIST="3.5+PTX 6.0+PTX 7.0+PTX"
docker run --rm -e CUDA_HOME=${CUDA_HOME} -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" -e cudatoolkit=${cudatoolkit} -e MAX_JOBS=${MAX_JOBS} $MOUNTS -w /msd_build_dir/ msd-build-env /opt/conda/bin/conda mambabuild conda/ -c aahendriksen -c pytorch -c defaults -c conda-forge -m conda/cuda-${cudatoolkit}.yaml
cp conda_docker_out/*tar.bz2 release_out/

export cudatoolkit=11.1
export CUDA_HOME=/usr/local/cuda-${cudatoolkit}/
export TORCH_CUDA_ARCH_LIST="3.5+PTX 6.0+PTX 7.0+PTX 8.0+PTX"
docker run --rm -e CUDA_HOME=${CUDA_HOME} -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" -e cudatoolkit=${cudatoolkit} -e MAX_JOBS=${MAX_JOBS} $MOUNTS -w /msd_build_dir/ msd-build-env /opt/conda/bin/conda mambabuild conda/ -c aahendriksen -c pytorch -c defaults -c conda-forge -m conda/cuda-${cudatoolkit}.yaml
cp conda_docker_out/*tar.bz2 release_out/
