#pragma once

#include <ATen/ATen.h>
#include "THC/THC.h"
#include "THC/THCDeviceTensor.cuh"

#define THC_INDEX int

#define dTensor1R THCDeviceTensor<scalar_t, 1, THC_INDEX, RestrictPtrTraits>
#define dTensor2R THCDeviceTensor<scalar_t, 2, THC_INDEX, RestrictPtrTraits>
#define dTensor3R THCDeviceTensor<scalar_t, 3, THC_INDEX, RestrictPtrTraits>
#define dTensor4R THCDeviceTensor<scalar_t, 4, THC_INDEX, RestrictPtrTraits>
#define dTensor5R THCDeviceTensor<scalar_t, 5, THC_INDEX, RestrictPtrTraits>
#define dTensor1 THCDeviceTensor<scalar_t, 1, THC_INDEX>
#define dTensor2 THCDeviceTensor<scalar_t, 2, THC_INDEX>
#define dTensor3 THCDeviceTensor<scalar_t, 3, THC_INDEX>
#define dTensor4 THCDeviceTensor<scalar_t, 4, THC_INDEX>
#define dTensor5 THCDeviceTensor<scalar_t, 5, THC_INDEX>

// https://github.com/ClementPinard/extension-cpp/blob/deviceTensorExperiments/cuda/lltm_cuda_kernel.cu
template <typename scalar_t, int Dim>
THCDeviceTensor<scalar_t, Dim, THC_INDEX, RestrictPtrTraits>
toDeviceTensorR(at::Tensor t) {
    auto _sizes = t.sizes().data();
    auto _strides = t.strides().data();
    THC_INDEX sizes[Dim] = {0};
    THC_INDEX strides[Dim] = {0};
    for (int i=0; i < Dim; i++) {
	sizes[i] = (THC_INDEX) _sizes[i];
	strides[i] = (THC_INDEX) _strides[i];
    }
    return THCDeviceTensor<scalar_t, Dim, THC_INDEX, RestrictPtrTraits>
	(t.data<scalar_t>(), sizes, strides);
}

template <typename scalar_t, int Dim>
THCDeviceTensor<scalar_t, Dim, THC_INDEX>
toDeviceTensor(at::Tensor t) {
    auto _sizes = t.sizes().data();
    auto _strides = t.strides().data();
    THC_INDEX sizes[Dim] = {0};
    THC_INDEX strides[Dim] = {0};
    for (int i=0; i < Dim; i++) {
	sizes[i] = (THC_INDEX) _sizes[i];
	strides[i] = (THC_INDEX) _strides[i];
    }
    return THCDeviceTensor<scalar_t, Dim, THC_INDEX>
	(t.data<scalar_t>(), sizes, strides);
}
