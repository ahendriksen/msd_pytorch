#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T, size_t N, template <typename U> class PtrTraits = torch::DefaultPtrTraits, typename index_t = int64_t>
class UnpackableTensorAccessor : public torch::PackedTensorAccessor<T,N,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;


    C10_HOST UnpackableTensorAccessor(PtrType data_,
                                  const index_t* sizes_,
                                  const index_t* strides_)
        : torch::PackedTensorAccessor<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

    // if index_t is not int64_t, we want to have an int64_t constructor
    template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
    C10_HOST UnpackableTensorAccessor(PtrType data_,
                                  const source_index_t* sizes_,
                                  const source_index_t* strides_)
        : torch::PackedTensorAccessor<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

    C10_DEVICE torch::TensorAccessor<T, N, PtrTraits, index_t> unpack() {
        return torch::TensorAccessor<T,N,PtrTraits,index_t>(this->data_, this->sizes_, this->strides_);
    }
    C10_DEVICE torch::TensorAccessor<T, N, PtrTraits, index_t> unpack() const {
        return torch::TensorAccessor<T,N,PtrTraits,index_t>(this->data_, this->sizes_, this->strides_);
    }

    C10_DEVICE torch::TensorAccessor<T, N, PtrTraits, index_t> unpack_from(PtrType data) {
        return torch::TensorAccessor<T,N,PtrTraits,index_t>(data, this->sizes_, this->strides_);
    }
    C10_DEVICE torch::TensorAccessor<T, N, PtrTraits, index_t> unpack_from(PtrType data) const {
        return torch::TensorAccessor<T,N,PtrTraits,index_t>(data, this->sizes_, this->strides_);
    }
};


#define DT_INDEX int

#define PT1R32 scalar_t, 1, torch::RestrictPtrTraits, DT_INDEX
#define PT2R32 scalar_t, 2, torch::RestrictPtrTraits, DT_INDEX
#define PT3R32 scalar_t, 3, torch::RestrictPtrTraits, DT_INDEX
#define PT4R32 scalar_t, 4, torch::RestrictPtrTraits, DT_INDEX
#define PT5R32 scalar_t, 4, torch::RestrictPtrTraits, DT_INDEX

#define PT1D32 scalar_t, 1, torch::DefaultPtrTraits, DT_INDEX
#define PT2D32 scalar_t, 2, torch::DefaultPtrTraits, DT_INDEX
#define PT3D32 scalar_t, 3, torch::DefaultPtrTraits, DT_INDEX
#define PT4D32 scalar_t, 4, torch::DefaultPtrTraits, DT_INDEX
#define PT5D32 scalar_t, 5, torch::DefaultPtrTraits, DT_INDEX


#define dTensor1R UnpackableTensorAccessor<PT1R32>
#define dTensor2R UnpackableTensorAccessor<PT2R32>
#define dTensor3R UnpackableTensorAccessor<PT3R32>
#define dTensor4R UnpackableTensorAccessor<PT4R32>
#define dTensor5R UnpackableTensorAccessor<PT5R32>

#define dTensor1D UnpackableTensorAccessor<PT1D32>
#define dTensor2D UnpackableTensorAccessor<PT2D32>
#define dTensor3D UnpackableTensorAccessor<PT3D32>
#define dTensor4D UnpackableTensorAccessor<PT4D32>
#define dTensor5D UnpackableTensorAccessor<PT5D32>

// https://github.com/ClementPinard/extension-cpp/blob/deviceTensorExperiments/cuda/lltm_cuda_kernel.cu
template <typename T, int Dim>
UnpackableTensorAccessor<T, Dim, torch::RestrictPtrTraits, DT_INDEX>
toDeviceTensorR(torch::Tensor x) {
    return UnpackableTensorAccessor<T, Dim,torch::RestrictPtrTraits,DT_INDEX>(
         x.data<T>(),
	 x.sizes().data(),
	 x.strides().data()
    );
}


template <typename T, int Dim>
UnpackableTensorAccessor<T, Dim, torch::DefaultPtrTraits, DT_INDEX>
toDeviceTensor(torch::Tensor x) {
    return UnpackableTensorAccessor<T, Dim,torch::DefaultPtrTraits,DT_INDEX>(
         x.data<T>(),
	 x.sizes().data(),
	 x.strides().data()
    );
}
