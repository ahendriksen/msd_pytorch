#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstddef>
#include <type_traits>
#include <algorithm>

#define DT_INDEX int

#define PT1R32 scalar_t, 1, mcc::RestrictPtrTraits, DT_INDEX
#define PT2R32 scalar_t, 2, mcc::RestrictPtrTraits, DT_INDEX
#define PT3R32 scalar_t, 3, mcc::RestrictPtrTraits, DT_INDEX
#define PT4R32 scalar_t, 4, mcc::RestrictPtrTraits, DT_INDEX
#define PT5R32 scalar_t, 5, mcc::RestrictPtrTraits, DT_INDEX

#define PT1D32 scalar_t, 1, mcc::DefaultPtrTraits, DT_INDEX
#define PT2D32 scalar_t, 2, mcc::DefaultPtrTraits, DT_INDEX
#define PT3D32 scalar_t, 3, mcc::DefaultPtrTraits, DT_INDEX
#define PT4D32 scalar_t, 4, mcc::DefaultPtrTraits, DT_INDEX
#define PT5D32 scalar_t, 5, mcc::DefaultPtrTraits, DT_INDEX


#define dTensor1R mcc::UnpackableTensorAccessor<PT1R32>
#define dTensor2R mcc::UnpackableTensorAccessor<PT2R32>
#define dTensor3R mcc::UnpackableTensorAccessor<PT3R32>
#define dTensor4R mcc::UnpackableTensorAccessor<PT4R32>
#define dTensor5R mcc::UnpackableTensorAccessor<PT5R32>

#define dTensor1D mcc::UnpackableTensorAccessor<PT1D32>
#define dTensor2D mcc::UnpackableTensorAccessor<PT2D32>
#define dTensor3D mcc::UnpackableTensorAccessor<PT3D32>
#define dTensor4D mcc::UnpackableTensorAccessor<PT4D32>
#define dTensor5D mcc::UnpackableTensorAccessor<PT5D32>

#if defined(__CUDACC__) || defined(__HIPCC__)

// Designates functions callable from the host (CPU) and the device (GPU)
#define MMC_HOST_DEVICE __host__ __device__
#define MMC_DEVICE __device__
#define MMC_HOST __host__

#else

#define MMC_HOST_DEVICE
#define MMC_HOST
#define MMC_DEVICE

#endif


namespace mcc {
    // The PtrTraits argument to the TensorAccessor/GenericPackedTensorAccessor
    // is used to enable the __restrict__ keyword/modifier for the data
    // passed to cuda.
    template <typename T>
    struct DefaultPtrTraits {
	typedef T* PtrType;
    };

#if defined(__CUDACC__) || defined(__HIPCC__)
    template <typename T>
    struct RestrictPtrTraits {
	typedef T* __restrict__ PtrType;
    };
#endif


    // TensorAccessorBase and TensorAccessor are used for both CPU and CUDA tensors.
    // For CUDA tensors it is used in device code (only). This means that we restrict ourselves
    // to functions and types available there (e.g. IntArrayRef isn't).

    // The PtrTraits argument is only relevant to cuda to support `__restrict__` pointers.
    template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
    class TensorAccessorBase {
    public:
	typedef typename PtrTraits<T>::PtrType PtrType;

	MMC_HOST_DEVICE TensorAccessorBase(
					   PtrType data_,
					   const index_t* sizes_,
					   const index_t* strides_)
	    : data_(data_), sizes_(sizes_), strides_(strides_) {}

	MMC_HOST_DEVICE index_t stride(index_t i) const {
	    return strides_[i];
	}
	MMC_HOST_DEVICE index_t size(index_t i) const {
	    return sizes_[i];
	}
	MMC_HOST_DEVICE PtrType data() {
	    return data_;
	}
	MMC_HOST_DEVICE const PtrType data() const {
	    return data_;
	}
    protected:
	PtrType data_;
	const index_t* sizes_;
	const index_t* strides_;
    };

    // The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
    // `Tensor.accessor<T, N>()`.
    // For CUDA `Tensor`s, `GenericPackedTensorAccessor` is used on the host and only
    // indexing on the device uses `TensorAccessor`s.
    template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
    class TensorAccessor : public TensorAccessorBase<T,N,PtrTraits,index_t> {
    public:
	typedef typename PtrTraits<T>::PtrType PtrType;

	MMC_HOST_DEVICE TensorAccessor(
				       PtrType data_,
				       const index_t* sizes_,
				       const index_t* strides_)
	    : TensorAccessorBase<T, N, PtrTraits, index_t>(data_,sizes_,strides_) {}

	MMC_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
	    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
	}

	MMC_HOST_DEVICE const TensorAccessor<T, N-1, PtrTraits, index_t> operator[](index_t i) const {
	    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
	}
    };

    template<typename T, template <typename U> class PtrTraits, typename index_t>
    class TensorAccessor<T,1,PtrTraits,index_t> : public TensorAccessorBase<T,1,PtrTraits,index_t> {
    public:
	typedef typename PtrTraits<T>::PtrType PtrType;

	MMC_HOST_DEVICE TensorAccessor(
				       PtrType data_,
				       const index_t* sizes_,
				       const index_t* strides_)
	    : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_,sizes_,strides_) {}
	MMC_HOST_DEVICE T & operator[](index_t i) {
	    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
	    return this->data_[this->strides_[0]*i];
	}
	MMC_HOST_DEVICE const T & operator[](index_t i) const {
	    return this->data_[this->strides_[0]*i];
	}
    };


    // GenericPackedTensorAccessorBase and GenericPackedTensorAccessor are used on for CUDA `Tensor`s on the host
    // and as
    // In contrast to `TensorAccessor`s, they copy the strides and sizes on instantiation (on the host)
    // in order to transfer them on the device when calling kernels.
    // On the device, indexing of multidimensional tensors gives to `TensorAccessor`s.
    // Use RestrictPtrTraits as PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
    // Instantiation from data, sizes, strides is only needed on the host and std::copy isn't available
    // on the device, so those functions are host only.
    template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
    class GenericPackedTensorAccessorBase {
    public:
	typedef typename PtrTraits<T>::PtrType PtrType;
	MMC_HOST GenericPackedTensorAccessorBase(
						 PtrType data_,
						 const index_t* sizes_,
						 const index_t* strides_)
	    : data_(data_) {
	    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
	    std::copy(strides_, strides_ + N, std::begin(this->strides_));
	}

	// // if index_t is not int64_t, we want to have an int64_t constructor
	template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
	MMC_HOST GenericPackedTensorAccessorBase(
						 PtrType data_,
						 const source_index_t* sizes_,
						 const source_index_t* strides_)
	    : data_(data_) {
	    for (int i = 0; i < N; i++) {
		this->sizes_[i] = sizes_[i];
		this->strides_[i] = strides_[i];
	    }
	}

	MMC_HOST_DEVICE index_t stride(index_t i) const {
	    return strides_[i];
	}
	MMC_HOST_DEVICE index_t size(index_t i) const {
	    return sizes_[i];
	}
	MMC_HOST_DEVICE PtrType data() {
	    return data_;
	}
	MMC_HOST_DEVICE const PtrType data() const {
	    return data_;
	}
    protected:
	PtrType data_;
	index_t sizes_[N];
	index_t strides_[N];
    };

    template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
    class GenericPackedTensorAccessor : public GenericPackedTensorAccessorBase<T,N,PtrTraits,index_t> {
    public:
	typedef typename PtrTraits<T>::PtrType PtrType;

	MMC_HOST GenericPackedTensorAccessor(
					     PtrType data_,
					     const index_t* sizes_,
					     const index_t* strides_)
	    : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

	// if index_t is not int64_t, we want to have an int64_t constructor
	template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
	MMC_HOST GenericPackedTensorAccessor(
					     PtrType data_,
					     const source_index_t* sizes_,
					     const source_index_t* strides_)
	    : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

	MMC_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
	    index_t* new_sizes = this->sizes_ + 1;
	    index_t* new_strides = this->strides_ + 1;
	    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
	}

	MMC_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) const {
	    const index_t* new_sizes = this->sizes_ + 1;
	    const index_t* new_strides = this->strides_ + 1;
	    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
	}
    };

    template<typename T, template <typename U> class PtrTraits, typename index_t>
    class GenericPackedTensorAccessor<T,1,PtrTraits,index_t> : public GenericPackedTensorAccessorBase<T,1,PtrTraits,index_t> {
    public:
	typedef typename PtrTraits<T>::PtrType PtrType;
	MMC_HOST GenericPackedTensorAccessor(
					     PtrType data_,
					     const index_t* sizes_,
					     const index_t* strides_)
	    : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

	// if index_t is not int64_t, we want to have an int64_t constructor
	template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
	MMC_HOST GenericPackedTensorAccessor(
					     PtrType data_,
					     const source_index_t* sizes_,
					     const source_index_t* strides_)
	    : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

	MMC_DEVICE T & operator[](index_t i) {
	    return this->data_[this->strides_[0] * i];
	}
	MMC_DEVICE const T& operator[](index_t i) const {
	    return this->data_[this->strides_[0]*i];
	}
    };

    template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
    class UnpackableTensorAccessor : public GenericPackedTensorAccessor<T,N,PtrTraits,index_t> {
    public:
	typedef typename PtrTraits<T>::PtrType PtrType;


	MMC_HOST UnpackableTensorAccessor(PtrType data_,
					  const index_t* sizes_,
					  const index_t* strides_)
	    : GenericPackedTensorAccessor<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

	// if index_t is not int64_t, we want to have an int64_t constructor
	template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
	MMC_HOST UnpackableTensorAccessor(PtrType data_,
					  const source_index_t* sizes_,
					  const source_index_t* strides_)
	    : GenericPackedTensorAccessor<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

	MMC_DEVICE TensorAccessor<T, N, PtrTraits, index_t> unpack() {
	    return TensorAccessor<T,N,PtrTraits,index_t>(this->data_, this->sizes_, this->strides_);
	}
	MMC_DEVICE TensorAccessor<T, N, PtrTraits, index_t> unpack() const {
	    return TensorAccessor<T,N,PtrTraits,index_t>(this->data_, this->sizes_, this->strides_);
	}

	MMC_DEVICE TensorAccessor<T, N, PtrTraits, index_t> unpack_from(PtrType data) {
	    return TensorAccessor<T,N,PtrTraits,index_t>(data, this->sizes_, this->strides_);
	}
	MMC_DEVICE TensorAccessor<T, N, PtrTraits, index_t> unpack_from(PtrType data) const {
	    return TensorAccessor<T,N,PtrTraits,index_t>(data, this->sizes_, this->strides_);
	}
    };

    template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
    using PackedTensorAccessor32 = GenericPackedTensorAccessor<T, N, PtrTraits, int32_t>;

    template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
    using PackedTensorAccessor64 = GenericPackedTensorAccessor<T, N, PtrTraits, int64_t>;

}
