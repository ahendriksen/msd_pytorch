#include <cuda.h>
#include <cuda_runtime.h>

#include "device_tensor.h"
#include "kernel_utils.cuh"

#define kernel_tuner 1

#ifdef kernel_tuner
    #define __kernel__ __device__
#else
    #define __kernel__ __global__
#endif

template <typename scalar_t>
static __inline__ __device__ void pack_shared_mem(scalar_t* dest, scalar_t* src, int n){
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    for (int i=pId; i < n; i+=num_threads) {
        dest[i] = src[i];
    }
    __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
//                            Convolution:Forward                            //
///////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__kernel__
void
conv_forward(dTensor4R input,
             dTensor4R kernel,
             dTensor1R bias,
             dTensor4R output,
             int dilation)
{
    // The following has been done to improve performance:
    // 1) This implementation caches the kernel weights.
    // 2) This implementation precomputes data offsets in x and y
    //    direction instead of pointers.

    // LIMITS:
    //    49152 bytes of shared memory per block
    //    12288 floats of shared memory per block
    // +-  1300 kernels can be stored in shared mem
    // So we must have:
    //     C_IN * C_OUT < 1300
    extern __shared__ int shared_memory[];
    scalar_t* buf = (scalar_t*) shared_memory;

    int B = output.size(0);
    int C_OUT = output.size(1);
    int C_IN = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;

    // // Load kernels into shared memory
    int num_kernel_elems = kernel.size(0) * kernel.size(1) * kernel.size(2) * kernel.size(3);
    pack_shared_mem<scalar_t>(buf, kernel.data(), num_kernel_elems);
    mcc::TensorAccessor<PT4R32> kernel_buffer = kernel.unpack_from(buf);

    if (W <= w || H <= h) {
        return;
    }

    // Precompute data offsets:
    int hs[3] = {0};
    int ws[3] = {0};

    for (int i=-1; i <= 1; i++) {
        hs[i + 1] = reflect(h + dilation * i, (int) H);
        ws[i + 1] = reflect(w + dilation * i, (int) W);
    }

    // Actually compute the convolution
    for (int b=0; b < B; b++) {
        for (int c_out=0; c_out < C_OUT; c_out++) {
            scalar_t o = bias[c_out];
            for (int c_in=0; c_in < C_IN; c_in++) {
                scalar_t *kernel0 = &kernel_buffer[c_out][c_in][0][0];
                #pragma unroll
                for (int p=-1; p <= 1; p++) {
                    #pragma unroll
                    for (int q=-1; q <= 1; q++) {
                        o += input[b][c_in][hs[p + 1]][ws[q + 1]] * (*kernel0);
                        // Incrementing the kernel pointer works because
                        // the kernel weights are contiguous and the
                        // data_offsets are prepared to be in the same
                        // order as the kernel weights.
                        kernel0++;
                    }
                }
            }
            output[b][c_out][h][w] = o;
        }
    }
}



template
__kernel__ void
conv_forward<float>(dTensor4Rfloat input,
             dTensor4Rfloat kernel,
             dTensor1Rfloat bias,
             dTensor4Rfloat output,
             int dilation);

template
__kernel__ void
conv_forward<double>(dTensor4Rdouble input,
             dTensor4Rdouble kernel,
             dTensor1Rdouble bias,
             dTensor4Rdouble output,
             int dilation);


#ifdef kernel_tuner
#include <initializer_list>
#include <type_traits>

extern "C" __global__ void conv_forward_tuner(float* __restrict__ input, float* __restrict__ kernel, float* __restrict__ bias, float* __restrict__ output, int dilation) {

    // TODO: Do not hard code sizes and strides...
    auto input_size = {1, 1, 10, 10};
    auto input_strides = {INPUT_STRIDES};
    auto kernel_size = {1, 1, 3, 3};
    auto kernel_strides = {9, 9, 3, 1};
    auto bias_size = {1};
    auto bias_strides = {1};

    // 1) type of the data in array
    // 2) dimensions of array (4-dimensional in this case)
    // 3) Does the array alias another array? No: mmc::RestrictPtrTraits; Yes: mcc::DefaultPtrTraits
    // 4) Array is indexed using int32 (int64 is also possible, but noticeably slower)
    // 5) Pointer to array data
    // 6) Size of array
    // 7) Strides of array
    //                                                 (1)  (2) (3)                     (4)       (5)      (6)              (7)

    auto input_tensor = mcc::UnpackableTensorAccessor<float, 4, mcc::RestrictPtrTraits, int32_t>(input, input_size.begin(), input_strides.begin());
    auto kernel_tensor = mcc::UnpackableTensorAccessor<float, 4, mcc::RestrictPtrTraits, int32_t>(kernel, kernel_size.begin(), kernel_strides.begin());
    auto bias_tensor = mcc::UnpackableTensorAccessor<float, 1, mcc::RestrictPtrTraits, int32_t>(bias, bias_size.begin(), bias_strides.begin());
    auto output_tensor = mcc::UnpackableTensorAccessor<float, 4, mcc::RestrictPtrTraits, int32_t>(output, input_size.begin(), input_strides.begin());

    conv_forward(input_tensor, kernel_tensor, bias_tensor, output_tensor, dilation);
}

// extern "C" __global__ void conv_forward_tuner_wrapper(float* __restrict__ input, float* __restrict__ kernel, float* __restrict__ bias, float* __restrict__ output, int dilation) {
//     conv_forward_tuner(input, kernel, bias, output, dilation);
// }

// extern "C" __global__ void conv_forward_wrapper(dTensor4Rfloat input, dTensor4Rfloat kernel, dTensor1Rfloat bias, dTensor4Rfloat output, int dilation) {
//   conv_forward<float>(input, kernel, bias, output, dilation);
// }
#endif
