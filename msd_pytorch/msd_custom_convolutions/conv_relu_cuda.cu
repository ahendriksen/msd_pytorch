// -*- eval:(c++-mode); c-file-style: "bsd"; -*-
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// For multi-device code, we have to get the correct CUDA stream to
// run the computations on. We therefore have to use this private API.
#include <c10/cuda/CUDAStream.h>

#include "device_tensor.h"
#include "utils.h"

/**
 * A OptionalDeviceGuard is an RAII class that sets a device to some value on
 * initialization, and resets the device to its original value on destruction.
 * Morally, a OptionalDeviceGuard is equivalent to optional<DeviceGuard>, but
 * with extra constructors and methods as appropriate.
 *
 * Unlike DeviceGuard, a OptionalDeviceGuard may be uninitialized.  This occurs
 * when you use the nullary constructor, or pass a nullopt to the constructor.
 * Uninitialized OptionalDeviceGuards do *nothing*; they do not know what the
 * original device was and they do not reset on destruction.  This is why
 * original_device() and current_device() return optional<Device> rather than
 * Device (as they do in DeviceGuard), and also is why we didn't just
 * provide OptionalDeviceGuard by default and hide DeviceGuard from users.
 *
 * The semantics of an OptionalDeviceGuard are exactly explained by thinking
 * of it as an optional<DeviceGuard>.  In particular, an initialized
 * OptionalDeviceGuard doesn't restore device to its value at construction; it
 * restores device to its value *at initialization*.  So if you have the
 * program:
 *
 *     setDevice(1);
 *     OptionalDeviceGuard g;
 *     setDevice(2);
 *     g.set_device(3);  // initializes!
 *
 * On destruction, g will reset device to 2, rather than 1.
 *
 * An uninitialized OptionalDeviceGuard is distinct from a (initialized)
 * DeviceGuard whose original_device_ and current_device_ match, since the
 * DeviceGuard will still reset the device to original_device_.
 */
// from: torch/include/c10/core/DeviceGuard.h
// We use the OptionalDeviceGuard for multi-GPU programming to make
// sure that the computations take place where the data is.
using torch::OptionalDeviceGuard;

// In the MSD pytorch source code, we sometimes need atomicAdd for 64bit floats.
// This is not supported for compute capability < 6.0 (pre-GTX 10XX series). So
// Nvidia proposes the following fix:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ __forceinline__ int
reflect(int i, int dimi) {
    if (i < 0) {
        i = -i - 1;
    } else if (i >= dimi) {
        i = 2 * dimi - i - 1;
    }
    return i;
}

template <typename scalar_t>
__inline__ __device__
scalar_t warpReduceSum(int mask, scalar_t val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
      val += __shfl_down_sync(mask, val, offset);
  return val;
}

///////////////////////////////////////////////////////////////////////////////
//                         Convolution: Backward Bias                        //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_relu_backward_bias(dTensor4R output,
                        dTensor4R grad_output,
                        dTensor1R grad_bias)
{
    // This implementation of the backward pass wrt the convolution
    // bias uses warp reduction.

    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int H = grad_output.size(2);
    int W = grad_output.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.y * blockDim.x + threadIdx.x;

    // Used for warpReduceSum:
    bool leader = 0 == pId % 32;
    int mask = __activemask();

    if (W <= w || H <= h) {
        return;
    }
    for (int c_out=0; c_out < C_OUT; c_out++) {
        scalar_t g = 0;
        for (int b=0; b < B; b++) {
            if (0.0 < output[b][c_out][h][w]) {
                g += grad_output[b][c_out][h][w];
            }
        }
        g = warpReduceSum(mask, g);
        if (leader) {
            atomicAdd(&grad_bias[c_out], g);
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
//                        Convolution: Backward Kernel                       //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_relu_backward_k(dTensor4R output,
                     dTensor4R grad_output,
                     dTensor4R input,
                     dTensor4R grad_kernel,
                     int dilation)
{
    // In this approach, the gradient sums are reduced accross the
    // warp before being written to global memory.  This
    // implementation does not perform too badly.
    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int C_IN = input.size(1);
    int H = grad_output.size(2);
    int W = grad_output.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.y * blockDim.x + threadIdx.x;

    if (W <= w || H <= h) {
        return;
    }

    bool leader = 0 == pId % 32;
    int mask = __activemask();

    for (int b=0; b < B; b++) {
        for (int c_in=0; c_in < C_IN; c_in++) {
            for (int c_out=0; c_out < C_OUT; c_out++) {
                for (int p=-1; p <= 1; p++) {
                    for (int q=-1; q <= 1; q++) {
                        int h_ = reflect(h + p * dilation, (int) H);
                        int w_ = reflect(w + q * dilation, (int) W);
                        // Only propagate gradient if relu is positive.
                        scalar_t propagate = (0.0 < output[b][c_out][h][w]) ? 1.0 : 0.0;

                        scalar_t g =
                            input[b][c_in][h_][w_] *
                            grad_output[b][c_out][h][w] *
                            propagate;
                        g = warpReduceSum(mask, g);
                        if (leader) {
                            atomicAdd(&grad_kernel[c_out][c_in][p+1][q+1], g);
                        }
                    }
                }
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
//                           Convolution: Backward Input                     //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_relu_backward_x(dTensor4R output,
                      dTensor4R grad_output,
                      dTensor4R kernel,
                      dTensor4R grad_input,
                      int dilation)
{
    // Performance improvements:
    // 1) Store kernel in shared memory.

    // LIMITS:
    //    49152 bytes of shared memory per block
    //    12288 floats of shared memory per block
    // +-  1300 kernels can be stored in shared mem
    // So we must have:
    //     C_IN * C_OUT < 1300
    extern __shared__ int shared_memory[];

    int B = grad_output.size(0);
    int C_OUT = grad_output.size(1);
    int C_IN = grad_input.size(1);
    int H = grad_output.size(2);
    int W = grad_output.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    int num_kernel_elems = kernel.size(0) * kernel.size(1) * kernel.size(2) * kernel.size(3);
    for (int i=pId; i < num_kernel_elems; i+=num_threads) {
        kernel_buf[i] = kernel.data()[i];
    }
    __syncthreads();

    // We can index kernel_buffer like a 4d tensor.
    torch::TensorAccessor<PT4R32> kernel_buffer = kernel.unpack_from(kernel_buf);

    if (W <= w || H <= h) {
        return;
    }

    // Calculate right procession through filter if we are at the border.
    int prog_h[] = {0, 1, 2};
    if (h < dilation) {
        prog_h[2] = 0;
    }
    if (H <= h + dilation) {
        prog_h[0] = 2;
    }

    int prog_w[] = {0, 1, 2};
    if (w < dilation) {
        prog_w[2] = 0;
    }
    if (W <= w + dilation) {
        prog_w[0] = 2;
    }

    DT_INDEX output_offsets[9];
    DT_INDEX grad_output_offsets[9];
    DT_INDEX kernel_offsets[9];
    int i = 0;
    for (int p=-1; p <= 1; p++) {
        for (int q=-1; q <= 1; q++) {
            int hp = reflect(h - dilation * p, (int) H);
            int wq = reflect(w - dilation * q, (int) W);
            output_offsets[i] = ((&output[0][0][hp][wq]) - (&output[0][0][0][0]));
            grad_output_offsets[i] = ((&grad_output[0][0][hp][wq]) - (&grad_output[0][0][0][0]));
            int p_ = prog_h[p+1];
            int q_ = prog_w[q+1];
            kernel_offsets[i] =  ((&kernel_buffer[0][0][p_][q_]) - (&kernel_buffer[0][0][0][0]));
            i++;
        }
    }

    for (int b=0; b < B; b++) {
        for (int c_in=0; c_in < C_IN; c_in++) {
            scalar_t g = 0;
            for (int c_out=0; c_out < C_OUT; c_out++) {
                int i = 0;
                scalar_t *grad_o_p = &grad_output[b][c_out][0][0];
                scalar_t *o_p = &output[b][c_out][0][0];
                scalar_t *kdata = &kernel_buffer[c_out][c_in][0][0];
                for (int p=-1; p <= 1; p++) {
                    for (int q=-1; q <= 1; q++) {
                        if (0.0 < *(o_p + output_offsets[i])) {
                            g += *(grad_o_p + grad_output_offsets[i]) *
                                *(kdata + kernel_offsets[i]);
                        }
                        i++;
                    }
                }
            }
            grad_input[b][c_in][h][w] += g;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
//                            Convolution:Forward                            //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_relu_forward(dTensor4R input,
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

    int B = output.size(0);
    int C_OUT = output.size(1);
    int C_IN = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int pId = threadIdx.x + blockDim.x * threadIdx.y;
    int num_threads = blockDim.x * blockDim.y;

    // Load kernels into shared memory
    scalar_t* kernel_buf = (scalar_t*) shared_memory;
    int num_kernel_elems = kernel.size(0) * kernel.size(1) * kernel.size(2) * kernel.size(3);

    for (int i=pId; i < num_kernel_elems; i+=num_threads) {
        kernel_buf[i] = kernel.data()[i];
    }
    // We can index kernel_buffer like a 4d tensor.
    torch::TensorAccessor<PT4R32> kernel_buffer = kernel.unpack_from(kernel_buf);

    __syncthreads();

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
            output[b][c_out][h][w] = max(0.0, o);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
//                        Kernel preparation functions                       //
///////////////////////////////////////////////////////////////////////////////

torch::Tensor conv_relu_cuda_forward(torch::Tensor input_t,
                             torch::Tensor kernel_t,
                             torch::Tensor bias_t,
                             torch::Tensor out_t,
                             int dilation,
                             int block_size) {
    OptionalDeviceGuard device_guard(device_of(input_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input_t.scalar_type(), "conv_relu_cuda_forward", ([&] {
        // Create device tensors:
        dTensor4R input_d = toDeviceTensorR<scalar_t,4>(input_t);
        dTensor4R kernel_d = toDeviceTensorR<scalar_t,4>(kernel_t);
        dTensor4R out_d = toDeviceTensorR<scalar_t,4>(out_t);
        dTensor1R bias_d = toDeviceTensorR<scalar_t, 1>(bias_t);

        dim3 gridSize(CeilDiv(input_d.size(3), block_size),
                      CeilDiv(input_d.size(2), block_size));
        dim3 blockSize(block_size, block_size);
        auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
        conv_relu_forward<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
            (input_d, kernel_d, bias_d, out_d, dilation);

        CudaCheck(cudaGetLastError());
    }));
    return out_t;
}

void conv_relu_cuda_backward_x(torch::Tensor output_t,
                               torch::Tensor grad_output_t,
                               torch::Tensor kernel_t,
                               torch::Tensor grad_input_t,
                               int dilation,
                               int block_size) {
    OptionalDeviceGuard device_guard(device_of(grad_output_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(output_t.scalar_type(), "conv_relu_cuda_backward_x", ([&] {
        // Create device tensors:
        dTensor4R output_d = toDeviceTensorR<scalar_t,4>(output_t);
        dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output_t);
        dTensor4R grad_input_d = toDeviceTensorR<scalar_t,4>(grad_input_t);
        dTensor4R kernel_d = toDeviceTensorR<scalar_t,4>(kernel_t);
        dim3 gridSize(CeilDiv((int) grad_output_d.size(3), block_size),
                      CeilDiv((int) grad_output_d.size(2), block_size));
        dim3 blockSize(block_size, block_size);
        auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
        conv_relu_backward_x<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
            (output_d, grad_output_d, kernel_d, grad_input_d, dilation);

        CudaCheck(cudaGetLastError());
    }));
}

void conv_relu_cuda_backward_k(torch::Tensor output, torch::Tensor grad_output, torch::Tensor input,
                               torch::Tensor grad_kernel,
                               int dilation, int block_size)
{
    OptionalDeviceGuard device_guard(device_of(grad_output));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv_relu_cuda_backward_k", ([&] {
        // Create device tensors:
        dTensor4R output_d = toDeviceTensorR<scalar_t,4>(output);
        dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output);
        dTensor4R input_d = toDeviceTensorR<scalar_t,4>(input);
        dTensor4R grad_kernel_d = toDeviceTensorR<scalar_t,4>(grad_kernel);
        dim3 gridSize(CeilDiv((int) grad_output_d.size(3), block_size),
                      CeilDiv((int) grad_output_d.size(2), block_size));
        dim3 blockSize(block_size, block_size);
        conv_relu_backward_k<scalar_t><<<gridSize, blockSize, 0, stream>>>
            (output_d, grad_output_d, input_d, grad_kernel_d, dilation);

        CudaCheck(cudaGetLastError());
    }));
}

void conv_relu_cuda_backward_bias(torch::Tensor output,
                                  torch::Tensor grad_output,
                                  torch::Tensor grad_bias,
                                  int block_size)
{
    OptionalDeviceGuard device_guard(device_of(grad_output));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv_relu_cuda_backward_bias", ([&] {
        // Create device tensors:
        dTensor4R output_d = toDeviceTensorR<scalar_t,4>(output);
        dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output);
        dTensor1R grad_bias_d = toDeviceTensorR<scalar_t,1>(grad_bias);
        dim3 gridSize(CeilDiv((int) grad_output_d.size(3), block_size),
                      CeilDiv((int) grad_output_d.size(2), block_size));
        dim3 blockSize(block_size, block_size);
        conv_relu_backward_bias<scalar_t><<<gridSize, blockSize, 0, stream>>>
            (output_d, grad_output_d, grad_bias_d);

        CudaCheck(cudaGetLastError());
    }));
}
