// -*- eval:(c++-mode); c-file-style: "bsd"; -*-
#include <tuple>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// For multi-device code, we have to get the correct CUDA stream to
// run the computations on. We therefore have to use this private API.
#include <c10/cuda/CUDAStream.h>

#include "torch_cuda_dispatch.h"
#include "device_tensor.h"
#include "kernels.cuh"

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


// https://github.com/ClementPinard/extension-cpp/blob/deviceTensorExperiments/cuda/lltm_cuda_kernel.cu
template <typename T, int Dim>
mcc::UnpackableTensorAccessor<T, Dim, mcc::RestrictPtrTraits, DT_INDEX>
toDeviceTensorR(torch::Tensor x) {
    return mcc::UnpackableTensorAccessor<T, Dim,mcc::RestrictPtrTraits,DT_INDEX>(
         x.data_ptr<T>(),
         x.sizes().data(),
         x.strides().data()
    );
}


template <typename T, int Dim>
mcc::UnpackableTensorAccessor<T, Dim, mcc::DefaultPtrTraits, DT_INDEX>
toDeviceTensor(torch::Tensor x) {
    return mcc::UnpackableTensorAccessor<T, Dim,mcc::DefaultPtrTraits,DT_INDEX>(
         x.data_ptr<T>(),
         x.sizes().data(),
         x.strides().data()
    );
}


void check_cuda_error() {
    cudaError err = cudaGetLastError();
    if(err != cudaSuccess) {
	AT_ERROR("Cuda error=", err, " : ", cudaGetErrorString(err));
    }
}

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}


///////////////////////////////////////////////////////////////////////////////
//                        Convolution (no relu)                              //
///////////////////////////////////////////////////////////////////////////////

void conv_cuda_forward(torch::Tensor input_t,
		       torch::Tensor kernel_t,
		       torch::Tensor bias_t,
		       torch::Tensor out_t,
		       int dilation,
		       std::tuple<int, int, int> block_size) {
    OptionalDeviceGuard device_guard(device_of(input_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int block_z, block_y, block_x;
    std::tie(block_z, block_y, block_x) = block_size;

    if (input_t.dim() == 4) {
	// 2D convolution
	AT_DISPATCH_FLOATING_TYPES(input_t.scalar_type(), "conv_cuda_forward", ([&] {
		    // Create device tensors:
		    dTensor4R input_d = toDeviceTensorR<scalar_t,4>(input_t);
		    dTensor4R kernel_d = toDeviceTensorR<scalar_t,4>(kernel_t);
		    dTensor4R out_d = toDeviceTensorR<scalar_t,4>(out_t);
		    dTensor1R bias_d = toDeviceTensorR<scalar_t, 1>(bias_t);

		    dim3 gridSize(CeilDiv((int) input_d.size(3), block_x),
				  CeilDiv((int) input_d.size(2), block_y));
		    dim3 blockSize(block_x, block_y);
		    auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);

		    conv_both_forward<scalar_t, false><<<gridSize, blockSize, buffer_sz, stream>>>
			(input_d, kernel_d, bias_d, out_d, dilation);

		    check_cuda_error();
		}));
    } else {
	// 3D convolution
	AT_DISPATCH_FLOATING_TYPES(input_t.scalar_type(), "conv3d_cuda_forward", ([&] {
		    // Create device tensors:
		    dTensor5R input_d = toDeviceTensorR<scalar_t,5>(input_t);
		    dTensor5R kernel_d = toDeviceTensorR<scalar_t,5>(kernel_t);
		    dTensor5R out_d = toDeviceTensorR<scalar_t,5>(out_t);
		    dTensor1R bias_d = toDeviceTensorR<scalar_t, 1>(bias_t);

		    dim3 gridSize(CeilDiv((int) input_d.size(4), block_x),
				  CeilDiv((int) input_d.size(3), block_y),
				  CeilDiv((int) input_d.size(2), block_z));
		    dim3 blockSize(block_x, block_y, block_z);

		    auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
		    conv3d_forward<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
			(input_d, kernel_d, bias_d, out_d, dilation);

		    check_cuda_error();
		}));
    }
}

void conv_cuda_backward_x(torch::Tensor grad_output_t,
                          torch::Tensor kernel_t,
                          torch::Tensor grad_input_t,
                          int dilation,
			  std::tuple<int, int, int> block_size){
    OptionalDeviceGuard device_guard(torch::device_of(grad_output_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int block_z, block_y, block_x;
    std::tie(block_z, block_y, block_x) = block_size;

    if (kernel_t.dim() == 4) {
	AT_DISPATCH_FLOATING_TYPES(grad_output_t.scalar_type(), "conv_cuda_backward_x", ([&] {
		    // Create device tensors:
		    dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output_t);
		    dTensor4R grad_input_d = toDeviceTensorR<scalar_t,4>(grad_input_t);
		    dTensor4R kernel_d = toDeviceTensorR<scalar_t,4>(kernel_t);

		    dim3 gridSize(CeilDiv((int) grad_output_d.size(3), block_x),
				  CeilDiv((int) grad_output_d.size(2), block_y));
		    dim3 blockSize(block_x, block_y);
		    auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
		    conv_backward_x<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
			(grad_output_d, kernel_d, grad_input_d, dilation);

		    check_cuda_error();
		}));
    } else {
	AT_DISPATCH_FLOATING_TYPES(grad_output_t.scalar_type(), "conv3d_cuda_backward_x", ([&] {
		    // Create device tensors:
		    dTensor5R grad_output_d = toDeviceTensorR<scalar_t,5>(grad_output_t);
		    dTensor5R grad_input_d = toDeviceTensorR<scalar_t,5>(grad_input_t);
		    dTensor5R kernel_d = toDeviceTensorR<scalar_t,5>(kernel_t);
		    dim3 gridSize(CeilDiv((int) grad_output_d.size(4), block_x),
				  CeilDiv((int) grad_output_d.size(3), block_y),
				  CeilDiv((int) grad_output_d.size(2), block_z));
		    dim3 blockSize(block_x, block_y, block_z);
		    auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
		    conv3d_backward_x<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
			(grad_output_d, kernel_d, grad_input_d, dilation);

		    check_cuda_error();
		}));
    }
}

void conv_cuda_backward_k(torch::Tensor grad_output, torch::Tensor input,
                          torch::Tensor grad_kernel,
                          int dilation, std::tuple<int, int, int> block_size)
{
    OptionalDeviceGuard device_guard(torch::device_of(grad_output));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int block_z, block_y, block_x;
    std::tie(block_z, block_y, block_x) = block_size;

    if (grad_kernel.dim() == 4) {
	AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv_cuda_backward_k", ([&] {
		    // Create device tensors:
		    dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output);
		    dTensor4R input_d = toDeviceTensorR<scalar_t,4>(input);
		    dTensor4R grad_kernel_d = toDeviceTensorR<scalar_t,4>(grad_kernel);

		    dim3 gridSize(CeilDiv((int) grad_output_d.size(3), block_x),
				  CeilDiv((int) grad_output_d.size(2), block_y));
		    dim3 blockSize(block_x, block_y);

		    conv_backward_k<scalar_t><<<gridSize, blockSize, 0, stream>>>
			(grad_output_d, input_d, grad_kernel_d, dilation);

		    check_cuda_error();
		}));
    } else {
	AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv3d_cuda_backward_k", ([&] {
		    // Create device tensors:
		    dTensor5R grad_output_d = toDeviceTensorR<scalar_t,5>(grad_output);
		    dTensor5R input_d = toDeviceTensorR<scalar_t,5>(input);
		    dTensor5R grad_kernel_d = toDeviceTensorR<scalar_t,5>(grad_kernel);

		    dim3 gridSize(CeilDiv((int) grad_output_d.size(4), block_x),
				  CeilDiv((int) grad_output_d.size(3), block_y),
				  CeilDiv((int) grad_output_d.size(2), block_z));
		    dim3 blockSize(block_x, block_y, block_z);

		    conv3d_backward_k<scalar_t><<<gridSize, blockSize, 0, stream>>>
			(grad_output_d, input_d, grad_kernel_d, dilation);

		    check_cuda_error();
		}));
    }
}

///////////////////////////////////////////////////////////////////////////////
//                        Convolution (including relu)                       //
///////////////////////////////////////////////////////////////////////////////

void conv_relu_cuda_forward(torch::Tensor input_t,
			    torch::Tensor kernel_t,
			    torch::Tensor bias_t,
			    torch::Tensor out_t,
			    int dilation,
			    std::tuple<int, int, int> block_size) {
    OptionalDeviceGuard device_guard(device_of(input_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int block_z, block_y, block_x;
    std::tie(block_z, block_y, block_x) = block_size;

    if (kernel_t.dim() == 4) {
	AT_DISPATCH_FLOATING_TYPES(input_t.scalar_type(), "conv_relu_cuda_forward", ([&] {
		    // Create device tensors:
		    dTensor4R input_d = toDeviceTensorR<scalar_t,4>(input_t);
		    dTensor4R kernel_d = toDeviceTensorR<scalar_t,4>(kernel_t);
		    dTensor4R out_d = toDeviceTensorR<scalar_t,4>(out_t);
		    dTensor1R bias_d = toDeviceTensorR<scalar_t, 1>(bias_t);

		    dim3 gridSize(CeilDiv((int) input_d.size(3), block_x),
				  CeilDiv((int) input_d.size(2), block_y));
		    dim3 blockSize(block_x, block_y);

		    auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
		    conv_both_forward<scalar_t, true><<<gridSize, blockSize, buffer_sz, stream>>>
			(input_d, kernel_d, bias_d, out_d, dilation);

		    check_cuda_error();
		}));
    } else {
	AT_DISPATCH_FLOATING_TYPES(input_t.scalar_type(), "conv3d_relu_cuda_forward", ([&] {
		    // Create device tensors:
		    dTensor5R input_d = toDeviceTensorR<scalar_t,5>(input_t);
		    dTensor5R kernel_d = toDeviceTensorR<scalar_t,5>(kernel_t);
		    dTensor5R out_d = toDeviceTensorR<scalar_t,5>(out_t);
		    dTensor1R bias_d = toDeviceTensorR<scalar_t, 1>(bias_t);

		    dim3 gridSize(CeilDiv((int) input_d.size(4), block_x),
				  CeilDiv((int) input_d.size(3), block_y),
				  CeilDiv((int) input_d.size(2), block_z));
		    dim3 blockSize(block_x, block_y, block_z);

		    auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
		    conv3d_relu_forward<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
			(input_d, kernel_d, bias_d, out_d, dilation);

		    check_cuda_error();
		}));
    }
}

void conv_relu_cuda_backward_x(torch::Tensor output_t,
                               torch::Tensor grad_output_t,
                               torch::Tensor kernel_t,
                               torch::Tensor grad_input_t,
                               int dilation,
                               std::tuple<int, int, int> block_size) {
    OptionalDeviceGuard device_guard(device_of(grad_output_t));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int block_z, block_y, block_x;
    std::tie(block_z, block_y, block_x) = block_size;

    if (kernel_t.dim() == 4) {
	AT_DISPATCH_FLOATING_TYPES(output_t.scalar_type(), "conv_relu_cuda_backward_x", ([&] {
		    // Create device tensors:
		    dTensor4R output_d = toDeviceTensorR<scalar_t,4>(output_t);
		    dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output_t);
		    dTensor4R grad_input_d = toDeviceTensorR<scalar_t,4>(grad_input_t);
		    dTensor4R kernel_d = toDeviceTensorR<scalar_t,4>(kernel_t);

		    dim3 gridSize(CeilDiv((int) grad_output_d.size(3), block_x),
				  CeilDiv((int) grad_output_d.size(2), block_y));
		    dim3 blockSize(block_x, block_y);
		    auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
		    conv_relu_backward_x<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
			(output_d, grad_output_d, kernel_d, grad_input_d, dilation);

		    check_cuda_error();
		}));
    } else {
	AT_DISPATCH_FLOATING_TYPES(output_t.scalar_type(), "conv3d_relu_cuda_backward_x", ([&] {
		    // Create device tensors:
		    dTensor5R output_d = toDeviceTensorR<scalar_t,5>(output_t);
		    dTensor5R grad_output_d = toDeviceTensorR<scalar_t,5>(grad_output_t);
		    dTensor5R grad_input_d = toDeviceTensorR<scalar_t,5>(grad_input_t);
		    dTensor5R kernel_d = toDeviceTensorR<scalar_t,5>(kernel_t);

		    dim3 gridSize(CeilDiv((int) grad_output_d.size(4), block_x),
				  CeilDiv((int) grad_output_d.size(3), block_y),
				  CeilDiv((int) grad_output_d.size(2), block_z));
		    dim3 blockSize(block_x, block_y, block_z);

		    auto buffer_sz = kernel_t.numel() * sizeof(scalar_t);
		    conv3d_relu_backward_x<scalar_t><<<gridSize, blockSize, buffer_sz, stream>>>
			(output_d, grad_output_d, kernel_d, grad_input_d, dilation);

		    check_cuda_error();
		}));
    }
}

void conv_relu_cuda_backward_k(torch::Tensor output, torch::Tensor grad_output, torch::Tensor input,
                               torch::Tensor grad_kernel,
                               int dilation, std::tuple<int, int, int> block_size)
{
    OptionalDeviceGuard device_guard(device_of(grad_output));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int block_z, block_y, block_x;
    std::tie(block_z, block_y, block_x) = block_size;

    if (output.dim() == 4) {
	AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv_relu_cuda_backward_k", ([&] {
		    // Create device tensors:
		    dTensor4R output_d = toDeviceTensorR<scalar_t,4>(output);
		    dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output);
		    dTensor4R input_d = toDeviceTensorR<scalar_t,4>(input);
		    dTensor4R grad_kernel_d = toDeviceTensorR<scalar_t,4>(grad_kernel);

		    dim3 gridSize(CeilDiv((int) grad_output_d.size(3), block_x),
				  CeilDiv((int) grad_output_d.size(2), block_y));
		    dim3 blockSize(block_x, block_y);

		    conv_relu_backward_k<scalar_t><<<gridSize, blockSize, 0, stream>>>
			(output_d, grad_output_d, input_d, grad_kernel_d, dilation);

		    check_cuda_error();
		}));
    } else {
	AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv3d_relu_cuda_backward_k", ([&] {
		    // Create device tensors:
		    dTensor5R output_d = toDeviceTensorR<scalar_t,5>(output);
		    dTensor5R grad_output_d = toDeviceTensorR<scalar_t,5>(grad_output);
		    dTensor5R input_d = toDeviceTensorR<scalar_t,5>(input);
		    dTensor5R grad_kernel_d = toDeviceTensorR<scalar_t,5>(grad_kernel);

		    dim3 gridSize(CeilDiv((int) grad_output_d.size(4), block_x),
				  CeilDiv((int) grad_output_d.size(3), block_y),
				  CeilDiv((int) grad_output_d.size(2), block_z));
		    dim3 blockSize(block_x, block_y, block_z);

		    conv3d_relu_backward_k<scalar_t><<<gridSize, blockSize, 0, stream>>>
			(output_d, grad_output_d, input_d, grad_kernel_d, dilation);

		    check_cuda_error();
		}));
    }
}

void conv_relu_cuda_backward_bias(torch::Tensor output,
                                  torch::Tensor grad_output,
                                  torch::Tensor grad_bias,
                                  std::tuple<int, int, int> block_size)
{
    OptionalDeviceGuard device_guard(device_of(grad_output));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int block_z, block_y, block_x;
    std::tie(block_z, block_y, block_x) = block_size;

    if (output.dim() == 4) {
	AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv_relu_cuda_backward_bias", ([&] {
		    // Create device tensors:
		    dTensor4R output_d = toDeviceTensorR<scalar_t,4>(output);
		    dTensor4R grad_output_d = toDeviceTensorR<scalar_t,4>(grad_output);
		    dTensor1R grad_bias_d = toDeviceTensorR<scalar_t,1>(grad_bias);

		    dim3 gridSize(CeilDiv((int) grad_output_d.size(3), block_x),
				  CeilDiv((int) grad_output_d.size(2), block_y));
		    dim3 blockSize(block_x, block_y);

		    conv_relu_backward_bias<scalar_t><<<gridSize, blockSize, 0, stream>>>
			(output_d, grad_output_d, grad_bias_d);

		    check_cuda_error();
		}));
    } else {
	AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv3d_relu_cuda_backward_bias", ([&] {
		    // Create device tensors:
		    dTensor5R output_d = toDeviceTensorR<scalar_t,5>(output);
		    dTensor5R grad_output_d = toDeviceTensorR<scalar_t,5>(grad_output);
		    dTensor1R grad_bias_d = toDeviceTensorR<scalar_t,1>(grad_bias);

		    dim3 gridSize(CeilDiv((int) grad_output_d.size(4), block_x),
				  CeilDiv((int) grad_output_d.size(3), block_y),
				  CeilDiv((int) grad_output_d.size(2), block_z));
		    dim3 blockSize(block_x, block_y, block_z);

		    conv3d_relu_backward_bias<scalar_t><<<gridSize, blockSize, 0, stream>>>
			(output_d, grad_output_d, grad_bias_d);

		    check_cuda_error();
		}));
    }
}