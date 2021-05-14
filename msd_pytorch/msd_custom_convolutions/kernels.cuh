#pragma once
#include "device_tensor.h"

///////////////////////////////////////////////////////////////////////////////
//                         Convolution: Backward Bias                        //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_relu_backward_bias(dTensor4R output,
                        dTensor4R grad_output,
                        dTensor1R grad_bias);

template <typename scalar_t>
__global__ void
conv3d_relu_backward_bias(dTensor5R output,
			  dTensor5R grad_output,
			  dTensor1R grad_bias);

///////////////////////////////////////////////////////////////////////////////
//                        Convolution: Backward Kernel                       //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_backward_k(dTensor4R grad_output,
                dTensor4R input,
                dTensor4R grad_kernel,
                int dilation);

template <typename scalar_t>
__global__ void
conv_relu_backward_k(dTensor4R output,
                     dTensor4R grad_output,
                     dTensor4R input,
                     dTensor4R grad_kernel,
                     int dilation);

template <typename scalar_t>
__global__ void
conv3d_backward_k(dTensor5R grad_output,
                  dTensor5R input,
                  dTensor5R grad_kernel,
                  int dilation);

template <typename scalar_t>
__global__ void
conv3d_relu_backward_k(dTensor5R output,
                       dTensor5R grad_output,
                       dTensor5R input,
                       dTensor5R grad_kernel,
                       int dilation);

///////////////////////////////////////////////////////////////////////////////
//                           Convolution: Backward Input                     //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_backward_x(dTensor4R grad_output,
                dTensor4R kernel,
                dTensor4R grad_input,
                int dilation);

template <typename scalar_t>
__global__ void
conv_relu_backward_x(dTensor4R output,
		     dTensor4R grad_output,
		     dTensor4R kernel,
		     dTensor4R grad_input,
		     int dilation);

template <typename scalar_t>
__global__ void
conv3d_backward_x(dTensor5R grad_output,
		  dTensor5R kernel,
		  dTensor5R grad_input,
		  int dilation);

template <typename scalar_t>
__global__ void
conv3d_relu_backward_x(dTensor5R output,
                       dTensor5R grad_output,
                       dTensor5R kernel,
                       dTensor5R grad_input,
                       int dilation);

///////////////////////////////////////////////////////////////////////////////
//                            Convolution:Forward                            //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_forward(dTensor4R input,
             dTensor4R kernel,
             dTensor1R bias,
             dTensor4R output,
             int dilation);

template <typename scalar_t>
__global__ void
conv_relu_forward(dTensor4R input,
                  dTensor4R kernel,
                  dTensor1R bias,
                  dTensor4R output,
                  int dilation);

template <typename scalar_t>
__global__ void
conv3d_forward(dTensor5R input,
	       dTensor5R kernel,
	       dTensor1R bias,
	       dTensor5R output,
	       int dilation);

template <typename scalar_t>
__global__ void
conv3d_relu_forward(dTensor5R input,
		    dTensor5R kernel,
		    dTensor1R bias,
		    dTensor5R output,
		    int dilation);
