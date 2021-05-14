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

///////////////////////////////////////////////////////////////////////////////
//                        Convolution: Backward Kernel                       //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_relu_backward_k(dTensor4R output,
                     dTensor4R grad_output,
                     dTensor4R input,
                     dTensor4R grad_kernel,
                     int dilation);

///////////////////////////////////////////////////////////////////////////////
//                           Convolution: Backward Input                     //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_relu_backward_x(dTensor4R output,
                      dTensor4R grad_output,
                      dTensor4R kernel,
                      dTensor4R grad_input,
		     int dilation);

///////////////////////////////////////////////////////////////////////////////
//                            Convolution:Forward                            //
///////////////////////////////////////////////////////////////////////////////


template <typename scalar_t>
__global__ void
conv_relu_forward(dTensor4R input,
                  dTensor4R kernel,
                  dTensor1R bias,
                  dTensor4R output,
                  int dilation);

///////////////////////////////////////////////////////////////////////////////
//                        Convolution: Backward Kernel                       //
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void
conv_backward_k(dTensor4R grad_output,
                dTensor4R input,
                dTensor4R grad_kernel,
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
