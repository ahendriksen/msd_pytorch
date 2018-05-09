#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "THC/THC.h"
#include "THC/THCDeviceTensor.cuh"
// #include "THC/THCDeviceTensorUtils.cuh"
#include "THC/THCAtomics.cuh"
#include "THC/THCDeviceUtils.cuh"
#include "THC/THCReduceApplyUtils.cuh"
#include <THC/THCApply.cuh>

template <typename scalar_t>
__global__ void
SpatialReflectionPadding_updateOutputInPlace(THCDeviceTensor<scalar_t, 4> input,
                                             int padL, int padR, int padT,
                                             int padB) {

  int pId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  int w = input.getSize(3);
  int h = input.getSize(2);
  int w_ = w - padL - padR; /* inner width */
  int h_ = h - padT - padB; /* inner height */
  int totalPaddingSize = h * (padL + padR) + w_ * (padT + padB);

  if (pId >= totalPaddingSize) {
    return;
  }

  /* calculate outer image x and y coordinates */
  int pId1 = max(0, pId - w * padT);
  int pId2 = max(0, pId1 - h_ * (padL + padR));

  int y = min(pId / w, padT) + min(pId1 / (padL + padR), h_) + pId2 / w;

  int x_offset = (pId1 - pId2) %
                 (padL + padR); /* non-zero when point is just to the right or
                                   left of input image */
  int x = (pId - pId1) % w /* non-zero when point is above the input image  */
          + x_offset % padL + (x_offset / padL) * (w_ + padL) +
          pId2 % w; /* non-zero when point is below input image */

  /* inner image x and y coordinates */
  int x_ = 2 * padL + abs(x - padL) - abs(x - (w_ + padL - 1)) - x + w_ - 1;
  int y_ = 2 * padT + abs(y - padT) - abs(y - (h_ + padT - 1)) - y + h_ - 1;

  scalar_t toCopy = input[batch][plane][y_][x_];
  input[batch][plane][y][x] = toCopy;
}

at::Tensor reflectionpad_inplace_cuda_forward(at::Tensor input_t, int padL,
                                              int padR, int padT, int padB) {

  auto sz = at::check_intlist<4>(input_t.sizes(), "name", 0);
  const int sizes[4] = {sz[0], sz[1], sz[2], sz[3]};

  int dimh = 2;
  int dimw = 3;
  int inputH = sizes[2];
  int inputW = sizes[3];

  int w_ = inputW - padL - padR;
  if (!(padL < w_ && padR < w_)) {
    std::stringstream ss;
    ss << "Padding size should not exceed corresponding input width " << w_
       << ", "
       << "but got: padding (" << padL << ", " << padR << ") at dimension "
       << dimw << ". Image size (" << inputW << ", " << inputH << ").";
    throw std::runtime_error(ss.str());
  }

  int h_ = inputH - padT - padB;
  if (!(padT < h_ && padB < h_)) {
    std::stringstream ss;
    ss << "Padding size should not exceed corresponding input dimension, "
       << "but got: padding (" << padT << ", " << padB << ") at dimension "
       << dimh << ". ";
  }

  int unpaddedWidth = inputW - padL - padR;
  int totalPaddingSize = inputH * (padL + padR) + unpaddedWidth * (padT + padB);
  if (totalPaddingSize == 0) {
    return input_t;
  }

  AT_DISPATCH_FLOATING_TYPES(
      input_t.type(), "reflectionpad_inplace_cuda_forward", ([&] {
        auto d = input_t.data<scalar_t>();
        THCDeviceTensor<scalar_t, 4> input_thc =
            THCDeviceTensor<scalar_t, 4>(d, sizes);

        dim3 gridSize(THCCeilDiv(totalPaddingSize, 256), input_thc.getSize(1),
                      input_thc.getSize(0));
        dim3 blockSize(totalPaddingSize > 256 ? 256 : totalPaddingSize);

        SpatialReflectionPadding_updateOutputInPlace<scalar_t>
            <<<gridSize, blockSize, 0>>>(input_thc, padL, padR, padT, padB);
        THCudaCheck(cudaGetLastError());
      }));

  return input_t;
}

template <typename scalar_t>
__global__ void SpatialReflectionPadding_updateGradInputInPlace(
    THCDeviceTensor<scalar_t, 4> gradInput,
    THCDeviceTensor<scalar_t, 4> gradOutput, int padL, int padR, int padT,
    int padB) {

  int pId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  int w = gradOutput.getSize(3);
  int h = gradOutput.getSize(2);
  int w_ = w - padL - padR; /* inner width */
  int h_ = h - padT - padB; /* inner height */
  int totalPaddingSize = h * (padL + padR) + w_ * (padT + padB);

  if (pId >= totalPaddingSize) {
    return;
  }

  /* calculate outer image x and y coordinates */
  int pId1 = max(0, pId - w * padT);
  int pId2 = max(0, pId1 - h_ * (padL + padR));

  int y = min(pId / w, padT) + min(pId1 / (padL + padR), h_) + pId2 / w;

  int x_offset = (pId1 - pId2) %
                 (padL + padR); /* non-zero when point is just to the right or
                                   left of input image */
  int x = (pId - pId1) % w /* non-zero when point is above the input image  */
          + x_offset % padL + (x_offset / padL) * (w_ + padL) +
          pId2 % w; /* non-zero when point is below input image */

  /* inner image x and y coordinates */
  int x_ = 2 * padL + abs(x - padL) - abs(x - (w_ + padL - 1)) - x + w_ - 1;
  int y_ = 2 * padT + abs(y - padT) - abs(y - (h_ + padT - 1)) - y + h_ - 1;

  // Add padding gradient to inner image gradient
  scalar_t toCopy = gradOutput[batch][plane][y][x];
  atomicAdd(&gradInput[batch][plane][y_][x_], toCopy);
  // Remove padding gradient
  scalar_t zero = ScalarConvert<int, scalar_t>::to(0);
  gradInput[batch][plane][y][x] = zero;
}

at::Tensor reflectionpad_inplace_cuda_backward(at::Tensor grad_output, int padL,
                                               int padR, int padT, int padB) {
  auto sz = at::check_intlist<4>(grad_output.sizes(), "name", 0);
  const int sizes[4] = {sz[0], sz[1], sz[2], sz[3]};
  int height = sizes[2];
  int width = sizes[3];

  auto grad_input = grad_output.clone();

  int unpaddedWidth = width - padL - padR;
  int totalPaddingSize = height * (padL + padR) + unpaddedWidth * (padT + padB);
  if (totalPaddingSize == 0) {
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(
      grad_output.type(), "reflectionpad_inplace_cuda_backward", ([&] {
        auto d_input = grad_input.data<scalar_t>();
        auto d_output = grad_output.data<scalar_t>();

        THCDeviceTensor<scalar_t, 4> devGradInput =
            THCDeviceTensor<scalar_t, 4>(d_input, sizes);
        THCDeviceTensor<scalar_t, 4> devGradOutput =
            THCDeviceTensor<scalar_t, 4>(d_output, sizes);

        dim3 gridSize(THCCeilDiv(totalPaddingSize, 256),
                      devGradOutput.getSize(1), devGradOutput.getSize(0));
        dim3 blockSize(totalPaddingSize > 256 ? 256 : totalPaddingSize);

        SpatialReflectionPadding_updateGradInputInPlace<scalar_t>
            <<<gridSize, blockSize, 0>>>(devGradInput, devGradOutput, padT,
                                         padB, padL, padR);
        THCudaCheck(cudaGetLastError());
      }));

  return grad_input;
}
