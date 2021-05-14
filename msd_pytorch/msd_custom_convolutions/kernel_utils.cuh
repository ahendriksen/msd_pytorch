#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

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
