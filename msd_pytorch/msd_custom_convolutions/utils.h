#pragma once


#define CudaCheck(err)                                                          \
    do {                                                                        \
        if(err != cudaSuccess) {                                                \
            AT_ERROR("Cuda error=", err, " : ", cudaGetErrorString(err));       \
        }                                                                       \
    } while(0)                                                                  \


/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}
