#ifndef HXINFER_TEMPLATED_KERNELS_H
#define HXINFER_TEMPLATED_KERNELS_H
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <iostream>

namespace hxinfer {

__device__ inline float to_float(float v) { return v; }
__device__ inline float to_float(__half v) { return __half2float(v); }

template<typename T>
__device__ inline T from_float(float v);

template<>
__device__ inline float from_float<float>(float v) { return v; }

template<>
__device__ inline __half from_float<__half>(float v) { return __float2half(v); }

template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = to_float(a[idx]), vb = to_float(b[idx]);
        out[idx] = from_float<T>(va + vb);
    }
}

template<typename T>
__global__ void mul_kernel(const T* a, const T* b, T* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = to_float(a[idx]), vb = to_float(b[idx]);
        out[idx] = from_float<T>(va * vb);
    }
}

template<typename T>
__global__ void silu_kernel(const T* in, T* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = to_float(in[idx]);
        out[idx] = from_float<T>(val / (1.0f + expf(-val)));
    }
}

} // namespace hxinfer
#endif
