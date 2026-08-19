#include "cuda_runtime.h"
#include "cuda_fp16.h"

namespace hxinfer {

__global__ void add_bias_fp16_kernel(__half* data, const __half* bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __hadd(data[idx], bias[idx]);
    }
}

__global__ void add_bias_fp32_kernel(float* data, const float* bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += bias[idx];
    }
}

extern "C" void add_bias_cuda_fp16(__half* data, const __half* bias, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bias_fp16_kernel<<<blocks, threads>>>(data, bias, n);
}

extern "C" void add_bias_cuda_fp32(float* data, const float* bias, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bias_fp32_kernel<<<blocks, threads>>>(data, bias, n);
}

} // namespace hxinfer
