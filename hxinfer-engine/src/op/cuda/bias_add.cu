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

__global__ void add_bias_broadcast_fp16_kernel(__half* data, const __half* bias, int total, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx] = __hadd(data[idx], bias[idx % N]);
    }
}

__global__ void add_bias_broadcast_fp32_kernel(float* data, const float* bias, int total, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        data[idx] += bias[idx % N];
    }
}

// CLAMP kernel: clamp FP16 data to valid range, replace NaN with 0
__global__ void clamp_fp16_kernel(__half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(data[idx]);
        if (std::isnan(v)) v = 0.0f;
        v = fmaxf(-65504.0f, fminf(65504.0f, v));
        data[idx] = __float2half(v);
    }
}

__global__ void clamp_fp32_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (std::isnan(data[idx])) data[idx] = 0.0f;
        data[idx] = fmaxf(-3.4e38f, fminf(3.4e38f, data[idx]));
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

extern "C" void add_bias_broadcast_fp16(__half* data, const __half* bias, int total, int N) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_broadcast_fp16_kernel<<<blocks, threads>>>(data, bias, total, N);
}

extern "C" void add_bias_broadcast_fp32(float* data, const float* bias, int total, int N) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_broadcast_fp32_kernel<<<blocks, threads>>>(data, bias, total, N);
}

extern "C" void clamp_fp16(__half* data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clamp_fp16_kernel<<<blocks, threads>>>(data, n);
}

extern "C" void clamp_fp32(float* data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clamp_fp32_kernel<<<blocks, threads>>>(data, n);
}

} // namespace hxinfer
