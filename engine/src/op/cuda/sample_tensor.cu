#include "op/math_ops.h"
#include "tensor/tensor.h"
#include "base/allocator.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>

namespace hxinfer{

// ==================== GPU softmax kernel ====================
__global__ void gpu_softmax_kernel(const float* logits, float* probs,
                                    int vocab_size, float inv_temp){
    int tid = threadIdx.x;
    int block = blockDim.x;

    __shared__ float s_vals[256];

    // Step 1: find max
    float local_max = -3.4e38f;
    for(int i = tid; i < vocab_size; i += block){
        float v = logits[i] * inv_temp;
        if(v > local_max) local_max = v;
    }
    s_vals[tid] = local_max;
    __syncthreads();
    for(int s = block/2; s > 0; s >>= 1){
        if(tid < s && s_vals[tid+s] > s_vals[tid]) s_vals[tid] = s_vals[tid+s];
        __syncthreads();
    }
    float g_max = s_vals[0];
    __syncthreads();

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for(int i = tid; i < vocab_size; i += block){
        float v = expf(logits[i] * inv_temp - g_max);
        probs[i] = v;
        local_sum += v;
    }
    s_vals[tid] = local_sum;
    __syncthreads();
    for(int s = block/2; s > 0; s >>= 1){
        if(tid < s) s_vals[tid] += s_vals[tid+s];
        __syncthreads();
    }
    float g_sum = s_vals[0];
    __syncthreads();

    // Step 3: normalize
    float inv_sum = 1.0f / g_sum;
    for(int i = tid; i < vocab_size; i += block){
        probs[i] *= inv_sum;
    }
}

// FP16 logits → FP32 probs softmax kernel
__global__ void gpu_softmax_kernel_fp16(const __half* logits, float* probs,
                                         int vocab_size, float inv_temp){
    int tid = threadIdx.x;
    int block = blockDim.x;

    __shared__ float s_vals[256];

    // Step 1: find max
    float local_max = -3.4e38f;
    for(int i = tid; i < vocab_size; i += block){
        float v = __half2float(logits[i]) * inv_temp;
        if(v > local_max) local_max = v;
    }
    s_vals[tid] = local_max;
    __syncthreads();
    for(int s = block/2; s > 0; s >>= 1){
        if(tid < s && s_vals[tid+s] > s_vals[tid]) s_vals[tid] = s_vals[tid+s];
        __syncthreads();
    }
    float g_max = s_vals[0];
    __syncthreads();

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for(int i = tid; i < vocab_size; i += block){
        float v = expf(__half2float(logits[i]) * inv_temp - g_max);
        probs[i] = v;
        local_sum += v;
    }
    s_vals[tid] = local_sum;
    __syncthreads();
    for(int s = block/2; s > 0; s >>= 1){
        if(tid < s) s_vals[tid] += s_vals[tid+s];
        __syncthreads();
    }
    float g_sum = s_vals[0];
    __syncthreads();

    // Step 3: normalize
    float inv_sum = 1.0f / g_sum;
    for(int i = tid; i < vocab_size; i += block){
        probs[i] *= inv_sum;
    }
}

static std::mt19937& get_rng(){
    static std::mt19937 rng(std::random_device{}());
    return rng;
}

int sample_tensor(const std::shared_ptr<Tensor>& logits, float temperature, float top_p){
    size_t vocab_size = logits->tensor_total_elements();

    std::vector<float> probs(vocab_size);

    bool logits_fp16 = (logits->tensor_data_type() == DataType::kDataTypeFP16);

    if(logits->tensor_device_type() == DeviceType::kDeviceCUDA){
        static float* d_probs = nullptr;
        static int    d_probs_size = 0;
        if(d_probs_size < (int)vocab_size){
            if(d_probs) cudaFree(d_probs);
            cudaMalloc(&d_probs, vocab_size * sizeof(float));
            d_probs_size = (int)vocab_size;
        }

        float inv_temp = 1.0f / temperature;

        int threads = 256;
        size_t shared_bytes = 256 * sizeof(float);

        if(logits_fp16){
            const __half* d_logits = logits->tensor_data_ptr<__half>();
            gpu_softmax_kernel_fp16<<<1, threads, shared_bytes>>>(
                d_logits, d_probs, (int)vocab_size, inv_temp);
        } else {
            const float* d_logits = logits->tensor_data_ptr<float>();
            gpu_softmax_kernel<<<1, threads, shared_bytes>>>(
                d_logits, d_probs, (int)vocab_size, inv_temp);
        }

        cudaMemcpy(probs.data(), d_probs, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

    } else {
        if(logits_fp16){
            const __half* ptr = logits->tensor_data_ptr<__half>();
            for(size_t i=0; i<vocab_size; i++) probs[i] = __half2float(ptr[i]) / temperature;
        } else {
            const float* ptr = logits->tensor_data_ptr<float>();
            std::copy(ptr, ptr + vocab_size, probs.begin());
            for(auto& v : probs) v /= temperature;
        }
        float max_val = *std::max_element(probs.begin(), probs.end());
        float sum = 0.f;
        for(auto& v : probs){ v = std::exp(v - max_val); sum += v; }
        for(auto& v : probs) v /= sum;
    }

    // CPU 端：正确的 top-p 采样
    std::vector<std::pair<float,int>> indexed(vocab_size);
    for(size_t i=0; i<vocab_size; i++) indexed[i] = {probs[i], (int)i};
    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    float cumulative = 0.f;
    int cutoff = 0;
    for(; cutoff < (int)vocab_size; cutoff++){
        cumulative += indexed[cutoff].first;
        if(cumulative >= top_p) break;
    }
    for(int i = cutoff+1; i < (int)vocab_size; i++) indexed[i].first = 0.f;

    float new_sum = 0.f;
    for(int i=0; i<=cutoff; i++) new_sum += indexed[i].first;
    float inv_new_sum = 1.0f / new_sum;
    for(int i=0; i<=cutoff; i++) indexed[i].first *= inv_new_sum;

    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(get_rng());
    cumulative = 0.f;
    for(int i=0; i<=cutoff; i++){
        cumulative += indexed[i].first;
        if(r <= cumulative) return indexed[i].second;
    }
    return indexed[0].second;
}

} // namespace hxinfer
