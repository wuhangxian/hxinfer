#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cmath"

namespace hxinfer{

// No YaRN at all - just standard RoPE with base
__global__ void rope_kernel_cuda(float* data, int num_heads, int head_dim, int step,
                                 const float* yarn_scale, float attn_factor, float base){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head_dim = head_dim / 2;
    int total_pairs = num_heads * half_head_dim;
    if(idx < total_pairs){
        int head_idx = idx / half_head_dim;
        int pair_idx = idx % half_head_dim;
        int d = pair_idx * 2;
        float scale = 1.0f / powf(base, static_cast<float>(d) / static_cast<float>(head_dim));
        // NO yarn_scale, NO attn_factor
        float angle = step * scale;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        float* ptr = data + head_idx * head_dim + d;
        float v0 = ptr[0];
        float v1 = ptr[1];
        ptr[0] = v0 * cos_val - v1 * sin_val;
        ptr[1] = v0 * sin_val + v1 * cos_val;
    }
}

__global__ void rope_kernel_cuda_fp16(__half* data, int num_heads, int head_dim, int step,
                                      const float* yarn_scale, float attn_factor, float base){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head_dim = head_dim / 2;
    int total_pairs = num_heads * half_head_dim;
    if(idx < total_pairs){
        int head_idx = idx / half_head_dim;
        int pair_idx = idx % half_head_dim;
        int d = pair_idx * 2;
        float scale = 1.0f / powf(base, static_cast<float>(d) / static_cast<float>(head_dim));
        float angle = step * scale;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        __half* ptr = data + head_idx * head_dim + d;
        float v0 = __half2float(ptr[0]);
        float v1 = __half2float(ptr[1]);
        ptr[0] = __float2half(v0 * cos_val - v1 * sin_val);
        ptr[1] = __float2half(v0 * sin_val + v1 * cos_val);
    }
}

void rope_cuda(std::shared_ptr<Tensor>& q, std::shared_ptr<Tensor>& k,
               ModelConfig& config, int step, float base){
    if(q->tensor_device_type() != DeviceType::kDeviceCUDA ||
        k->tensor_device_type() != DeviceType::kDeviceCUDA){
        return;
    }
    int head_dim = config.dim / config.head;
    int threads = 256;
    // NO YaRN - skip yarn_scale and attn_factor entirely
    if(q->tensor_data_type() == DataType::kDataTypeFP16){
        __half* q_data = q->tensor_data_ptr<__half>();
        __half* k_data = k->tensor_data_ptr<__half>();
        int q_pairs = config.head * (head_dim / 2);
        int q_blocks = (q_pairs + threads - 1) / threads;
        rope_kernel_cuda_fp16<<<q_blocks, threads>>>(q_data, config.head, head_dim, step,
                                                      nullptr, 1.0f, base);
        int k_pairs = config.kv_head * (head_dim / 2);
        int k_blocks = (k_pairs + threads - 1) / threads;
        rope_kernel_cuda_fp16<<<k_blocks, threads>>>(k_data, config.kv_head, head_dim, step,
                                                      nullptr, 1.0f, base);
    } else {
        float* q_data = q->tensor_data_ptr<float>();
        float* k_data = k->tensor_data_ptr<float>();
        int q_pairs = config.head * (head_dim / 2);
        int q_blocks = (q_pairs + threads - 1) / threads;
        rope_kernel_cuda<<<q_blocks, threads>>>(q_data, config.head, head_dim, step,
                                                nullptr, 1.0f, base);
        int k_pairs = config.kv_head * (head_dim / 2);
        int k_blocks = (k_pairs + threads - 1) / threads;
        rope_kernel_cuda<<<k_blocks, threads>>>(k_data, config.kv_head, head_dim, step,
                                                nullptr, 1.0f, base);
    }
}

} // namespace hxinfer
