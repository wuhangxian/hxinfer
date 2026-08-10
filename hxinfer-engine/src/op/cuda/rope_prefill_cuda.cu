#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cmath"

namespace hxinfer {

// Standard RoPE only, no YaRN - matches rope_cuda.cu (decode path)
__global__ void rope_kernel_single_fp16(
    __half* data, int num_heads, int head_dim, int step,
    const float* yarn_scale, float attn_factor, float base)
{
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

        __half* ptr = data + head_idx * head_dim + d;
        float v0 = __half2float(ptr[0]);
        float v1 = __half2float(ptr[1]);
        ptr[0] = __float2half(v0 * cos_val - v1 * sin_val);
        ptr[1] = __float2half(v0 * sin_val + v1 * cos_val);
    }
}

void rope_prefill_cuda(std::shared_ptr<Tensor>& q, std::shared_ptr<Tensor>& k,
                       ModelConfig& config, int seq_len, int start_pos, float base)
{
    int head_dim = config.dim / config.head;
    int threads = 256;

    if(q->tensor_data_type() == DataType::kDataTypeFP16){
        __half* q_data = q->tensor_data_ptr<__half>();
        __half* k_data = k->tensor_data_ptr<__half>();

        int q_pairs = config.head * (head_dim / 2);
        int q_blocks = (q_pairs + threads - 1) / threads;
        int k_pairs = config.kv_head * (head_dim / 2);
        int k_blocks = (k_pairs + threads - 1) / threads;

        for(int s = 0; s < seq_len; s++){
            int step = start_pos + s;
            rope_kernel_single_fp16<<<q_blocks, threads>>>(
                q_data + s * config.head * head_dim, config.head, head_dim, step,
                nullptr, 1.0f, base);
            rope_kernel_single_fp16<<<k_blocks, threads>>>(
                k_data + s * config.kv_head * head_dim, config.kv_head, head_dim, step,
                nullptr, 1.0f, base);
        }
    }

    cudaDeviceSynchronize();
}

} // namespace hxinfer
