#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"
#include "cmath"
namespace hxinfer{

// ===== Helper: YaRN blended inverse frequency =====
static float find_correction_dim(int num_rotations, int dim, float base, int max_pos){
    return (dim * logf(max_pos / (num_rotations * 2.0f * M_PI))) / (2.0f * logf(base));
}

static void compute_yarn_scale_multipliers(float* out, int half_dim, int head_dim,
                                           float factor, float base,
                                           float beta_fast, float beta_slow,
                                           int orig_max_pos){
    float low  = find_correction_dim((int)beta_fast, head_dim, base, orig_max_pos);
    float high = find_correction_dim((int)beta_slow, head_dim, base, orig_max_pos);
    low  = fmaxf(low, 0.0f);
    high = fminf(high, (float)(half_dim - 1));
    float high_minus_low = high - low;
    if(high_minus_low <= 0.0f) high_minus_low = 1.0f;

    for(int d = 0; d < half_dim; d++){
        float ramp = (float)(d) - low;
        ramp = ramp / high_minus_low;
        ramp = fmaxf(0.0f, fminf(1.0f, ramp));
        out[d] = 1.0f - ramp * (1.0f - 1.0f / factor);
    }
}

static float yarn_attention_factor(float factor){
    if(factor <= 1.0f) return 1.0f;
    return 0.1f * logf(factor) + 1.0f;
}

// ===== NeoX-style RoPE =====
// For each head of size head_dim, split into two halves:
//   x1 = data[0 .. half-1]
//   x2 = data[half .. head_dim-1]
// freq[d] = 1 / base^(2*d / head_dim),  d = 0..half-1
// angle = step * freq[d]
// out1[d] = x1[d]*cos - x2[d]*sin
// out2[d] = x2[d]*cos + x1[d]*sin

__global__ void rope_kernel_cuda_neox(float* data, int num_heads, int head_dim, int step,
                                      const float* yarn_scale, float attn_factor, float base){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = num_heads * half_dim;

    if(idx < total){
        int head_idx = idx / half_dim;
        int d = idx % half_dim;

        float scale = 1.0f / powf(base, static_cast<float>(2 * d) / head_dim);
        if(yarn_scale){
            scale *= yarn_scale[d];
        }
        float angle = step * scale;
        float cos_val = cosf(angle) * attn_factor;
        float sin_val = sinf(angle) * attn_factor;

        float* base_ptr = data + head_idx * head_dim;
        float v1 = base_ptr[d];           // first half
        float v2 = base_ptr[d + half_dim]; // second half
        base_ptr[d]            = v1 * cos_val - v2 * sin_val;
        base_ptr[d + half_dim] = v2 * cos_val + v1 * sin_val;
    }
}

__global__ void rope_kernel_cuda_neox_fp16(__half* data, int num_heads, int head_dim, int step,
                                           const float* yarn_scale, float attn_factor, float base){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = num_heads * half_dim;

    if(idx < total){
        int head_idx = idx / half_dim;
        int d = idx % half_dim;

        float scale = 1.0f / powf(base, static_cast<float>(2 * d) / head_dim);
        if(yarn_scale){
            scale *= yarn_scale[d];
        }
        float angle = step * scale;
        float cos_val = cosf(angle) * attn_factor;
        float sin_val = sinf(angle) * attn_factor;

        __half* base_ptr = data + head_idx * head_dim;
        float v1 = __half2float(base_ptr[d]);           // first half
        float v2 = __half2float(base_ptr[d + half_dim]); // second half
        base_ptr[d]            = __float2half(v1 * cos_val - v2 * sin_val);
        base_ptr[d + half_dim] = __float2half(v2 * cos_val + v1 * sin_val);
    }
}

void rope_cuda(std::shared_ptr<Tensor>& q, std::shared_ptr<Tensor>& k,
               ModelConfig& config, int step, float base){
    if(q->tensor_device_type() != DeviceType::kDeviceCUDA ||
        k->tensor_device_type() != DeviceType::kDeviceCUDA){
        std::cerr << "[Fatal Error] rope_cuda expects CUDA Tensors!" << std::endl;
        return;
    }
    int head_dim = config.dim / config.head;
    int threads = 256;

    float* d_yarn_scale = nullptr;
    float  attn_factor  = 1.0f;
    int half_dim = head_dim / 2;

    if(config.rope_use_yarn){
        std::vector<float> h_yarn_scale(half_dim);
        compute_yarn_scale_multipliers(h_yarn_scale.data(), half_dim, head_dim,
                                       config.rope_factor, base,
                                       config.rope_beta_fast, config.rope_beta_slow,
                                       config.rope_orig_max_pos);
        cudaMalloc(&d_yarn_scale, half_dim * sizeof(float));
        cudaMemcpy(d_yarn_scale, h_yarn_scale.data(), half_dim * sizeof(float), cudaMemcpyHostToDevice);
        attn_factor = yarn_attention_factor(config.rope_factor);
    }

    if(q->tensor_data_type() == DataType::kDataTypeFP16){
        __half* q_data = q->tensor_data_ptr<__half>();
        __half* k_data = k->tensor_data_ptr<__half>();

        int q_total = config.head * half_dim;
        int q_blocks = (q_total + threads - 1) / threads;
        rope_kernel_cuda_neox_fp16<<<q_blocks, threads>>>(q_data, config.head, head_dim, step,
                                                           d_yarn_scale, attn_factor, base);

        int k_total = config.kv_head * half_dim;
        int k_blocks = (k_total + threads - 1) / threads;
        rope_kernel_cuda_neox_fp16<<<k_blocks, threads>>>(k_data, config.kv_head, head_dim, step,
                                                           d_yarn_scale, attn_factor, base);
    } else {
        float* q_data = q->tensor_data_ptr<float>();
        float* k_data = k->tensor_data_ptr<float>();

        int q_total = config.head * half_dim;
        int q_blocks = (q_total + threads - 1) / threads;
        rope_kernel_cuda_neox<<<q_blocks, threads>>>(q_data, config.head, head_dim, step,
                                                      d_yarn_scale, attn_factor, base);

        int k_total = config.kv_head * half_dim;
        int k_blocks = (k_total + threads - 1) / threads;
        rope_kernel_cuda_neox<<<k_blocks, threads>>>(k_data, config.kv_head, head_dim, step,
                                                      d_yarn_scale, attn_factor, base);
    }

    if(d_yarn_scale) cudaFree(d_yarn_scale);
}

} // namespace hxinfer
