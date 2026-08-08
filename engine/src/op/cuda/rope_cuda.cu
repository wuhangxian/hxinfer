#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"
#include "cmath"
namespace hxinfer{

// ===== Helper: YaRN blended inverse frequency =====
// Pre-compute per-dimension scale factors on host, upload to constant/device memory.
// inv_freq[d] = (1/(factor*base^(2d/dim))) * ramp[d] + (1/base^(2d/dim)) * (1 - ramp[d])
// We encode this as a per-dim multiplier on the standard scale = 1/base^(2d/dim):
//   effective_scale[d] = scale[d] * (ramp[d]/factor + (1-ramp[d]))
//   effective_scale[d] = scale[d] * (1 - ramp[d]*(1 - 1/factor))

static float find_correction_dim(int num_rotations, int dim, float base, int max_pos){
    return (dim * logf(max_pos / (num_rotations * 2.0f * M_PI))) / (2.0f * logf(base));
}

// Build YaRN per-dim scale multipliers (host side)
// out_scale[d] = (1 - ramp[d]*(1 - 1/factor))  for d = 0..half_dim-1
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
        // ramp=0 for low-d (high-freq) → extrapolation → multiplier = 1.0
        // ramp=1 for high-d (low-freq) → interpolation → multiplier = 1/factor
        out[d] = 1.0f - ramp * (1.0f - 1.0f / factor);
    }
}

// YaRN attention factor: mscale = 0.1 * ln(factor) + 1.0
static float yarn_attention_factor(float factor){
    if(factor <= 1.0f) return 1.0f;
    return 0.1f * logf(factor) + 1.0f;
}

// ===== FP32 RoPE kernel (standard + YaRN) =====
// yarn_scale[d] is pre-computed multiplier for dimension d
__global__ void rope_kernel_cuda(float* data, int num_heads, int head_dim, int step,
                                 const float* yarn_scale, float attn_factor, float base){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head_dim = head_dim / 2;
    int total_pairs = num_heads * half_head_dim;

    if(idx < total_pairs){
        int head_idx = idx / half_head_dim;
        int pair_idx = idx % half_head_dim;
        int d = pair_idx * 2;

        float scale = 1.0f / powf(base, static_cast<float>(d) / head_dim);
        if(yarn_scale){
            scale *= yarn_scale[pair_idx];
        }
        float angle = step * scale;
        float cos_val = cosf(angle) * attn_factor;
        float sin_val = sinf(angle) * attn_factor;

        float* ptr = data + head_idx * head_dim + d;
        float v0 = ptr[0];
        float v1 = ptr[1];
        ptr[0] = v0 * cos_val - v1 * sin_val;
        ptr[1] = v0 * sin_val + v1 * cos_val;
    }
}

// ===== FP16 RoPE kernel (standard + YaRN) =====
__global__ void rope_kernel_cuda_fp16(__half* data, int num_heads, int head_dim, int step,
                                      const float* yarn_scale, float attn_factor, float base){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head_dim = head_dim / 2;
    int total_pairs = num_heads * half_head_dim;

    if(idx < total_pairs){
        int head_idx = idx / half_head_dim;
        int pair_idx = idx % half_head_dim;
        int d = pair_idx * 2;

        float scale = 1.0f / powf(base, static_cast<float>(d) / head_dim);
        if(yarn_scale){
            scale *= yarn_scale[pair_idx];
        }
        float angle = step * scale;
        float cos_val = cosf(angle) * attn_factor;
        float sin_val = sinf(angle) * attn_factor;

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
        std::cerr << "[Fatal Error] rope_cuda expects CUDA Tensors!" << std::endl;
        return;
    }
    int head_dim = config.dim / config.head;
    int threads = 256;

    // YaRN: pre-compute per-dim scale multipliers on host, upload to GPU
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

        int q_pairs = config.head * (head_dim / 2);
        int q_blocks = (q_pairs + threads - 1) / threads;
        rope_kernel_cuda_fp16<<<q_blocks, threads>>>(q_data, config.head, head_dim, step,
                                                      d_yarn_scale, attn_factor, base);

        int k_pairs = config.kv_head * (head_dim / 2);
        int k_blocks = (k_pairs + threads - 1) / threads;
        rope_kernel_cuda_fp16<<<k_blocks, threads>>>(k_data, config.kv_head, head_dim, step,
                                                      d_yarn_scale, attn_factor, base);
    } else {
        float* q_data = q->tensor_data_ptr<float>();
        float* k_data = k->tensor_data_ptr<float>();

        int q_pairs = config.head * (head_dim / 2);
        int q_blocks = (q_pairs + threads - 1) / threads;
        rope_kernel_cuda<<<q_blocks, threads>>>(q_data, config.head, head_dim, step,
                                                d_yarn_scale, attn_factor, base);

        int k_pairs = config.kv_head * (head_dim / 2);
        int k_blocks = (k_pairs + threads - 1) / threads;
        rope_kernel_cuda<<<k_blocks, threads>>>(k_data, config.kv_head, head_dim, step,
                                                d_yarn_scale, attn_factor, base);
    }

    if(d_yarn_scale) cudaFree(d_yarn_scale);
}

} // namespace hxinfer
