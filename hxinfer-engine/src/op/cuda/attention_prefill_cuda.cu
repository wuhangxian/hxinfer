#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"
#include <cfloat>

namespace hxinfer {

// Simple 2-pass parallel attention, no tiling
// Pass 1: compute ALL scores in shared memory, find max
// Pass 2: compute exp, sum, and weighted V
// Requires: seq_len <= 4096 (shared memory = (4096+128)*4 = 17KB < 48KB)

__global__ void attention_prefill_kernel_fp16(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ output,
    int S, int num_heads, int num_kv_heads, int head_dim,
    int kv_dim, float scale, int kv_group_size)
{
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int kv_head_idx = head_idx / kv_group_size;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const __half* q = Q + seq_idx * num_heads * head_dim + head_idx * head_dim;

    // Shared memory: scores[S] + reduce[block_size]
    extern __shared__ float shared[];
    float* scores = shared;        // [S]
    float* reduce = shared + S;   // [block_size]

    // Step 1: Compute all scores Q[seq_idx] . K[k]^T for k=0..seq_idx
    for (int k = tid; k <= seq_idx; k += block_size) {
        const __half* kk = K + k * kv_dim + kv_head_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q[d]) * __half2float(kk[d]);
        }
        scores[k] = dot * scale;
    }
    __syncthreads();

    // Step 2: Find max
    float local_max = -FLT_MAX;
    for (int k = tid; k <= seq_idx; k += block_size) {
        if (scores[k] > local_max) local_max = scores[k];
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s && reduce[tid + s] > reduce[tid]) reduce[tid] = reduce[tid + s];
        __syncthreads();
    }
    float max_val = reduce[0];
    __syncthreads();

    // Step 3: Compute exp and sum
    float local_sum = 0.0f;
    for (int k = tid; k <= seq_idx; k += block_size) {
        scores[k] = expf(scores[k] - max_val);
        local_sum += scores[k];
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float sum_val = reduce[0];
    if (sum_val == 0.0f) sum_val = 1e-30f;
    __syncthreads();

    // Step 4: Normalize
    for (int k = tid; k <= seq_idx; k += block_size) {
        scores[k] /= sum_val;
    }
    __syncthreads();

    // Step 5: Compute weighted V
    for (int d = tid; d < head_dim; d += block_size) {
        float sum = 0.0f;
        for (int k = 0; k <= seq_idx; k++) {
            sum += scores[k] * __half2float(V[k * kv_dim + kv_head_idx * head_dim + d]);
        }
        output[seq_idx * num_heads * head_dim + head_idx * head_dim + d] = __float2half(sum);
    }
}

void attention_prefill_cuda(
    const std::shared_ptr<Tensor>& Q,
    const std::shared_ptr<Tensor>& K,
    const std::shared_ptr<Tensor>& V,
    std::shared_ptr<Tensor>& output,
    ModelConfig& config, int seq_len)
{
    int dim = config.dim;
    int head = config.head;
    int head_dim = dim / head;
    int kv_head = config.kv_head > 0 ? config.kv_head : head;
    int kv_dim = kv_head * head_dim;
    int kv_group_size = head / kv_head;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    bool fp16 = (Q->tensor_data_type() == DataType::kDataTypeFP16);
    if (!fp16) return;

    int threads = 128;
    dim3 grid(seq_len, head);
    size_t shared_bytes = (seq_len + threads) * sizeof(float);

    // Set opt-in for large shared memory
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            (const void*)attention_prefill_kernel_fp16,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            65536);
        attr_set = true;
    }

    const __half* d_q = Q->tensor_data_ptr<__half>();
    const __half* d_k = K->tensor_data_ptr<__half>();
    const __half* d_v = V->tensor_data_ptr<__half>();
    __half* d_out = output->tensor_data_ptr<__half>();

    attention_prefill_kernel_fp16<<<grid, threads, shared_bytes>>>(
        d_q, d_k, d_v, d_out,
        seq_len, head, kv_head, head_dim,
        kv_dim, scale, kv_group_size);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[attention_prefill] error: " << cudaGetErrorString(err) << std::endl;
    }
}

} // namespace hxinfer
