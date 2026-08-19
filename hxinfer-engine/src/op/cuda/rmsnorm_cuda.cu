
#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"
namespace hxinfer{
    // FP32 RMSNorm kernel
    __global__ void rmsnorm_kernel_cuda(float *out_data,const float* in_data,
                                        const float *weight,int hidden_dim,float eps){
        int tid=threadIdx.x;
        int bid=blockIdx.x;

        const float *current_in=in_data+bid*hidden_dim;
        float *current_out=out_data+bid*hidden_dim;

        float local_sum=0;
        for(int i=tid;i<hidden_dim;i=i+blockDim.x){
            local_sum=local_sum+current_in[i]*current_in[i];
        }
        extern __shared__ float s_sum[];
        s_sum[tid]=local_sum;
        __syncthreads();

        for(int stride=blockDim.x/2;stride>0;stride>>=1){
            if(tid<stride){
                s_sum[tid]=s_sum[tid]+s_sum[tid+stride];
            }
            __syncthreads();
        }
        __shared__ float s_inv_rms;
        if(tid==0){
            s_inv_rms= rsqrtf(s_sum[0]/hidden_dim+eps);
        }
        __syncthreads();
        for(int i=tid;i<hidden_dim;i += blockDim.x){
            current_out[i]=current_in[i]*s_inv_rms*weight[i];
        }
    }

    // FP16 RMSNorm kernel: input/output FP16, weight FP32 (small, loaded as FP32 for precision)
    __global__ void rmsnorm_kernel_cuda_fp16(__half *out_data,const __half* in_data,
                                             const float *weight,int hidden_dim,float eps){
        int tid=threadIdx.x;
        int bid=blockIdx.x;

        const __half *current_in=in_data+bid*hidden_dim;
        __half *current_out=out_data+bid*hidden_dim;

        float local_sum=0;
        for(int i=tid;i<hidden_dim;i=i+blockDim.x){
            float val=__half2float(current_in[i]);
            local_sum+=val*val;
        }
        extern __shared__ float s_sum[];
        s_sum[tid]=local_sum;
        __syncthreads();

        for(int stride=blockDim.x/2;stride>0;stride>>=1){
            if(tid<stride){
                s_sum[tid]=s_sum[tid]+s_sum[tid+stride];
            }
            __syncthreads();
        }
        __shared__ float s_inv_rms;
        if(tid==0){
            s_inv_rms= rsqrtf(s_sum[0]/hidden_dim+eps);
        }
        __syncthreads();
        for(int i=tid;i<hidden_dim;i += blockDim.x){
            current_out[i]=__float2half(__half2float(current_in[i])*s_inv_rms*weight[i]);
        }
    }

    void rmsnorm_cuda(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                      std::shared_ptr<Tensor>& output,float eps){
        if(input->tensor_device_type() != DeviceType::kDeviceCUDA ||
           weight->tensor_device_type() != DeviceType::kDeviceCUDA ||
           output->tensor_device_type() != DeviceType::kDeviceCUDA){
            std::cerr << "[Fatal Error] rmsnorm_cuda expects CUDA Tensors!" << std::endl;
            return;
        }
        size_t  total_elements=input->tensor_total_elements();
        size_t  hidden_dim=weight->tensor_total_elements();
        if(total_elements==0||hidden_dim==0){
            return;
        }
        size_t num_tokens=total_elements/hidden_dim;

        int threads_per_block = 256;
        int blocks_per_grid = num_tokens;
        size_t shared_mem_bytes=threads_per_block*sizeof(float);

        if(input->tensor_data_type()==DataType::kDataTypeFP16){
            const __half *d_in=input->tensor_data_ptr<__half>();
            const float *d_weight=weight->tensor_data_ptr<float>();
            __half *d_out=output->tensor_data_ptr<__half>();
            rmsnorm_kernel_cuda_fp16<<<blocks_per_grid,threads_per_block,shared_mem_bytes>>>(
                d_out, d_in, d_weight, hidden_dim, eps);
        } else {
            const float *d_in=input->tensor_data_ptr<float>();
            const float *d_weight=weight->tensor_data_ptr<float>();
            float *d_out=output->tensor_data_ptr<float>();
            rmsnorm_kernel_cuda<<<blocks_per_grid,threads_per_block,shared_mem_bytes>>>(
                d_out, d_in, d_weight, hidden_dim, eps);
        }
    }

    // ===== Fused Add + RMSNorm (SGLang-style) =====
    __global__ void fused_add_rmsnorm_kernel_fp16(
            __half* hidden_out, __half* residual_out,
            const __half* hidden_states, const __half* residual,
            const float* weight, int hidden_dim, float eps) {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        const __half* curr_hidden = hidden_states + bid * hidden_dim;
        const __half* curr_residual = residual + bid * hidden_dim;
        __half* curr_h_out = hidden_out + bid * hidden_dim;
        __half* curr_r_out = residual_out + bid * hidden_dim;
        float local_sum = 0.0f;
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            float val = __half2float(curr_hidden[i]) + __half2float(curr_residual[i]);
            curr_r_out[i] = __float2half(val);
            local_sum += val * val;
        }
        extern __shared__ float s_sum[];
        s_sum[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_sum[tid] += s_sum[tid + stride];
            __syncthreads();
        }
        __shared__ float s_inv_rms;
        if (tid == 0) s_inv_rms = rsqrtf(s_sum[0] / hidden_dim + eps);
        __syncthreads();
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            float val = __half2float(curr_r_out[i]);
            curr_h_out[i] = __float2half(val * s_inv_rms * weight[i]);
        }
    }

    __global__ void fused_add_rmsnorm_kernel_fp32(
            float* hidden_out, float* residual_out,
            const float* hidden_states, const float* residual,
            const float* weight, int hidden_dim, float eps) {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        const float* curr_hidden = hidden_states + bid * hidden_dim;
        const float* curr_residual = residual + bid * hidden_dim;
        float* curr_h_out = hidden_out + bid * hidden_dim;
        float* curr_r_out = residual_out + bid * hidden_dim;
        float local_sum = 0.0f;
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            float val = curr_hidden[i] + curr_residual[i];
            curr_r_out[i] = val;
            local_sum += val * val;
        }
        extern __shared__ float s_sum[];
        s_sum[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_sum[tid] += s_sum[tid + stride];
            __syncthreads();
        }
        __shared__ float s_inv_rms;
        if (tid == 0) s_inv_rms = rsqrtf(s_sum[0] / hidden_dim + eps);
        __syncthreads();
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            curr_h_out[i] = curr_r_out[i] * s_inv_rms * weight[i];
        }
    }

    void fused_add_rmsnorm_cuda(
            std::shared_ptr<Tensor>& hidden_states,
            std::shared_ptr<Tensor>& residual,
            std::shared_ptr<Tensor>& hidden_out,
            std::shared_ptr<Tensor>& residual_out,
            const std::shared_ptr<Tensor>& weight,
            float eps) {
        size_t total_elements = hidden_states->tensor_total_elements();
        size_t hidden_dim = weight->tensor_total_elements();
        if (total_elements == 0 || hidden_dim == 0) return;
        size_t num_tokens = total_elements / hidden_dim;
        int threads_per_block = 256;
        int blocks_per_grid = num_tokens;
        size_t shared_mem_bytes = threads_per_block * sizeof(float);
        if (hidden_states->tensor_data_type() == DataType::kDataTypeFP16) {
            fused_add_rmsnorm_kernel_fp16<<<blocks_per_grid, threads_per_block, shared_mem_bytes>>>(
                hidden_out->tensor_data_ptr<__half>(),
                residual_out->tensor_data_ptr<__half>(),
                hidden_states->tensor_data_ptr<__half>(),
                residual->tensor_data_ptr<__half>(),
                weight->tensor_data_ptr<float>(), hidden_dim, eps);
        } else {
            fused_add_rmsnorm_kernel_fp32<<<blocks_per_grid, threads_per_block, shared_mem_bytes>>>(
                hidden_out->tensor_data_ptr<float>(),
                residual_out->tensor_data_ptr<float>(),
                hidden_states->tensor_data_ptr<float>(),
                residual->tensor_data_ptr<float>(),
                weight->tensor_data_ptr<float>(), hidden_dim, eps);
        }
    }
}
