#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"

namespace hxinfer{
    __global__ void embedding_kernel_fp32(const int* token_ids, const float* weight,
                                          float* output, int dim, size_t num_tokens){
        size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<num_tokens*dim){
            int token_id=token_ids[idx/dim];
            output[idx]=weight[token_id*dim + idx%dim];
        }
    }

    __global__ void embedding_kernel_fp16(const int* token_ids, const __half* weight,
                                          float* output, int dim, size_t num_tokens){
        size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<num_tokens*dim){
            int token_id=token_ids[idx/dim];
            output[idx]=__half2float(weight[token_id*dim + idx%dim]);
        }
    }

    // FP16 权重 → FP16 输出（7B 激活值 FP16 路径）
    __global__ void embedding_kernel_fp16_to_fp16(const int* token_ids, const __half* weight,
                                                   __half* output, int dim, size_t num_tokens){
        size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<num_tokens*dim){
            int token_id=token_ids[idx/dim];
            output[idx]=weight[token_id*dim + idx%dim];
        }
    }

    void embedding_cuda(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output){
        int dim=weight->tensor_shapes()[1];
        size_t num_tokens=token_ids->tensor_total_elements();
        size_t total=num_tokens*dim;
        int threads=256;
        int blocks=(total+threads-1)/threads;

        const int* d_ids=token_ids->tensor_data_ptr<int>();

        if(weight->tensor_data_type()==DataType::kDataTypeFP16){
            const __half* d_w=weight->tensor_data_ptr<__half>();
            if(output->tensor_data_type()==DataType::kDataTypeFP16){
                // FP16 权重 → FP16 输出（全程 FP16）
                __half* d_out=output->tensor_data_ptr<__half>();
                embedding_kernel_fp16_to_fp16<<<blocks,threads>>>(d_ids,d_w,d_out,dim,num_tokens);
            } else {
                // FP16 权重 → FP32 输出（旧路径，15M 兼容）
                float* d_out=output->tensor_data_ptr<float>();
                embedding_kernel_fp16<<<blocks,threads>>>(d_ids,d_w,d_out,dim,num_tokens);
            }
        } else {
            const float* d_w=weight->tensor_data_ptr<float>();
            float* d_out=output->tensor_data_ptr<float>();
            embedding_kernel_fp32<<<blocks,threads>>>(d_ids,d_w,d_out,dim,num_tokens);
        }
    }
}
