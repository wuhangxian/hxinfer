#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "iostream"
namespace hxinfer{
    __global__ void embedding_kernel_cuda(const int* token_ids,const float* weight,
                                          float* output,int dim,size_t num_tokens){
        size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
        size_t total=num_tokens*dim;
        if(idx<total){
            size_t token_idx=idx/dim;
            int dim_idx=idx%dim;
            int token_id=token_ids[token_idx];
            output[idx]=weight[token_id*dim+dim_idx];
        }
    }

    void embedding_cuda(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output){
        if(weight->tensor_device_type()!=DeviceType::kDeviceCUDA||
            output->tensor_device_type()!=DeviceType::kDeviceCUDA){
            std::cerr<<"[Fatal Error] embedding_cuda expects CUDA Tensors!"<<std::endl;
            return;
        }
        std::vector<int> weight_shapes=weight->tensor_shapes();
        int dim=weight_shapes[1];
        size_t num_tokens=token_ids->tensor_total_elements();

        const int* d_ids=token_ids->tensor_data_ptr<int>();
        const float* d_weight=weight->tensor_data_ptr<float>();
        float* d_out=output->tensor_data_ptr<float>();

        size_t total=num_tokens*dim;
        int threads=256;
        int blocks=(total+threads-1)/threads;
        embedding_kernel_cuda<<<blocks,threads>>>(d_ids,d_weight,d_out,dim,num_tokens);
        cudaDeviceSynchronize();
    }
}
