
#include "cuda_runtime.h"
#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    __global__ void add_kernel_cuda(const float *in_a,const float *in_b,float *out,size_t total_elements){
        size_t index=blockIdx.x*blockDim.x+threadIdx.x;
        if(index<total_elements){
            out[index]=in_a[index]+in_b[index];
        }
    }

    void add_cuda(const std::shared_ptr<Tensor> &input_a,const std::shared_ptr<Tensor> &input_b,
                  std::shared_ptr<Tensor>& output){
        if(input_a->tensor_device_type()!=DeviceType::kDeviceCUDA||
                    input_b->tensor_device_type()!=DeviceType::kDeviceCUDA||
                    output->tensor_device_type()!=DeviceType::kDeviceCUDA){
            std::cerr<<"[Fatal error] add_cuda expects CUDA Tensors!\n";
            return;
        }
        if(input_a->tensor_total_elements()!=input_b->tensor_total_elements()||
                input_a->tensor_total_elements()!=output->tensor_total_elements()){
            throw std::runtime_error("[Fatal error] add_cuda的input_a与input_b与output数量大小不匹配!\n");
        }
        size_t total_elements=input_a->tensor_total_elements();
        if(total_elements==0){
            return;
        }
        int threads_per_Block = 256;
        int blocks_per_Grid= (total_elements+threads_per_Block-1)/threads_per_Block;
        const float *in_a_ptr=input_a->tensor_data_ptr<float>();
        const float *in_b_ptr=input_b->tensor_data_ptr<float>();
        float *out_ptr=output->tensor_data_ptr<float>();
        add_kernel_cuda<<<blocks_per_Grid,threads_per_Block>>>(in_a_ptr,in_b_ptr,out_ptr,total_elements);
        cudaDeviceSynchronize();
    }
}