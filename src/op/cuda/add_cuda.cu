#include "cuda_runtime.h"
#include "op/math_ops.h"
#include "cuda_fp16.h"
#include "iostream"
namespace hxinfer{
    __global__ void add_kernel_cuda(const float *in_a,const float *in_b,float *out,size_t total_elements){
        size_t index=blockIdx.x*blockDim.x+threadIdx.x;
        if(index<total_elements){
            out[index]=in_a[index]+in_b[index];
        }
    }
    __global__ void add_kernel_cuda_fp16(const __half *in_a,const __half *in_b,__half *out,size_t total_elements){
        size_t index=blockIdx.x*blockDim.x+threadIdx.x;
        if(index<total_elements){
            out[index]=__float2half(__half2float(in_a[index])+__half2float(in_b[index]));
        }
    }

    void add_cuda(const std::shared_ptr<Tensor> &input_a,const std::shared_ptr<Tensor> &input_b,
                  std::shared_ptr<Tensor>& output){
        if(input_a->tensor_device_type()!=DeviceType::kDeviceCUDA||
           input_b->tensor_device_type()!=DeviceType::kDeviceCUDA||
           output->tensor_device_type()!=DeviceType::kDeviceCUDA){
            std::cerr<<"[Fatal Error] add_cuda expects CUDA Tensors\n"<<std::endl;
            return;
        }
        if(input_a->tensor_total_elements()!=input_b->tensor_total_elements()||
           input_a->tensor_total_elements()!=output->tensor_total_elements()){
            std::cerr<<"[Fatal Error] add_cuda expects same total_elements\n"<<std::endl;
            return;
        }
        size_t total_elements=input_a->tensor_total_elements();
        if(total_elements==0){
            return;
        }
        int threads_per_Block = 256;
        int blocks_per_Grid= (total_elements+threads_per_Block-1)/threads_per_Block;

        if(input_a->tensor_data_type()==DataType::kDataTypeFP16){
            const __half *in_a_ptr=input_a->tensor_data_ptr<__half>();
            const __half *in_b_ptr=input_b->tensor_data_ptr<__half>();
            __half *out_ptr=output->tensor_data_ptr<__half>();
            add_kernel_cuda_fp16<<<blocks_per_Grid,threads_per_Block>>>(in_a_ptr,in_b_ptr,out_ptr,total_elements);
        } else {
            const float *in_a_ptr=input_a->tensor_data_ptr<float>();
            const float *in_b_ptr=input_b->tensor_data_ptr<float>();
            float *out_ptr=output->tensor_data_ptr<float>();
            add_kernel_cuda<<<blocks_per_Grid,threads_per_Block>>>(in_a_ptr,in_b_ptr,out_ptr,total_elements);
        }
    }
}
