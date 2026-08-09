#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"
namespace hxinfer{
    __global__ void mul_kernel_cuda(const float *in_a,const float *in_b,float *out,size_t total_elements){
        int idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<total_elements){
            out[idx]=in_a[idx]*in_b[idx];
        }
    }
    __global__ void mul_kernel_cuda_fp16(const __half *in_a,const __half *in_b,__half *out,size_t total_elements){
        int idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<total_elements){
            out[idx]=__float2half(__half2float(in_a[idx])*__half2float(in_b[idx]));
        }
    }

    void mul_cuda(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                  std::shared_ptr<Tensor>& output){
        if(input_a->tensor_device_type()!=DeviceType::kDeviceCUDA||
                input_b->tensor_device_type()!=DeviceType::kDeviceCUDA||
                output->tensor_device_type()!=DeviceType::kDeviceCUDA){
            std::cerr<<"[Fatal Error] mul_cuda expects CUDA Tensors\n"<<std::endl;
            return;
        }
        if(input_a->tensor_total_elements()!=input_b->tensor_total_elements()||
            input_a->tensor_total_elements()!=output->tensor_total_elements()){
            std::cerr<<"[Fatal Error] mul_cuda expects same total_elements\n"<<std::endl;
            return;
        }
        size_t total_elements=input_a->tensor_total_elements();
        int threads_per_block=256;
        int blocks_per_grid=(total_elements+threads_per_block-1)/threads_per_block;

        if(input_a->tensor_data_type()==DataType::kDataTypeFP16){
            const __half *in_a=input_a->tensor_data_ptr<__half>();
            const __half *in_b=input_b->tensor_data_ptr<__half>();
            __half *out=output->tensor_data_ptr<__half>();
            mul_kernel_cuda_fp16<<<blocks_per_grid,threads_per_block>>>(in_a,in_b,out,total_elements);
        } else {
            const float *in_a=input_a->tensor_data_ptr<float>();
            const float *in_b=input_b->tensor_data_ptr<float>();
            float *out=output->tensor_data_ptr<float>();
            mul_kernel_cuda<<<blocks_per_grid,threads_per_block>>>(in_a,in_b,out,total_elements);
        }
    }
}
