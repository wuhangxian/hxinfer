#include "op/math_ops.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"
namespace hxinfer{
    __global__ void silu_kernel_cuda(const float* in_data, float* out_data, size_t totalElements){
        size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<totalElements){
            float val=in_data[idx];
            out_data[idx]=val/(1+expf(-val));
        }
    }
    __global__ void silu_kernel_cuda_fp16(const __half* in_data, __half* out_data, size_t totalElements){
        size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<totalElements){
            float val=__half2float(in_data[idx]);
            out_data[idx]=__float2half(val/(1.0f+expf(-val)));
        }
    }
    void silu_cuda(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output){
        if(input->tensor_device_type()!=DeviceType::kDeviceCUDA||
            output->tensor_device_type()!=DeviceType::kDeviceCUDA){
            std::cerr<<"[Fatal Error]silu_cuda expects CUDA Tensors!"<<std::endl;
            return;
        }
        size_t total_elements=input->tensor_total_elements();
        if(total_elements==0){
            return;
        }
        int threads_per_block=256;
        int blocks_per_grid=(total_elements+threads_per_block-1)/threads_per_block;

        if(input->tensor_data_type()==DataType::kDataTypeFP16){
            const __half *d_in=input->tensor_data_ptr<__half>();
            __half *d_out=output->tensor_data_ptr<__half>();
            silu_kernel_cuda_fp16<<<blocks_per_grid,threads_per_block>>>(d_in,d_out,total_elements);
        } else {
            const float *d_in=input->tensor_data_ptr<float>();
            float *d_out=output->tensor_data_ptr<float>();
            silu_kernel_cuda<<<blocks_per_grid,threads_per_block>>>(d_in,d_out,total_elements);
        }
    }
}
