#include "op/math_ops.h"
#include "cuda_runtime.h"
#include "iostream"
namespace hxinfer{
    //还可以优化的方向--使用float4,一个线程运输4个float捆绑到一起,128bits
    __global__ void silu_kernel_cuda(const float* in_data, float* out_data, size_t totalElements){
        size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
        if(idx<totalElements){
            float val=in_data[idx];//访问寄存器,减少时间
            out_data[idx]=val/(1+expf(-val));
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
        const float *d_in=input->tensor_data_ptr<float>();
        float *d_out=output->tensor_data_ptr<float>();

        int threads_per_block=256;
        int blocks_per_grid=(total_elements+threads_per_block-1)/threads_per_block;
        silu_kernel_cuda<<<blocks_per_grid,threads_per_block>>>(d_in,d_out,total_elements);
    }
}