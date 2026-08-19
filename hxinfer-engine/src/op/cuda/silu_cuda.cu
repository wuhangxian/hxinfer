#include "op/math_ops.h"
#include "op/templated_kernels.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"
namespace hxinfer{
    void silu_cuda(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output){
        if(input->tensor_device_type()!=DeviceType::kDeviceCUDA||
            output->tensor_device_type()!=DeviceType::kDeviceCUDA){
            std::cerr<<"[Fatal Error] silu_cuda expects CUDA Tensors!"<<std::endl;
            return;
        }
        size_t total_elements=input->tensor_total_elements();
        if(total_elements==0) return;
        int threads=256;
        int blocks=(total_elements+threads-1)/threads;
        if(input->tensor_data_type()==DataType::kDataTypeFP16){
            silu_kernel<__half><<<blocks,threads>>>(input->tensor_data_ptr<__half>(),output->tensor_data_ptr<__half>(),total_elements);
        } else {
            silu_kernel<float><<<blocks,threads>>>(input->tensor_data_ptr<float>(),output->tensor_data_ptr<float>(),total_elements);
        }
    }
}
