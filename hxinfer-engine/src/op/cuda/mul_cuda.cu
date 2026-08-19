#include "op/templated_kernels.h"
#include "op/math_ops.h"
#include "cuda_fp16.h"
#include <iostream>
namespace hxinfer{
    void mul_cuda(const std::shared_ptr<Tensor>& a,const std::shared_ptr<Tensor>& b,std::shared_ptr<Tensor>& out){
        if(a->tensor_device_type()!=DeviceType::kDeviceCUDA||b->tensor_device_type()!=DeviceType::kDeviceCUDA||out->tensor_device_type()!=DeviceType::kDeviceCUDA){std::cerr<<"[Error] mul_cuda expects CUDA\n";return;}
        if(a->tensor_total_elements()!=b->tensor_total_elements()||a->tensor_total_elements()!=out->tensor_total_elements()){std::cerr<<"[Error] mul_cuda size mismatch\n";return;}
        size_t n=a->tensor_total_elements();if(n==0)return;
        int t=256;int bl=(n+t-1)/t;
        if(a->tensor_data_type()==DataType::kDataTypeFP16){mul_kernel<__half><<<bl,t>>>(a->tensor_data_ptr<__half>(),b->tensor_data_ptr<__half>(),out->tensor_data_ptr<__half>(),n);}
        else{mul_kernel<float><<<bl,t>>>(a->tensor_data_ptr<float>(),b->tensor_data_ptr<float>(),out->tensor_data_ptr<float>(),n);}
    }
}
