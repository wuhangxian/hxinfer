#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    void silu_tensor(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output){
        if(input->tensor_device_type()==DeviceType::kDeviceCUDA){
            silu_cuda(input,output);
        }else if (input->tensor_device_type()==DeviceType::kDeviceCPU){
            silu_cpu(input,output);
        }else{
            std::cerr<<"Unknown device type for silu_tensor!"<<std::endl;
        }
    }
}