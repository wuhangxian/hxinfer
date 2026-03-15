#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    void silu_tensor(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output){
        if(input->deviceType()==DeviceType::kDeviceCUDA){
            silu_cuda(input,output);
        }else if (input->deviceType()==DeviceType::kDeviceCPU){
            silu_cpu(input,output);
        }else{
            std::cerr<<"Unknown device type for silu_tensor!"<<std::endl;
        }
    }
}