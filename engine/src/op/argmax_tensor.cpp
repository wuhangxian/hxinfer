#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    int argmax_tensor(const std::shared_ptr<Tensor>& input){
        if(input->tensor_device_type()==DeviceType::kDeviceCUDA){
            return argmax_cuda(input);
        }else if(input->tensor_device_type()==DeviceType::kDeviceCPU){
            return argmax_cpu(input);
        }else{
            std::cerr<<"Unknown device type for argmax_tensor!"<<std::endl;
            return -1;
        }
    }
}
