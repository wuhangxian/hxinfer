#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    void softmax_tensor(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output){
        if(input->tensor_device_type()==DeviceType::kDeviceCUDA){
            softmax_cuda(input,output);
        }else if(input->tensor_device_type()==DeviceType::kDeviceCPU){
            softmax_cpu(input,output);
        }else{
            std::cerr<<"Unknown device type for softmax_tensor!"<<std::endl;
        }
    }
}
