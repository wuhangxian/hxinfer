
#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    void mul_tensor(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                  std::shared_ptr<Tensor>& output){
        if(input_a->tensor_device_type()==DeviceType::kDeviceCPU){
            mul_cpu(input_a,input_b,output);
        }else if(input_a->tensor_device_type()==DeviceType::kDeviceCUDA){
            mul_cuda(input_a,input_b,output);
        }else{
            std::cerr<<"Unknown device type for mul_tensor!"<<std::endl;
        }
    }
}