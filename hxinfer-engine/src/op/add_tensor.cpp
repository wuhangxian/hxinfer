#include "iostream"
#include "op/math_ops.h"
namespace hxinfer{
    void add_tensor(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output){
        if(input_a->tensor_device_type()==DeviceType::kDeviceCPU){
            add_cpu(input_a,input_b,output);
        }else if(input_a->tensor_device_type()==DeviceType::kDeviceCUDA){
            add_cuda(input_a,input_b,output);
        }else{
            std::cerr<<"Unknown device type for add_tensor!"<<std::endl;
        }
    }
};
