#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    void rmsnorm_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output,float eps){
        if(input->tensor_device_type()==DeviceType::kDeviceCPU){
            rmsnorm_cpu(input,weight,output,eps);
        }else if(input->tensor_device_type()==DeviceType::kDeviceCUDA){
            rmsnorm_cuda(input,weight,output,eps);
        }else{
            std::cerr<<"Unknown device type for mul_tensor!"<<std::endl;
        }
    }
}
