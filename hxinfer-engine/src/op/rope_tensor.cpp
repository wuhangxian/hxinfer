#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    void rope_tensor(std::shared_ptr<Tensor>& q,std::shared_ptr<Tensor>& k,ModelConfig& config,int step,float base){
        if(q->tensor_device_type()==DeviceType::kDeviceCUDA){
            rope_cuda(q,k,config,step,base);
        }else if(q->tensor_device_type()==DeviceType::kDeviceCPU){
            rope_cpu(q,k,config,step,base);
        }else{
            std::cerr<<"Unknown device type for rope_tensor!"<<std::endl;
        }
    }
}
