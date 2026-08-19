#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    void embedding_tensor(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                          std::shared_ptr<Tensor>& output){
        if(weight->tensor_device_type()==DeviceType::kDeviceCUDA){
            embedding_cuda(token_ids,weight,output);
        }else if(weight->tensor_device_type()==DeviceType::kDeviceCPU){
            embedding_cpu(token_ids,weight,output);
        }else{
            std::cerr<<"Unknown device type for embedding_tensor!"<<std::endl;
        }
    }
}
