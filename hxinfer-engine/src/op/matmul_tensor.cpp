#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    void matmul_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                       std::shared_ptr<Tensor>& output){
        if(input->tensor_device_type()==DeviceType::kDeviceCPU){
            matmul_cpu(input,weight,output);
        }else if(input->tensor_device_type()==DeviceType::kDeviceCUDA){
            matmul_cuda(input,weight,output);
        }else{
            std::cerr<<"Unknown device type for matmul_tensor!"<<std::endl;
        }
    }
}