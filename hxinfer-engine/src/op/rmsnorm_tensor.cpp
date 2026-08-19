#include "op/math_ops.h"
#include "iostream"
namespace hxinfer{
    void rmsnorm_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output,float eps){
        if(input->tensor_device_type()==DeviceType::kDeviceCPU){
            rmsnorm_cpu(input,weight,output);
        }else if(input->tensor_device_type()==DeviceType::kDeviceCUDA){
            rmsnorm_cuda(input,weight,output);
        }else{
            std::cerr<<"Unknown device type for rmsnorm_tensor!"<<std::endl;
        }
    }

    void fused_add_rmsnorm_tensor(
            std::shared_ptr<Tensor>& hidden_states,
            std::shared_ptr<Tensor>& residual,
            std::shared_ptr<Tensor>& hidden_out,
            std::shared_ptr<Tensor>& residual_out,
            const std::shared_ptr<Tensor>& weight,
            float eps){
        if(hidden_states->tensor_device_type()==DeviceType::kDeviceCUDA){
            fused_add_rmsnorm_cuda(hidden_states,residual,hidden_out,residual_out,weight,eps);
        }else{
            // CPU fallback: do add + rmsnorm separately
            add_tensor(hidden_states,residual,residual_out);
            rmsnorm_tensor(residual_out,weight,hidden_out,eps);
        }
    }
}
