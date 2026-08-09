#include "tensor/tensor.h"
#include "base/dispatch.h"
#include "cmath"
#include "iostream"
namespace hxinfer{
    void silu_cpu(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output){
        DataType type_out=output->tensor_data_type();
        if(input->tensor_device_type()!=DeviceType::kDeviceCPU||
           output->tensor_device_type()!=DeviceType::kDeviceCPU){
            std::cerr<<"[Fatal Error]silu_cpu expects CPU Tensors!"<<std::endl;
            return;
        }
        auto silu_logic=[&](const auto*ptr_in,auto *ptr_out){
            using OutType=std::decay_t<decltype(*ptr_out)>;
            size_t total_elements=input->tensor_total_elements();
            for(size_t i=0;i<total_elements;i++){
                float val=static_cast<float>(ptr_in[i]);
                float result_val=val/(1+std::exp(-val));
                ptr_out[i]=static_cast<OutType>(result_val);
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_out,"silu",[&](){
            silu_logic(input->tensor_data_ptr<scalar_t>(),
                    output->tensor_data_ptr<scalar_t>());
        });
    }
}