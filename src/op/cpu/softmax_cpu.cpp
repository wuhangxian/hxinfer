#include "tensor/tensor.h"
#include "op/math_ops.h"
#include "base/dispatch.h"
#include "cmath"
namespace hxinfer{
    void softmax_tensor(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output){
        size_t input_tensor_total_elements=input->tensor_total_elements();
        size_t output_tensor_total_elements=output->tensor_total_elements();
        if(input_tensor_total_elements!=output_tensor_total_elements){
            throw std::runtime_error("softmax_tensor的input和output数量大小不匹配!\n");
        }
        DataType type_input=input->tensor_data_type();
        DataType type_output=output->tensor_data_type();
        if(type_input!=type_output){
            throw std::runtime_error("softmax_tensor的input和output数据类型不匹配!\n");
        }

        auto softmax_logic=[](const auto* in_ptr,auto *out_ptr,size_t total_elements){
            using OutType=std::decay_t<decltype(*out_ptr)>;
            float max_val=std::numeric_limits<float>::lowest();
            for(size_t i=0;i<total_elements;i++){
                float real_val=static_cast<float >(in_ptr[i]);
                if(real_val>max_val){
                    max_val=real_val;
                }
            }
            float sum=0;
            for(size_t i=0;i<total_elements;i++){
                float real_val=static_cast<float >(in_ptr[i]);
                float exp_val=std::exp(real_val-max_val);
                sum=sum+exp_val;
                out_ptr[i]=static_cast<OutType>(exp_val);
            }
            for(size_t i=0;i<total_elements;i++){
                float current_val=static_cast<float >(out_ptr[i]);
                float final_prob=current_val/sum;
                out_ptr[i]=static_cast<OutType>(final_prob);
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_output,"softmax",[&](){
            softmax_logic(input->tensor_data_ptr<scalar_t >(),
                    output->tensor_data_ptr<scalar_t>(),
                    output->tensor_total_elements());
        });

    }
}
