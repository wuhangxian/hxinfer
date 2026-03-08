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
            using T=std::decay_t<decltype(*in_ptr)>;
            T max_val=std::numeric_limits<T>::lowest();
            for(size_t i=1;i<total_elements;i++){
                if(in_ptr[i]>max_val){
                    max_val=in_ptr[i];
                }
            }
            T sum=0;
            for(size_t i=0;i<total_elements;i++){
                out_ptr[i]=std::exp(in_ptr[i]-max_val);
                sum=sum+out_ptr[i];
            }
            for(size_t i=0;i<total_elements;i++){
                out_ptr[i]=out_ptr[i]/sum;
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_output,"softmax",[&]{
            softmax_logic(input->tensor_data_ptr<scalar_t >(),
                    output->tensor_data_ptr<scalar_t>(),
                    output->tensor_total_elements());
        });

    }
}
