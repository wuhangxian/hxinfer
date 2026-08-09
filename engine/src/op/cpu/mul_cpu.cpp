#include "op/math_ops.h"
#include "tensor/tensor.h"
#include "base/dispatch.h"
namespace hxinfer{
    void mul_cpu(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output){
        size_t input_a_tensor_total_elements=input_a->tensor_total_elements();
        size_t input_b_tensor_total_elements=input_b->tensor_total_elements();
        size_t output_tensor_total_elements=output->tensor_total_elements();
        if(input_a_tensor_total_elements!=input_b_tensor_total_elements){
            throw std::runtime_error("mul_tensor的input_a与input_b数量大小不匹配!\n");
        }
        if(input_a_tensor_total_elements!=output_tensor_total_elements){
            throw std::runtime_error("mul_tensor的input_a与output数量大小不匹配!\n");
        }
        if(input_b_tensor_total_elements!=output_tensor_total_elements){
            throw std::runtime_error("mul_tensor的input_b与output数量大小不匹配!\n");
        }
        DataType type_a=input_a->tensor_data_type();
        DataType type_b=input_b->tensor_data_type();
        DataType type_out=output->tensor_data_type();
        if(type_a!=type_b){
            throw std::runtime_error("mul_tensor的input_a与input_b数据类型不匹配!\n");
        }
        if(type_a!=type_out){
            throw std::runtime_error("mul_tensor的input_a与output数据类型不匹配!\n");
        }
        if(type_b!=type_out){
            throw std::runtime_error("mul_tensor的input_b与output数据类型不匹配!\n");
        }
        auto mul_logic=[](const auto *ptr_a,
                const auto *ptr_b,auto *ptr_out,size_t total_elements){
            using OutType=std::decay_t<decltype(*ptr_out)>;
            for(size_t i=0;i<total_elements;i++){
                float val_a=ptr_a[i];
                float val_b=ptr_b[i];
                float val_out=val_a*val_b;
                ptr_out[i]=static_cast<OutType>(val_out);
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_a,"mul_tensor",[&](){
            mul_logic(input_a->tensor_data_ptr<scalar_t>(),
                    input_b->tensor_data_ptr<scalar_t>(),
                    output->tensor_data_ptr<scalar_t >(),
                    output->tensor_total_elements());
        });

    }
}

