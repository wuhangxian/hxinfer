#include "op/math_ops.h"
#include "base/dispatch.h"
namespace hxinfer{
    void add_tensor(const std::shared_ptr<Tensor>& input_a,const std::shared_ptr<Tensor>& input_b,
                    std::shared_ptr<Tensor>& output){
        size_t input_a_tensor_total_elements=input_a->tensor_total_elements();
        size_t input_b_tensor_total_elements=input_b->tensor_total_elements();
        size_t output_tensor_total_elements=output->tensor_total_elements();
        if(input_a_tensor_total_elements!=input_b_tensor_total_elements){
            throw std::runtime_error("add_tensor的input_a与input_b数量大小不匹配!\n");
        }
        if(input_a_tensor_total_elements!=output_tensor_total_elements){
            throw std::runtime_error("add_tensor的input_a与output数量大小不匹配!\n");
        }
        if(input_b_tensor_total_elements!=output_tensor_total_elements){
            throw std::runtime_error("add_tensor的input_b与output数量大小不匹配!\n");
        }
        DataType type_a=input_a->tensor_data_type();
        DataType type_b=input_b->tensor_data_type();
        DataType type_out=output->tensor_data_type();
        if(type_a!=type_b){
            throw std::runtime_error("add_tensor的input_a与input_b数据类型不匹配!\n");
        }
        if(type_a!=type_out){
            throw std::runtime_error("add_tensor的input_a与output数据类型不匹配!\n");
        }
        if(type_b!=type_out){
            throw std::runtime_error("add_tensor的input_b与output数据类型不匹配!\n");
        }
        auto add_logic=[](const auto* ptr_a,const auto* ptr_b,auto* ptr_out,size_t total_elements){
            for(size_t i=0;i<total_elements;i++){
                ptr_out[i]=ptr_a[i]+ptr_b[i];
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_a,"add_tensor",[&](){
            add_logic(input_a->tensor_data_ptr<scalar_t>(),
                    input_b->tensor_data_ptr<scalar_t>(),
                    output->tensor_data_ptr<scalar_t>(),
                    output->tensor_total_elements());
        });
    }
    void matmul_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                       std::shared_ptr<Tensor>& output);
    void rope_tensor(std::shared_ptr<Tensor>& q,std::shared_ptr<Tensor>& k,int step);
    void rmsnorm_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor> weight,
                        std::shared_ptr<Tensor>& output);
    void embedding_tensor(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                          std::shared_ptr<Tensor>& output);
    void silu_tensor(const std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output);
    int  argmax_tensor(const std::shared_ptr<Tensor>& input);
}