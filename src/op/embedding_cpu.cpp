#include "tensor/tensor.h"
#include "base/dispatch.h"
namespace hxinfer{
    void embedding_tensor(const std::shared_ptr<Tensor>& token_ids,const std::shared_ptr<Tensor>& weight,
                          std::shared_ptr<Tensor>& output){
        DataType type_out=output->tensor_data_type();
        if (weight->tensor_data_type() != type_out) {
            throw std::runtime_error("embedding_tensor: weight 和 output 的数据类型必须一致!");
        }
        auto embedding_logic=[&](const auto* ptr_ids,const auto* ptr_weight,auto* ptr_out){
            using weight_type=std::decay_t<decltype(*ptr_weight)>;
            std::vector<int> weight_shapes=weight->tensor_shapes();
            int vocab_size=weight_shapes[0];
            int dim=weight_shapes[1];
            size_t total_elements=token_ids->tensor_total_elements();
            for(size_t i=0;i<total_elements;i++){
                int index=ptr_ids[i];
                if (index < 0 || index >= vocab_size){
                    throw std::runtime_error("embedding_tensor 词表越界!\n");
                }
                memcpy(ptr_out+i*dim,ptr_weight+index*dim,size_t(dim)*sizeof (weight_type));
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_out,"embedding",[&](){
            embedding_logic(token_ids->tensor_data_ptr<int32_t>(),
                    weight->tensor_data_ptr<scalar_t>(),
                    output->tensor_data_ptr<scalar_t>());
        });
    }
}