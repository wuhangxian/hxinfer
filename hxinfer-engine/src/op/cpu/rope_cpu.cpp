#include "tensor/tensor.h"
#include "base/dispatch.h"
#include "cmath"
namespace hxinfer{
    void rope_cpu(std::shared_ptr<Tensor>& q,std::shared_ptr<Tensor>& k,ModelConfig& config,int step,float base){
        // 🚀 1. 极其严密的安检门
        DataType type_q = q->tensor_data_type();
        if (type_q != k->tensor_data_type()) {
            throw std::runtime_error("rope_tensor: Q 和 K 的数据类型必须完全一致!\n");
        }

        auto rope_logic=[&](auto* ptr,std::shared_ptr<Tensor>& tensor_obj,int num_heads){
            using OutType=std::decay_t<decltype(*ptr)>;
            int dim=tensor_obj->tensor_shapes().back();
            int head_dim=config.dim/config.head;
            size_t row=tensor_obj->tensor_total_elements()/dim;

            for(size_t i=0;i<row;i++){
                for(int j=0;j<num_heads;j++){
                    int half_head_dim=head_dim/2;
                    for(int pair_idx=0;pair_idx<half_head_dim;pair_idx++){
                        auto *head_ptr=ptr+i*dim+j*head_dim;
                        int d=pair_idx*2;
                        float scale = 1.0f / std::pow(base, static_cast<float>(d) / head_dim);
                        float angle = step * scale;
                        float angle_cos = std::cos(angle);
                        float angle_sin = std::sin(angle);
                        float v0 = static_cast<float>(head_ptr[pair_idx]);
                        float v1 = static_cast<float>(head_ptr[pair_idx+half_head_dim]);
                        float out0 = v0 * angle_cos - v1 * angle_sin;
                        float out1 = v1 * angle_cos + v0 * angle_sin;
                        head_ptr[pair_idx] = static_cast<OutType>(out0);
                        head_ptr[pair_idx+half_head_dim] = static_cast<OutType>(out1);
                    }
                }
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_q,"rope",[&](){
            rope_logic(q->tensor_data_ptr<scalar_t>(),q,config.head);
            rope_logic(k->tensor_data_ptr<scalar_t>(),k,config.kv_head);
        });
    }
}
