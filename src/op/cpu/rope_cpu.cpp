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
                    for(int d=0;d<head_dim;d=d+2){
                        auto *curr_ptr=ptr+i*dim+j*head_dim+d;
                        float scale = 1.0f / std::pow(base, static_cast<float>(d) / head_dim);
                        float angle = step * scale;
                        float angle_cos = std::cos(angle);
                        float angle_sin = std::sin(angle);
                        float v0 = static_cast<float>(curr_ptr[0]);
                        float v1 = static_cast<float>(curr_ptr[1]);
                        float out0 = v0 * angle_cos - v1 * angle_sin;
                        float out1 = v0 * angle_sin + v1 * angle_cos;
                        curr_ptr[0] = static_cast<OutType>(out0);
                        curr_ptr[1] = static_cast<OutType>(out1);
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