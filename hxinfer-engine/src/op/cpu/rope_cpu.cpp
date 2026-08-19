#include "tensor/tensor.h"
#include "base/dispatch.h"
#include "cmath"
namespace hxinfer{
    void rope_cpu(std::shared_ptr<Tensor>& q,std::shared_ptr<Tensor>& k,ModelConfig& config,int step,float base){
        DataType type_q = q->tensor_data_type();
        if (type_q != k->tensor_data_type()) {
            throw std::runtime_error("rope_tensor: Q and K must have the same dtype!\n");
        }

        // NeoX-style RoPE: split head_dim into first/second half
        auto rope_logic=[&](auto* ptr,std::shared_ptr<Tensor>& tensor_obj,int num_heads){
            using OutType=std::decay_t<decltype(*ptr)>;
            int dim=tensor_obj->tensor_shapes().back();
            int head_dim=config.dim/config.head;
            int half_dim=head_dim/2;
            size_t row=tensor_obj->tensor_total_elements()/dim;

            for(size_t i=0;i<row;i++){
                for(int j=0;j<num_heads;j++){
                    auto* base_ptr=ptr+i*dim+j*head_dim;
                    for(int d=0;d<half_dim;d++){
                        float scale=1.0f/std::pow(base, static_cast<float>(2*d)/head_dim);
                        float angle=step*scale;
                        float cos_val=std::cos(angle);
                        float sin_val=std::sin(angle);
                        float v1=static_cast<float>(base_ptr[d]);
                        float v2=static_cast<float>(base_ptr[d+half_dim]);
                        base_ptr[d]=static_cast<OutType>(v1*cos_val - v2*sin_val);
                        base_ptr[d+half_dim]=static_cast<OutType>(v2*cos_val + v1*sin_val);
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
