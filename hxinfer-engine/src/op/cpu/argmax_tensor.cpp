#include "tensor/tensor.h"
#include "base/dispatch.h"
namespace hxinfer{
    int  argmax_cpu(const std::shared_ptr<Tensor>& input){
        DataType type_in=input->tensor_data_type();
        int global_max_index=0;
        auto argmax_logic=[&](const auto* ptr_in){
            size_t total_element=input->tensor_total_elements();
            int ans=0;
            float max_val=static_cast<float>(ptr_in[0]);
            for(size_t i=1;i<total_element;i++){

                float curr_val = static_cast<float>(ptr_in[i]);

                if (curr_val > max_val) {
                    max_val = curr_val; // 🚀 必须更新最大值！
                    ans = static_cast<int>(i);
                }
            }
            global_max_index= ans;
        };
        HXINFER_DISPATCH_ALL_TYPES(type_in,"argmax",[&](){
            argmax_logic(input->tensor_data_ptr<scalar_t>());
        });
        return global_max_index;
    }
}
