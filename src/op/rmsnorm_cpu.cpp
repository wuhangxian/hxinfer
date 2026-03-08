#include "tensor/tensor.h"
#include "base/dispatch.h"
#include "cmath"
namespace hxinfer{
    void rmsnorm_tensor(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                        std::shared_ptr<Tensor>& output,float eps){
        DataType type_out=output->tensor_data_type();
        auto rmsnorm_logic=[&](const auto* ptr_in,const auto* ptr_weight,auto* ptr_out){
            using OutType=std::decay_t<decltype(*ptr_out)>;
            std::vector<int> in_shapes=input->tensor_shapes();
            int dim=in_shapes[in_shapes.size()-1];
            float scale=1.0/ dim;
            size_t row=input->tensor_total_elements()/dim;
            for(size_t i=0;i<row;i++){
                auto *curr_in=ptr_in+i*dim;
                auto *curr_out=ptr_out+i*dim;
                float sum=0;
                for(int j=0;j<dim;j++){
                    float val=static_cast<float>(curr_in[j]);
                    sum=sum+val*val;
                }
                float rms=1.0/std::sqrt(sum*scale+eps);
                for(int j=0;j<dim;j++){
                    float result_val=curr_in[j]*rms*ptr_weight[j];
                    curr_out[j]=static_cast<OutType>(result_val);
                }
            }
        };
        HXINFER_DISPATCH_ALL_TYPES(type_out,"rmsnorm",[&]{
            rmsnorm_logic(input->tensor_data_ptr<scalar_t>(),
            weight->tensor_data_ptr<scalar_t>(),
            output->tensor_data_ptr<scalar_t>());
        });
    }

}