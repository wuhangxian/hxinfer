
#include "tensor/tensor.h"
namespace hxinfer{
    void rmsnorm_cuda(const std::shared_ptr<Tensor>& input,const std::shared_ptr<Tensor>& weight,
                      std::shared_ptr<Tensor>& output,float eps=1e-5){
        const float *in_ptr=input->tensor_data_ptr<float>();
        const float *w_ptr=weight->tensor_data_ptr<float>();
        float *out_ptr=output->tensor_data_ptr<float>();

        std::vector<int> shapes=input->tensor_shapes();
        int hidden_dim=shapes.back();

    }
}