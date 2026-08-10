#ifndef HXINFER_RMSNORM_H
#define HXINFER_RMSNORM_H
#include "layer.h"
#include "op/math_ops.h"
namespace hxinfer{
    class RMSNormLayer:public Layer{
    private:
        std::shared_ptr<Tensor> weight_;
        float eps_;
    public:
        RMSNormLayer(std::shared_ptr<Tensor> weight, float eps=1e-5f):
            Layer("RMSNorm"),weight_(weight),eps_(eps){};
        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output) override{
            rmsnorm_tensor(input,weight_,output,eps_);
        }
    };
}
#endif //HXINFER_RMSNORM_H
