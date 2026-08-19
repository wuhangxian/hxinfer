#ifndef HXINFER_RMSNORM_H
#define HXINFER_RMSNORM_H
#include "layer.h"
#include "op/math_ops.h"
namespace hxinfer{
    class RMSNormLayer:public Layer{
    private:
        std::shared_ptr<Tensor> weight_;
    public:
        RMSNormLayer(std::shared_ptr<Tensor> weight): Layer("RMSNorm"),weight_(weight){};
        std::shared_ptr<Tensor> get_weight() { return weight_; }
        void forward(std::shared_ptr<Tensor>& input,std::shared_ptr<Tensor>& output) override{
            rmsnorm_tensor(input,weight_,output);
        }
    };
}
#endif //HXINFER_RMSNORM_H
